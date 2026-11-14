# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for types/char-strings: string literals, ``[a, b]`` and ``strcmp``.

A literal becomes a Python ``str`` (its escapes were resolved when parsing,
per dialect); bracket concatenation of strings is ``+``; ``strcmp(a, b)`` is
``a == b`` when both are strings.
"""

import ast
from functools import reduce

from dualect import ir
from dualect.emit import Context
from dualect.infer import Kind
from dualect.lift import LiftContext
from dualect.pyinfer import PyKind


def translate(node: ir.Str | ir.Matrix | ir.Call, ctx: Context) -> ast.expr:
    """A string literal, a string concatenation or ``strcmp``."""
    if isinstance(node, ir.Str):
        return ast.Constant(node.value)
    if isinstance(node, ir.Matrix):
        if len(node.rows) != 1:
            ctx.decline("stacking strings into a char matrix")
        parts = [ctx.expr(item) for item in node.rows[0]]
        return reduce(lambda a, b: ast.BinOp(left=a, op=ast.Add(), right=b), parts)
    if node.func == "strcmp" and len(node.args) == 2:
        if all(ctx.info(arg).kind is Kind.STRING for arg in node.args):
            a, b = (ctx.expr(arg) for arg in node.args)
            return ast.Compare(left=a, ops=[ast.Eq()], comparators=[b])
        ctx.decline("strcmp on values not known to be strings")
    ctx.decline(f"call {node.func}")


# --- Python -> .m ------------------------------------------------------------
#
# A str literal is a single-quoted char array (quotes doubled); a + b of two
# strings is [a, b]; a == b of two strings compares the whole text, strcmp(a,
# b) (== on char arrays compares character by character). Control characters
# would need escapes the two dialects read differently, so they are written
# with char(): "a\tb" is ['a', char(9), 'b'].


def lift(node: ast.AST, ctx: LiftContext) -> ir.Expr:
    """A string literal."""
    assert isinstance(node, ast.Constant) and isinstance(node.value, str)
    loc = ctx.loc(node)
    text = node.value
    if not any(ord(c) < 32 or ord(c) == 127 for c in text):
        return ir.Str(loc=loc, value=text, quote="'")
    # ['a', char(9), 'b']: the same characters in either dialect, no escapes.
    parts: list[ir.Expr] = []
    run = ""
    for char in text:
        if ord(char) < 32 or ord(char) == 127:
            if run:
                parts.append(ir.Str(loc=loc, value=run, quote="'"))
                run = ""
            parts.append(ctx.call("char", ctx.num(ord(char)), loc=loc))
        else:
            run += char
    if run:
        parts.append(ir.Str(loc=loc, value=run, quote="'"))
    return ir.Matrix(loc=loc, rows=[parts])


def lift_concat(node: ast.BinOp, ctx: LiftContext) -> ir.Expr:
    """``a + b`` of two strings."""
    if not isinstance(node.op, ast.Add):
        ctx.decline("this operation on strings")
    if not all(ctx.info(x).kind is PyKind.STRING for x in (node.left, node.right)):
        ctx.decline("adding a string and a value that is not a string")
    return ir.Matrix(loc=ctx.loc(node), rows=[[ctx.expr(node.left), ctx.expr(node.right)]])


def lift_compare(node: ast.Compare, ctx: LiftContext) -> ir.Expr:
    """``a == b`` (or ``!=``, negated by the caller) of two strings."""
    left, right = node.left, node.comparators[0]
    if not isinstance(node.ops[0], ast.Eq | ast.NotEq):
        ctx.decline("ordering strings")
    if not all(ctx.info(x).kind is PyKind.STRING for x in (left, right)):
        ctx.decline("comparing a string with a value that is not a string")
    return ctx.call("strcmp", ctx.expr(left), ctx.expr(right), loc=ctx.loc(node))
