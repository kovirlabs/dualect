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
