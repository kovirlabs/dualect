# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for control-flow/if-elseif.

The structure maps directly. Conditions are where the languages differ: an
array condition holds only when it is non-empty and all its elements are
nonzero, which NumPy will not guess, so :func:`condition` (shared with the
loops) spells it out as ``x.size > 0 and x.all()``. A condition of unknown
shape is left as is: Python raises for a multi-element array, so a wrong guess
fails loudly.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.infer import Kind


def condition(node: ir.Expr, ctx: Context) -> ast.expr:
    """A condition with the source's truth rule for arrays."""
    info = ctx.info(node)
    expr = ctx.expr(node)
    if info.kind is Kind.STRING:
        ctx.decline("a char array as a condition")
    if not info.is_array:
        return expr
    if isinstance(expr, ast.Name):
        size = ast.Attribute(value=expr, attr="size")
        nonempty = ast.Compare(left=size, ops=[ast.Gt()], comparators=[ast.Constant(0)])
        return ast.BoolOp(op=ast.And(), values=[nonempty, ctx.method(expr, "all")])
    if info.numel is not None and info.numel > 0:
        return ctx.method(expr, "all")
    ctx.decline("an array condition that is neither a variable nor known to be non-empty")


def translate(stmt: ir.If, ctx: Context) -> list[Line]:
    """``if`` / ``elseif`` / ``else``."""
    lines: list[Line] = []
    for k, (cond, body) in enumerate(stmt.branches):
        keyword = "if" if k == 0 else "elif"
        lines.append(ctx.line(f"{keyword} {ast.unparse(condition(cond, ctx))}:"))
        lines.extend(ctx.block(body))
    if stmt.orelse is not None:
        lines.append(ctx.line("else:"))
        lines.extend(ctx.block(stmt.orelse))
    return lines
