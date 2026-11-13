# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for operators/not-equal: ``~=`` / ``!=`` and ``~x`` / ``!x``.

Not-equal is an element-wise comparison (shared with operators/arithmetic).
Logical NOT depends on the operand: ``not x`` for a scalar, ``~x`` for a
logical array, and ``np.logical_not(x)`` otherwise, because Python's ``~`` on
integers is bitwise.
"""

import ast

from dualect import ir
from dualect.emit import Context


def translate(node: ir.BinOp | ir.UnOp, ctx: Context) -> ast.expr:
    """``a ~= b`` or ``~x``."""
    if isinstance(node, ir.BinOp):
        a, b = ctx.module("operators/arithmetic").expand(node, ctx)
        return ast.Compare(left=a, ops=[ast.NotEq()], comparators=[b])
    info = ctx.info(node.operand)
    operand = ctx.expr(node.operand)
    if info.is_scalar:
        return ast.UnaryOp(op=ast.Not(), operand=operand)
    if info.is_logical and info.is_array:
        return ast.UnaryOp(op=ast.Invert(), operand=operand)
    return ctx.call(ctx.np("logical_not"), operand)
