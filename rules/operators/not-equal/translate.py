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
from dualect.lift import LiftContext


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


# --- Python -> .m ------------------------------------------------------------
#
# != is ~=. `not x` is ~x for a scalar (Python raises for an array of several
# elements). ~m is ~m only on booleans: on integers Python's ~ is bitwise
# (~5 is -6), which is declined. np.logical_not(x) is ~x.


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``!=``, ``not``, ``~`` and ``np.logical_not``."""
    loc = ctx.loc(node)
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            ctx.decline("a chained comparison")
        left, right = node.left, node.comparators[0]
        if ctx.info(left).kind.value == "string" or ctx.info(right).kind.value == "string":
            equal = ctx.module_for("types/char-strings").lift_compare(node, ctx)
            return ir.UnOp(loc=loc, op="not", operand=equal)
        return ctx.binop("ne", ctx.expr(left), ctx.expr(right), loc=loc)
    if isinstance(node, ast.Call):
        if len(node.args) != 1 or node.keywords:
            ctx.decline("np.logical_not with these arguments")
        operand = ctx.module_for("syntax/array-literals").lift_items(node.args[0], ctx)
        return ir.UnOp(loc=loc, op="not", operand=operand)
    assert isinstance(node, ast.UnaryOp)
    info = ctx.info(node.operand)
    if isinstance(node.op, ast.Not):
        if info.is_array:
            ctx.decline("`not` of an array")
        return ir.UnOp(loc=loc, op="not", operand=ctx.expr(node.operand))
    if not info.is_bool:
        ctx.decline("~ of a value not known to be boolean (bitwise on integers)")
    return ir.UnOp(loc=loc, op="not", operand=ctx.expr(node.operand))
