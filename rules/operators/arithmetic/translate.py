# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for operators/arithmetic: ``+``, ``-``, unary minus, comparisons.

These are element-wise in both languages. The one difference is implicit
expansion against a *column*: NumPy vectors are 1-D, so a column operand that
must stretch across columns gets an explicit axis, ``c[:, np.newaxis]``.
:func:`expand` is shared with the other element-wise operators.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Kind

_BINARY: dict[str, ast.operator] = {"plus": ast.Add(), "minus": ast.Sub()}
_COMPARE: dict[str, ast.cmpop] = {
    "eq": ast.Eq(),
    "ne": ast.NotEq(),
    "lt": ast.Lt(),
    "le": ast.LtE(),
    "gt": ast.Gt(),
    "ge": ast.GtE(),
}


def _column(expr: ast.expr, ctx: Context) -> ast.expr:
    """``expr[:, np.newaxis]``: a 1-D vector as a column that broadcasts across columns."""
    return ast.Subscript(value=expr, slice=ast.Tuple(elts=[ast.Slice(), ctx.np("newaxis")]))


def expand(node: ir.BinOp, ctx: Context) -> tuple[ast.expr, ast.expr]:
    """Both operands of an element-wise operator, with columns given their axis.

    A column against a row (or a matrix) needs ``[:, np.newaxis]``; two vectors
    of unknown orientation are assumed to have the same one.
    """
    left, right = ctx.info(node.left), ctx.info(node.right)
    a, b = ctx.expr(node.left), ctx.expr(node.right)
    if left.is_scalar or right.is_scalar:
        return a, b
    kinds = {left.kind, right.kind}
    if kinds == {Kind.ROW, Kind.COL} or kinds == {Kind.MATRIX, Kind.COL}:
        if left.kind is Kind.COL:
            a = _column(a, ctx)
        else:
            b = _column(b, ctx)
        return a, b
    if left.kind is right.kind and left.kind in (Kind.ROW, Kind.COL, Kind.MATRIX, Kind.EMPTY):
        return a, b
    if kinds <= {Kind.MATRIX, Kind.ROW, Kind.EMPTY}:
        return a, b
    if left.kind is Kind.STRING or right.kind is Kind.STRING:
        ctx.decline("arithmetic or comparison on char arrays")
    ctx.assume("operands of matching shape")
    return a, b


def translate(node: ir.BinOp | ir.UnOp, ctx: Context) -> ast.expr:
    """``a + b``, ``a - b``, ``-a``, ``+a`` and the comparisons."""
    if isinstance(node, ir.UnOp):
        if node.op == "uplus":
            return ctx.expr(node.operand)
        if node.op == "uminus":
            return ast.UnaryOp(op=ast.USub(), operand=ctx.expr(node.operand))
        ctx.decline(f"unary {node.op}")
    a, b = expand(node, ctx)
    if node.op in _BINARY:
        return ast.BinOp(left=a, op=_BINARY[node.op], right=b)
    if node.op in _COMPARE:
        return ast.Compare(left=a, ops=[_COMPARE[node.op]], comparators=[b])
    ctx.decline(f"operator {node.op}")
