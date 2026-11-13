# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for operators/division: ``./``, ``/``, ``\\`` and ``.\\``.

Element-wise division and division by a scalar are Python's ``/``. Matrix
divisions become linear solves: ``A \\ b`` is ``np.linalg.solve(A, b)`` and
``B / A`` is ``np.linalg.solve(A.T, B.T).T``. A divisor of unknown shape is
assumed to be a scalar (docs/design.md section 5.2).
"""

import ast

from dualect import ir
from dualect.emit import Context


def _transpose(expr: ast.expr) -> ast.expr:
    return ast.Attribute(value=expr, attr="T")


def translate(node: ir.BinOp, ctx: Context) -> ast.expr:
    """One division."""
    arithmetic = ctx.module("operators/arithmetic")
    left, right = ctx.info(node.left), ctx.info(node.right)
    if node.op == "rdivide":
        a, b = arithmetic.expand(node, ctx)
        return ast.BinOp(left=a, op=ast.Div(), right=b)
    if node.op == "ldivide":
        a, b = arithmetic.expand(node, ctx)
        return ast.BinOp(left=b, op=ast.Div(), right=a)
    if node.op == "mrdivide":
        if right.is_scalar:
            return ast.BinOp(left=ctx.expr(node.left), op=ast.Div(), right=ctx.expr(node.right))
        if right.is_matrix and (left.is_matrix or left.kind.value == "row"):
            solved = ctx.call(
                ctx.np("linalg.solve"),
                _transpose(ctx.expr(node.right)),
                _transpose(ctx.expr(node.left)),
            )
            return _transpose(solved) if left.is_matrix else solved
        if right.kind.value in ("unknown", "array"):
            ctx.assume(f"scalar divisor {ast.unparse(ctx.expr(node.right))}")
            return ast.BinOp(left=ctx.expr(node.left), op=ast.Div(), right=ctx.expr(node.right))
        ctx.decline("matrix right division with these shapes")
    if node.op == "mldivide":
        if left.is_scalar:
            return ast.BinOp(left=ctx.expr(node.right), op=ast.Div(), right=ctx.expr(node.left))
        if left.is_matrix and left.rows is not None and left.rows == left.cols:
            return ctx.call(ctx.np("linalg.solve"), ctx.expr(node.left), ctx.expr(node.right))
        ctx.decline("left division by a matrix that is not known to be square")
    ctx.decline(f"operator {node.op}")
