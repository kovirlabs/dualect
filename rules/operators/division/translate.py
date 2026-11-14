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
from dualect.lift import LiftContext


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
        if (
            left.is_matrix
            and left.rows is not None
            and left.cols is not None
            and left.rows > left.cols
        ):
            # Overdetermined: the least-squares solution. The source's QR gives
            # the same answer when A has full column rank (otherwise a "basic"
            # solution, not NumPy's minimum-norm one), so that is assumed.
            ctx.assume("full column rank in a least-squares solve")
            solved = ctx.call(
                ctx.np("linalg.lstsq"),
                ctx.expr(node.left),
                ctx.expr(node.right),
                rcond=ast.Constant(None),
            )
            return ast.Subscript(value=solved, slice=ast.Constant(0))
        ctx.decline("left division by a matrix that is not known to be square or tall")
    ctx.decline(f"operator {node.op}")


# --- Python -> .m ------------------------------------------------------------
#
# Python's / is always true division: ./ (or / by a scalar). a // b floors:
# floor(a ./ b). np.linalg.solve(A, b) is A \ b; a 1-D b is a row on the .m side,
# so the solve is (A \ b.').'.


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``/``, ``//`` and ``np.linalg.solve``."""
    loc = ctx.loc(node)
    if isinstance(node, ast.Call):
        if len(node.args) != 2 or node.keywords:
            ctx.decline("np.linalg.solve with these arguments")
        a_node, b_node = node.args
        if not ctx.info(a_node).is_matrix:
            ctx.decline("solving with a value not known to be a matrix")
        a = ctx.expr(a_node)
        b_info = ctx.info(b_node)
        if b_info.is_matrix:
            return ctx.binop("mldivide", a, ctx.expr(b_node), loc=loc)
        if b_info.is_vector or isinstance(b_node, ast.List):
            b = ctx.module_for("syntax/array-literals").lift_items(b_node, ctx)
            return ctx.transpose(ctx.binop("mldivide", a, ctx.transpose(b), loc=loc))
        ctx.decline("solving for a right-hand side of unknown dimensions")
    assert isinstance(node, ast.BinOp)
    left, right = ctx.expr(node.left), ctx.expr(node.right)
    by_scalar = ctx.info(node.right).is_scalar
    op = "mrdivide" if by_scalar else "rdivide"
    if not (by_scalar or ctx.info(node.right).is_array):
        ctx.assume("element-wise division")
    quotient = ctx.binop(op, left, right, loc=loc)
    if isinstance(node.op, ast.Div):
        return quotient
    if isinstance(node.op, ast.FloorDiv):
        divisor = ctx.info(node.right).value
        if divisor == 0:
            ctx.decline("floor division by zero")
        if divisor is None:
            ctx.assume("nonzero divisor in //")
        return ctx.call("floor", quotient, loc=loc)
    ctx.decline(f"operator {type(node.op).__name__}")
