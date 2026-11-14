# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for operators/matrix-multiply: ``*`` and ``.*``.

``.*`` is NumPy's ``*``. ``*`` is a scalar product when either side is a
scalar, and otherwise the matrix product ``@``, with the 1-D cases spelled out:
row times column is the inner product ``u @ v``, column times row the outer
product ``np.outer(u, v)``. With an operand of unknown shape the product is
assumed to be a matrix product (docs/design.md section 5.2); a vector whose
orientation is not known (row at one call site, column at another) is declined,
since the product means an inner product for one and an outer for the other.
Two logical operands are multiplied as numbers (see operators/arithmetic).
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Kind
from dualect.lift import LiftContext


def translate(node: ir.BinOp, ctx: Context) -> ast.expr:
    """One multiplication."""
    left, right = ctx.info(node.left), ctx.info(node.right)
    if node.op == "times":
        arithmetic = ctx.module("operators/arithmetic")
        a, b = arithmetic.expand(node, ctx)
        return ast.BinOp(left=arithmetic.numeric(a, left, right, ctx), op=ast.Mult(), right=b)
    if node.op != "mtimes":
        ctx.decline(f"operator {node.op}")
    a, b = ctx.expr(node.left), ctx.expr(node.right)
    a = ctx.module("operators/arithmetic").numeric(a, left, right, ctx)
    if left.is_scalar or right.is_scalar:
        return ast.BinOp(left=a, op=ast.Mult(), right=b)
    if left.kind is Kind.COL and right.kind is Kind.ROW:
        return ctx.call(ctx.np("outer"), a, b)
    known = {Kind.ROW, Kind.COL, Kind.MATRIX}
    if left.kind in known and right.kind in known:
        if left.kind is Kind.COL or right.kind is Kind.ROW:
            ctx.decline("a matrix product with these shapes")
        return ast.BinOp(left=a, op=ast.MatMult(), right=b)
    if left.kind is Kind.STRING or right.kind is Kind.STRING:
        ctx.decline("multiplying char arrays")
    if Kind.VECTOR in (left.kind, right.kind):
        # A vector whose orientation differs between uses (a function called
        # with a row and with a column): `u * v` is an inner product for one
        # and an outer product for the other, and `@` can only be one of them.
        ctx.decline("a product with a vector whose orientation is not known")
    ctx.assume("matrix multiply")
    return ast.BinOp(left=a, op=ast.MatMult(), right=b)


# --- Python -> .m ------------------------------------------------------------
#
# NumPy's * is element-wise (.*), @ the matrix product. A 1-D array is a row on
# the .m side, so @ with a 1-D operand needs a transpose: u @ v is u * v.',
# A @ v is (A * v.').', v @ A is v * A; np.outer(u, v) is u.' * v.


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``*``, ``@`` and ``np.outer``."""
    loc = ctx.loc(node)
    if isinstance(node, ast.Call):
        if len(node.args) != 2 or node.keywords:
            ctx.decline("np.outer with these arguments")
        u, v = node.args
        if not all(ctx.info(x).is_vector or isinstance(x, ast.List) for x in (u, v)):
            ctx.decline("np.outer of values not known to be 1-D (it flattens them)")
        literals = ctx.module_for("syntax/array-literals")
        return ctx.binop(
            "mtimes",
            ctx.transpose(literals.lift_items(u, ctx)),
            literals.lift_items(v, ctx),
            loc=loc,
        )
    assert isinstance(node, ast.BinOp)
    a, b = ctx.info(node.left), ctx.info(node.right)
    if isinstance(node.op, ast.Mult):
        if a.kind.value == "string" or b.kind.value == "string":
            ctx.decline("repeating a string")
        if ctx.module_for("operators/arithmetic").both_bool_arrays(node.left, node.right, ctx):
            ctx.decline("multiplying boolean arrays (NumPy keeps bool * bool as a logical AND)")
        left, right = ctx.expr(node.left), ctx.expr(node.right)
        op = "mtimes" if a.is_scalar or b.is_scalar else "times"
        if not (a.is_scalar or b.is_scalar or a.is_array or b.is_array):
            ctx.assume("element-wise product")
        return ctx.binop(op, left, right, loc=loc)
    if not isinstance(node.op, ast.MatMult):
        ctx.decline(f"operator {type(node.op).__name__}")
    left, right = ctx.expr(node.left), ctx.expr(node.right)
    if a.is_matrix and b.is_matrix:
        return ctx.binop("mtimes", left, right, loc=loc)
    if a.is_vector and b.is_vector:
        return ctx.binop("mtimes", left, ctx.transpose(right), loc=loc)
    if a.is_matrix and b.is_vector:
        return ctx.transpose(ctx.binop("mtimes", left, ctx.transpose(right), loc=loc))
    if a.is_vector and b.is_matrix:
        return ctx.binop("mtimes", left, right, loc=loc)
    ctx.decline("@ with an operand whose dimensions are unknown")
