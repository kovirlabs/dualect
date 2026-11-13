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
