# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for indexing/colon-all: ``A(:)`` and ``A(:) = v``.

Reading ``A(:)`` walks the columns first: ``A.ravel(order="F")`` (a vector is
already 1-D). Assigning ``A(:) = v`` keeps ``A``'s shape: a scalar fills it,
an array is laid out column by column with
``v.reshape(A.shape, order="F")``. ``sum(A(:))`` and friends are handled by
builtins/sum-mean-max as a plain ``A.sum()``.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line


def colon_all(target: ir.Expr, ctx: Context, *, store: bool) -> ast.expr:
    """``A(:)`` as a value (or, with ``store``, as the target ``A[:]``)."""
    info = ctx.info(target)
    expr = ctx.expr(target)
    if store:
        return ast.Subscript(value=expr, slice=ast.Slice())
    if info.is_matrix:
        return ctx.method(expr, "ravel", order=ast.Constant("F"))
    if info.is_vector:
        return expr
    ctx.decline("x(:) of a value whose shape is unknown")


def translate(node: ir.Index, ctx: Context) -> ast.expr:
    """``A(:)`` read as a value."""
    return colon_all(node.target, ctx, store=False)


def translate_store(stmt: ir.Assign, ctx: Context) -> list[Line]:
    """``A(:) = v``: fill ``A`` in column-major order, keeping its shape."""
    target = stmt.targets[0]
    assert isinstance(target, ir.Index)
    info = ctx.info(target.target)
    value = ctx.info(stmt.value)
    name = ctx.expr(target.target)
    lhs = colon_all(target.target, ctx, store=True)
    if info.int_dtype and not (value.integer and value.is_real):
        ctx.decline("storing possibly fractional values into an integer array")
    rhs = ctx.expr(stmt.value)
    if value.is_scalar or info.is_vector:
        return [ctx.line(f"{ast.unparse(lhs)} = {ast.unparse(rhs)}")]
    if info.is_matrix:
        shape = ast.Attribute(value=name, attr="shape")
        laid_out = ctx.method(rhs, "reshape", shape, order=ast.Constant("F"))
        return [ctx.line(f"{ast.unparse(lhs)} = {ast.unparse(laid_out)}")]
    ctx.decline("A(:) = v with A of unknown shape")
