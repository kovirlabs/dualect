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
from dualect.lift import LiftContext


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


# --- Python -> .m ------------------------------------------------------------
#
# A.ravel(order="F") walks the columns: A(:).', a row like every 1-D array.
# The default (row-major) A.ravel() is reshape(A.', 1, []). v.reshape(shape,
# order="F") fills column by column, as reshape does on the .m side; the
# default row-major order fills rows, which is declined.


def _order(node: ast.Call) -> str:
    for keyword in node.keywords:
        if keyword.arg == "order" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return "C"


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``.ravel()``, ``.flatten()`` and ``.reshape(...)``."""
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    loc = ctx.loc(node)
    method = node.func.attr
    receiver = node.func.value
    info = ctx.info(receiver)
    if any(k.arg != "order" for k in node.keywords):
        ctx.decline(f"{method} with these options")
    order = _order(node)
    if method in ("ravel", "flatten"):
        if node.args:
            ctx.decline(f"{method} with arguments")
        if info.is_vector:
            return ctx.expr(receiver)
        if not info.is_matrix:
            ctx.decline(f"{method} of a value whose dimensions are unknown")
        value = ctx.expr(receiver)
        if order == "F":
            return ctx.transpose(ir.Index(loc=loc, target=value, args=[ir.Colon(loc=loc)]))
        if order == "C":
            empty = ir.Matrix(loc=loc, rows=[])
            return ctx.call("reshape", ctx.transpose(value), ctx.num(1), empty, loc=loc)
        ctx.decline(f"order={order!r}")
    if method == "reshape":
        if order != "F":
            ctx.decline("a row-major reshape (the .m reshape fills columns)")
        if len(node.args) == 1:
            shape = node.args[0]
            if isinstance(shape, ast.Attribute) and shape.attr == "shape":
                if not ctx.info(shape.value).is_matrix:
                    ctx.decline("reshaping to the shape of a value that is not 2-D")
                dims: list[ir.Expr] = [ctx.call("size", ctx.expr(shape.value), loc=loc)]
            elif isinstance(shape, ast.Tuple) and len(shape.elts) == 2:
                dims = [ctx.expr(e) for e in shape.elts]
            else:
                ctx.decline("reshape to this shape")
        elif len(node.args) == 2:
            dims = [ctx.expr(e) for e in node.args]
        else:
            ctx.decline("reshape to this shape")
        return ctx.call("reshape", ctx.expr(receiver), *dims, loc=loc)
    ctx.decline(f"method {method}")
