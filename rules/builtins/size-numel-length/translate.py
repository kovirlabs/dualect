# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/size-numel-length.

``numel`` is ``x.size`` (``len`` for a string); ``size(A)`` and ``size(A, d)``
read ``A.shape`` for a matrix. A vector is 1-D in NumPy, so its sizes are
spelled out from its known orientation (``size(v, 1)`` of a row is 1).
``length`` is the largest dimension.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.infer import Info, Kind


def _count(expr: ast.expr, info: Info, ctx: Context) -> ast.expr:
    if info.kind is Kind.STRING:
        return ctx.call(ast.Name("len"), expr)
    if info.is_array:
        return ast.Attribute(value=expr, attr="size")
    if info.is_scalar:
        return ast.Constant(1)
    ctx.assume(f"{ast.unparse(expr)} is numeric")
    return ctx.call(ctx.np("size"), expr)


def _dims(info: Info, expr: ast.expr, ctx: Context) -> list[ast.expr]:
    """Rows and columns of a value, as source sizes."""
    if info.is_matrix or info.kind is Kind.EMPTY:
        shape = ast.Attribute(value=expr, attr="shape")
        if info.kind is Kind.EMPTY:
            return [ast.Constant(0), ast.Constant(0)]
        return [ast.Subscript(value=shape, slice=ast.Constant(k)) for k in (0, 1)]
    if info.kind is Kind.ROW or info.kind is Kind.STRING:
        return [ast.Constant(1), _count(expr, info, ctx)]
    if info.kind is Kind.COL:
        return [_count(expr, info, ctx), ast.Constant(1)]
    if info.is_scalar:
        return [ast.Constant(1), ast.Constant(1)]
    ctx.decline("size of a value whose shape is unknown")


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``size``, ``numel`` or ``length``."""
    if not node.args:
        ctx.decline(f"{node.func} without arguments")
    info = ctx.info(node.args[0])
    expr = ctx.expr(node.args[0])
    if node.func == "numel" and len(node.args) == 1:
        return _count(expr, info, ctx)
    if node.func == "length" and len(node.args) == 1:
        if info.is_matrix or info.kind is Kind.EMPTY:
            return ctx.call(ast.Name("max"), ast.Attribute(value=expr, attr="shape"))
        if info.is_vector or info.kind is Kind.STRING or info.is_scalar:
            return _count(expr, info, ctx)
        ctx.decline("length of a value whose shape is unknown")
    if node.func == "size":
        if len(node.args) == 1:
            if info.is_matrix:
                return ast.Attribute(value=expr, attr="shape")
            return ast.Tuple(elts=_dims(info, expr, ctx))
        if len(node.args) == 2:
            dim = ctx.info(node.args[1]).value
            if not isinstance(dim, int | float) or not float(dim).is_integer() or dim < 1:
                ctx.decline("size with a computed dimension")
            if dim > 2:
                return ast.Constant(1)
            if info.is_matrix:
                shape = ast.Attribute(value=expr, attr="shape")
                return ast.Subscript(value=shape, slice=ast.Constant(int(dim) - 1))
            return _dims(info, expr, ctx)[int(dim) - 1]
    ctx.decline(f"{node.func} with these arguments")


def translate_multi(stmt: ir.Assign, ctx: Context) -> list[Line]:
    """``[r, c] = size(A)``."""
    call = stmt.value
    assert isinstance(call, ir.Call)
    if call.func != "size" or len(call.args) != 1 or len(stmt.targets) != 2:
        ctx.decline(f"{call.func} with several outputs")
    info = ctx.info(call.args[0])
    expr = ctx.expr(call.args[0])
    names = [t.id if isinstance(t, ir.Name) else "_" for t in stmt.targets]
    if info.is_matrix:
        rhs: ast.expr = ast.Attribute(value=expr, attr="shape")
    else:
        rhs = ast.Tuple(elts=_dims(info, expr, ctx))
    return [ctx.line(f"{', '.join(names)} = {ast.unparse(rhs)}")]
