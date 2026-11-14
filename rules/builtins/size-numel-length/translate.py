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
from dualect.lift import LiftContext


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


# --- Python -> .m ------------------------------------------------------------
#
# x.size and np.size(x) are numel(x). A.shape of a 2-D array is size(A) and
# A.shape[k] is size(A, k + 1); a 1-D array is a row on the .m side, so its
# only dimension is numel(x). len() is the length of the first axis:
# size(A, 1) for a 2-D array, numel for a 1-D array or a string.


def lift_shape(node: ast.Attribute, ctx: LiftContext) -> ir.Expr:
    """``A.shape`` as a value."""
    loc = ctx.loc(node)
    info = ctx.info(node.value)
    value = ctx.expr(node.value)
    if info.is_matrix:
        return ctx.call("size", value, loc=loc)
    if info.is_vector:
        return ctx.call("numel", value, loc=loc)
    ctx.decline(".shape of a value whose dimensions are unknown")


def lift_shape_index(node: ast.Subscript, ctx: LiftContext) -> ir.Expr:
    """``A.shape[k]`` with a literal ``k``."""
    shape = node.value
    assert isinstance(shape, ast.Attribute)
    loc = ctx.loc(node)
    info = ctx.info(shape.value)
    k = node.slice
    axis = None
    if isinstance(k, ast.Constant) and type(k.value) is int:
        axis = k.value
    elif (
        isinstance(k, ast.UnaryOp)
        and isinstance(k.op, ast.USub)
        and isinstance(k.operand, ast.Constant)
    ):
        axis = -k.operand.value if type(k.operand.value) is int else None
    if axis is None:
        ctx.decline("a computed axis of .shape")
    value = ctx.expr(shape.value)
    if info.is_vector and axis in (0, -1):
        return ctx.call("numel", value, loc=loc)
    if info.is_matrix and axis in (0, 1, -1, -2):
        return ctx.call("size", value, ctx.num(axis % 2 + 1), loc=loc)
    ctx.decline(".shape[k] of a value whose dimensions are unknown")


def lift_multi(stmt: ast.Assign, ctx: LiftContext) -> list[ir.Stmt]:
    """``r, c = A.shape``."""
    target = stmt.targets[0]
    value = stmt.value
    assert isinstance(target, ast.Tuple | ast.List) and isinstance(value, ast.Attribute)
    if not ctx.info(value.value).is_matrix or len(target.elts) != 2:
        ctx.decline("unpacking the shape of a value that is not 2-D")
    names: list[ir.Expr] = []
    for element in target.elts:
        if not isinstance(element, ast.Name):
            ctx.decline("unpacking into an element")
        names.append(ir.Name(loc=ctx.loc(element), id=element.id))
    call = ctx.call("size", ctx.expr(value.value), loc=ctx.loc(value))
    return [ir.Assign(loc=ctx.loc(stmt), targets=names, value=call)]


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``x.size``, ``A.shape``, ``len(x)``, ``np.size(x)``."""
    loc = ctx.loc(node)
    if isinstance(node, ast.Attribute):
        if node.attr == "shape":
            return lift_shape(node, ctx)
        info = ctx.info(node.value)
        if not (info.is_array or info.is_scalar):
            ctx.assume(f"{ast.unparse(node.value)} is an array")
        return ctx.call("numel", ctx.expr(node.value), loc=loc)
    assert isinstance(node, ast.Call)
    if len(node.args) != 1 or node.keywords:
        ctx.decline("this size query")
    arg = node.args[0]
    info = ctx.info(arg)
    if ctx.tag(node) == "py:call:np.size":
        return ctx.call("numel", ctx.expr(arg), loc=loc)
    if info.is_matrix:
        return ctx.call("size", ctx.expr(arg), ctx.num(1), loc=loc)
    if info.is_vector or info.kind.value == "string":
        return ctx.call("numel", ctx.expr(arg), loc=loc)
    ctx.decline("len() of a value that is not a known array or string")
