# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/filter: ``filter(b, a, x)`` is ``scipy.signal.lfilter``.

The arguments keep their order and meaning (``lfilter`` rejects integer
input, so all-integer data is converted to floats). A vector (1-D in NumPy) is
filtered as it is; a matrix is filtered down its columns, ``axis=0``, because
``lfilter`` works along the last axis. The forms with initial conditions, a
dimension or a second output are declined: they change the call's shape.

From Python, ``lfilter`` imported from ``scipy.signal`` becomes ``filter``;
on a 2-D array its default axis (the rows) is ``filter(b, a, X, [], 2)``.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.lift import LiftContext


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``filter(b, a, x)``."""
    if len(node.args) != 3:
        ctx.decline("filter with initial conditions or a dimension")
    b, a, x = node.args
    for coefficients in (b, a):
        info = ctx.info(coefficients)
        if not (info.is_scalar or info.is_vector):
            ctx.decline("filter coefficients that are not a vector")
    data = ctx.info(x)
    if "lfilter" in ctx.scope.variables:
        ctx.decline("a variable named lfilter would hide scipy's lfilter")
    ctx.imports.add("from scipy.signal import lfilter")
    args = [ctx.arg(b), ctx.arg(a), ctx.arg(x)]
    if all(ctx.info(arg).int_dtype for arg in node.args):
        # lfilter rejects integer input; the source computes in doubles.
        data_expr = args[2]
        if isinstance(data_expr, ast.Name | ast.Attribute | ast.Subscript):
            args[2] = ctx.method(data_expr, "astype", ast.Name("float"))
        else:
            args[2] = ctx.call(ctx.np("asarray"), data_expr, dtype=ast.Name("float"))
    if data.is_matrix:
        # lfilter runs along the last axis (rows); filter down the columns.
        return ctx.call(ast.Name("lfilter"), *args, axis=ast.Constant(0))
    if not data.is_vector or data.kind.value == "empty":
        # lfilter needs at least one dimension: a scalar is declined too.
        ctx.decline("filter of a value that is not known to be a vector or a matrix")
    return ctx.call(ast.Name("lfilter"), *args)


# --- Python -> .m ------------------------------------------------------------


def _imported_from_scipy(ctx: LiftContext) -> bool:
    """``lfilter`` is the name ``from scipy.signal import lfilter`` binds."""
    for stmt in ctx.module.body:
        if (
            isinstance(stmt, ast.ImportFrom)
            and stmt.module == "scipy.signal"
            and any(alias.name == "lfilter" and alias.asname is None for alias in stmt.names)
        ):
            return True
    return False


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``lfilter(b, a, x)`` (and ``axis=``)."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if not _imported_from_scipy(ctx):
        ctx.decline("lfilter that is not scipy.signal's")
    if len(node.args) != 3:
        ctx.decline("lfilter with these arguments")
    axis: int | None = None
    for keyword in node.keywords:
        value = ctx.info(keyword.value).value
        if keyword.arg != "axis" or not isinstance(value, int):
            ctx.decline(f"lfilter with {keyword.arg}=")
        axis = value
    b, a, x = node.args
    for coefficients in (b, a):
        info = ctx.info(coefficients)
        if not (info.is_scalar or info.is_vector or isinstance(coefficients, ast.List)):
            ctx.decline("lfilter coefficients that are not 1-D")
    literals = ctx.module_for("syntax/array-literals")
    args = [literals.lift_items(arg, ctx) for arg in node.args]
    data = ctx.info(x)
    if data.is_matrix:
        if axis in (0, -2):
            return ctx.call("filter", *args, loc=loc)
        if axis in (None, 1, -1):
            empty = ir.Matrix(loc=loc, rows=[])
            return ctx.call("filter", *args, empty, ctx.num(2), loc=loc)
        ctx.decline(f"lfilter along axis {axis}")
    if not (data.is_vector or isinstance(x, ast.List)) or axis not in (None, 0, -1):
        ctx.decline("lfilter of a value that is not known to be 1-D or 2-D")
    return ctx.call("filter", *args, loc=loc)
