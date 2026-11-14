# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/trapz: ``np.trapezoid``.

``trapz(y)`` integrates with unit spacing; ``trapz(x, y)`` takes the sample
points first, while ``np.trapezoid(y, x)`` takes them second. A matrix is
integrated down its columns (``axis=0``); NumPy's default axis is the last.
A dimension argument, a scalar and an empty input are declined. (NumPy 2
renamed ``np.trapz`` to ``np.trapezoid``.)
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Info, Kind, array, scalar
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind, as_array


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``trapz(y)`` or ``trapz(x, y)``."""
    if len(node.args) not in (1, 2):
        ctx.decline("trapz with a dimension")
    y = node.args[-1]
    info = ctx.info(y)
    if not info.is_real:
        ctx.decline("trapz of complex values or values of unknown type")
    args = [ctx.arg(y)]
    if len(node.args) == 2:
        points = ctx.info(node.args[0])
        if not points.is_vector or points.kind is Kind.EMPTY:
            ctx.decline("trapz with sample points that are not a vector")
        args.append(ctx.arg(node.args[0]))
    if info.is_matrix:
        return ctx.call(ctx.np("trapezoid"), *args, axis=ast.Constant(0))
    if not info.is_vector or info.kind is Kind.EMPTY:
        ctx.decline("trapz of a scalar, an empty array or a value of unknown shape")
    return ctx.call(ctx.np("trapezoid"), *args)


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """A scalar for a vector, a row for a matrix."""
    if not args:
        return None
    y = args[-1]
    finite = all(a.finite for a in args)
    if y.is_vector:
        return scalar(finite=finite)
    if y.is_matrix:
        return array(1, y.cols, kind=Kind.ROW, finite=finite)
    return None


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.trapezoid(y[, x][, axis=...])``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if not 1 <= len(node.args) <= 2:
        ctx.decline("np.trapezoid with these arguments")
    y = node.args[0]
    x = node.args[1] if len(node.args) > 1 else None
    axis = None
    for keyword in node.keywords:
        if keyword.arg == "x" and x is None:
            x = keyword.value
        elif keyword.arg == "axis":
            axis = ctx.info(keyword.value).value
        else:
            ctx.decline(f"np.trapezoid with {keyword.arg}=")
    literals = ctx.module_for("syntax/array-literals")
    info = ctx.info(y)
    if info.dtype in (DType.COMPLEX, DType.UNKNOWN) and not isinstance(y, ast.List):
        ctx.decline("np.trapezoid of complex data or data of unknown type")
    args = [literals.lift_items(y, ctx)]
    if x is not None:
        if not (ctx.info(x).is_vector or isinstance(x, ast.List)):
            ctx.decline("np.trapezoid with sample points that are not 1-D")
        args.insert(0, literals.lift_items(x, ctx))
    if info.is_vector or isinstance(y, ast.List):
        if axis not in (None, 0, -1):
            ctx.decline("np.trapezoid of a 1-D array along axis 1")
        return ctx.call("trapz", *args, loc=loc)
    if not info.is_matrix or axis not in (0, -2, None, 1, -1):
        ctx.decline("np.trapezoid of a value whose dimensions are unknown")
    if axis in (0, -2):
        return ctx.call("trapz", *args, loc=loc)
    # NumPy's default: along the rows, dimension 2, a column on the .m side.
    return ctx.transpose(ctx.call("trapz", *args, ctx.num(2), loc=loc))


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """A scalar of a 1-D array, a 1-D array of a 2-D one."""
    y = as_array(args[0]) if args else None
    if y is None or not y.is_array:
        return None
    kind = PyKind.VECTOR if y.is_matrix else PyKind.SCALAR
    return PyInfo(kind=kind, dtype=DType.FLOAT, finite=y.finite)
