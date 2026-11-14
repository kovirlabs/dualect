# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/polyfit-polyval: ``np.polyfit`` and ``np.polyval``.

Both keep the source's argument order and its coefficient order (highest
power first), so they map one to one. ``polyfit`` needs vectors and a literal
whole degree; its extra outputs (``[p, S, mu]``, the error estimate and the
centring) and ``polyval``'s matching arguments are declined, because NumPy
returns them in different forms.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, array
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind, as_array


def _degree(node: ir.Expr, ctx: Context) -> None:
    value = ctx.info(node).value
    if not isinstance(value, int | float) or not float(value).is_integer() or value < 0:
        ctx.decline("polyfit with a degree that is not a literal whole number")


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``polyfit(x, y, n)`` or ``polyval(p, x)``."""
    if node.func == "polyfit":
        if len(node.args) != 3:
            ctx.decline("polyfit with these arguments")
        x, y, degree = node.args
        for data in (x, y):
            info = ctx.info(data)
            if not info.is_vector or not info.is_real:
                ctx.decline("polyfit of data that is not a real vector")
        _degree(degree, ctx)
        return ctx.call(ctx.np("polyfit"), ctx.arg(x), ctx.arg(y), ctx.expr(degree))
    if len(node.args) != 2:
        ctx.decline("polyval with an error structure or centring")
    p, x = node.args
    coefficients = ctx.info(p)
    if not (coefficients.is_vector or coefficients.is_scalar):
        ctx.decline("polyval with coefficients that are not a vector")
    if not (ctx.info(x).is_scalar or ctx.info(x).is_array):
        ctx.decline("polyval at points of unknown type")
    return ctx.call(ctx.np("polyval"), ctx.arg(p), ctx.arg(x))


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """``polyfit``: a row of n + 1 coefficients; ``polyval``: the shape of the points."""
    if node.func == "polyfit" and len(args) == 3:
        degree = args[2].value
        count = int(degree) + 1 if isinstance(degree, int | float) else None
        return array(1, count, elem=Elem.REAL)
    if node.func == "polyval" and len(args) == 2:
        x = args[1]
        return replace(
            x, elem=Elem.REAL, finite=False, nonneg=False, integer=False, value=None, func=None
        )
    return None


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.polyfit(x, y, deg)`` or ``np.polyval(p, x)``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    name = (ctx.tag(node) or "").removeprefix("py:call:np.")
    if node.keywords:
        ctx.decline(f"np.{name} with keyword arguments")
    literals = ctx.module_for("syntax/array-literals")
    if name == "polyfit":
        if len(node.args) != 3:
            ctx.decline("np.polyfit with these arguments")
        x, y, degree = node.args
        for data in (x, y):
            info = ctx.info(data)
            if not (info.is_vector or isinstance(data, ast.List)) or info.dtype is DType.COMPLEX:
                ctx.decline("np.polyfit of data that is not a real 1-D array")
        value = ctx.info(degree).value
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            ctx.decline("np.polyfit with a degree that is not a literal whole number")
        args = [literals.lift_items(x, ctx), literals.lift_items(y, ctx), ctx.expr(degree)]
        return ctx.call("polyfit", *args, loc=loc)
    if len(node.args) != 2:
        ctx.decline("np.polyval with these arguments")
    p, x = node.args
    if not (ctx.info(p).is_vector or isinstance(p, ast.List)):
        ctx.decline("np.polyval with coefficients that are not 1-D")
    return ctx.call("polyval", *(literals.lift_items(a, ctx) for a in node.args), loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """1-D coefficients from ``np.polyfit``; the points' shape from ``np.polyval``."""
    if tag == "py:call:np.polyfit" and len(args) == 3:
        degree = args[2].value
        length = degree + 1 if isinstance(degree, int) else None
        return PyInfo(kind=PyKind.VECTOR, dtype=DType.FLOAT, length=length)
    if tag == "py:call:np.polyval" and len(args) == 2:
        x = as_array(args[1])
        if x.is_scalar or x.is_array:
            return replace(x, dtype=DType.FLOAT, finite=False, nonneg=False, value=None)
    return None
