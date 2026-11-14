# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/interp1: linear interpolation with ``np.interp``.

``interp1(x, y, xi)`` (or with ``'linear'``) is ``np.interp(xi, x, y,
left=np.nan, right=np.nan)``: the query points come first in NumPy, and
outside ``[x(1), x(end)]`` the source returns NaN where ``np.interp`` would
repeat the end values. ``np.interp`` also needs increasing sample points
(the source sorts them; NumPy silently returns wrong values), so ``x`` that
is not a literal increasing vector, an increasing range or an increasing
``linspace`` is assumed increasing, out loud. Other methods, extrapolation
and a matrix of values are declined.

From Python, ``np.interp(xi, x, y, left=np.nan, right=np.nan)`` is
``interp1(x, y, xi)``; plain ``np.interp`` clamps to the end values, which is
``interp1(x, y, min(max(xi, x(1)), x(end)))`` for NaN-free query points.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, as_array


def _assigned(node: ir.Name, ctx: Context) -> list[ir.Expr] | None:
    """Every value a script variable is assigned, or ``None`` if it changes otherwise."""
    if not ctx.scope.is_script or node.id in ctx.scope.mutated:
        return None
    body = [s for s in ctx.scan.program.body if not isinstance(s, ir.FunctionDef)]
    stmts = ir.walk_stmts(body)
    if any(isinstance(s, ir.For) and s.var.id == node.id for s in stmts):
        return None
    if any(isinstance(s, ir.AugAssign) for s in stmts):
        return None
    values = [
        stmt.value
        for stmt in stmts
        if isinstance(stmt, ir.Assign)
        and any(isinstance(t, ir.Name) and t.id == node.id for t in stmt.targets)
    ]
    if any(
        isinstance(stmt, ir.Assign)
        and len(stmt.targets) > 1
        and any(isinstance(t, ir.Name) and t.id == node.id for t in stmt.targets)
        for stmt in stmts
    ):
        return None
    return values or None


def _literal_values(node: ir.Expr, ctx: Context) -> bool:
    """Sample points known to be a literal list of numbers (or a variable only ever one)."""
    if isinstance(node, ir.Name):
        values = _assigned(node, ctx)
        return values is not None and all(_literal_values(v, ctx) for v in values)
    return isinstance(node, ir.Matrix) and all(
        isinstance(ctx.info(item).value, int | float) for row in node.rows for item in row
    )


def _increasing(node: ir.Expr, ctx: Context) -> bool:
    """Whether sample points are known to be strictly increasing."""
    if isinstance(node, ir.Matrix):
        values = [ctx.info(item).value for row in node.rows for item in row]
        if all(isinstance(v, int | float) for v in values) and len(values) > 1:
            return all(a < b for a, b in zip(values, values[1:], strict=False))  # type: ignore[operator]
        return False
    if isinstance(node, ir.Range):
        step = ctx.info(node.step).value if node.step is not None else 1
        return isinstance(step, int | float) and step > 0
    if isinstance(node, ir.Call) and node.func == "linspace" and len(node.args) >= 2:
        low, high = (ctx.info(a).value for a in node.args[:2])
        return isinstance(low, int | float) and isinstance(high, int | float) and low < high
    if isinstance(node, ir.Name):
        # A script variable only ever assigned increasing values (`t = 0:h:T`).
        assigned = _assigned(node, ctx)
        return assigned is not None and all(_increasing(v, ctx) for v in assigned)
    return False


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``interp1(x, y, xi[, 'linear'])``."""
    args = node.args
    if len(args) == 4:
        method = args[3]
        if not (isinstance(method, ir.Str) and method.value == "linear"):
            ctx.decline("interp1 with a method other than linear, or with extrapolation")
    elif len(args) != 3:
        ctx.decline("interp1 with these arguments")
    x, y, xi = args[:3]
    for data in (x, y):
        info = ctx.info(data)
        if not info.is_vector or info.elem not in (Elem.REAL, Elem.LOGICAL):
            ctx.decline("interp1 of sample points or values that are not a real vector")
    query = ctx.info(xi)
    if not (query.is_scalar or query.is_array) or not query.is_real:
        ctx.decline("interp1 at query points that are not real numbers")
    if not _increasing(x, ctx):
        if _literal_values(x, ctx):
            ctx.decline("interp1 with sample points that are not increasing (np.interp needs them)")
        ctx.assume("increasing sample points (np.interp does not sort them)")
    nan = ctx.np("nan")
    return ctx.call(
        ctx.np("interp"), ctx.arg(xi), ctx.arg(x), ctx.arg(y), left=nan, right=ctx.np("nan")
    )


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """The shape of the query points; NaN outside the samples."""
    if len(args) < 3:
        return None
    query = args[2]
    if query.kind.value == "unknown":
        return None
    return replace(
        query,
        elem=Elem.REAL,
        finite=False,
        nonneg=False,
        integer=False,
        value=None,
        func=None,
        int_dtype=False,
    )


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.interp(xi, x, y[, left=np.nan, right=np.nan])``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if len(node.args) != 3:
        ctx.decline("np.interp with these arguments")
    xi, x, y = node.args
    bounds = {}
    for keyword in node.keywords:
        if keyword.arg not in ("left", "right") or ctx.tag(keyword.value) != "py:const:np.nan":
            ctx.decline(f"np.interp with {keyword.arg}= other than np.nan")
        bounds[keyword.arg] = keyword.value
    for data in (x, y):
        info = ctx.info(data)
        if not (info.is_vector or isinstance(data, ast.List)) or info.dtype is DType.COMPLEX:
            ctx.decline("np.interp of sample points or values that are not real 1-D arrays")
    ctx.assume("increasing sample points (the .m interp1 would sort them)")
    literals = ctx.module_for("syntax/array-literals")
    points = literals.lift_items(x, ctx)
    values = literals.lift_items(y, ctx)
    query = literals.lift_items(xi, ctx)
    if len(bounds) == 2:
        return ctx.call("interp1", points, values, query, loc=loc)
    if bounds:
        ctx.decline("np.interp with only one of left= and right=")
    # np.interp repeats the end values outside the samples: clamp the query.
    if not isinstance(points, ir.Name):
        ctx.decline("np.interp clamping with sample points that are not a variable")
    finite = (
        all(ctx.info(item).finite for item in xi.elts)
        if isinstance(xi, ast.List)
        else ctx.info(xi).finite
    )
    if not finite:
        ctx.decline("np.interp of query points that may be NaN (the .m max skips NaN)")
    first = ir.Index(loc=loc, target=points, args=[ctx.num(1)])
    last = ir.Index(loc=loc, target=points, args=[ir.End(loc=loc, target=points, dim=0, ndims=1)])
    clamped = ctx.call("min", ctx.call("max", query, first, loc=loc), last, loc=loc)
    return ctx.call("interp1", points, values, clamped, loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """The shape of the query points, in floating point."""
    if not args:
        return None
    query = as_array(args[0])
    if not (query.is_scalar or query.is_array):
        return None
    clamped = not keywords
    finite = clamped and all(a.finite for a in args[:3])
    return replace(query, dtype=DType.FLOAT, finite=finite, nonneg=False, value=None, view=False)
