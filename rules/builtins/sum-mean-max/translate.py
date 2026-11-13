# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/sum-mean-max: ``sum``, ``mean``, ``prod``, ``max``, ``min``.

A vector reduces to a scalar (``x.sum()``); a matrix reduces down its columns
(``axis=0``) unless a dimension is given (``d`` is ``axis=d - 1``). ``sum(A(:))``
is ``A.sum()``. ``max``/``min`` skip NaN, so data not known to be NaN-free uses
``np.nanmax``/``np.nanmin``; the two-argument form is ``np.maximum`` (or
``np.fmax``, which also skips NaN), and ``min(max(x, lo), hi)`` is
``np.clip(x, lo, hi)``. The position output of ``[m, i] = max(v)`` is
``np.argmax``, 0-based when the emitter stores ``i`` 0-based.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.infer import Info, Kind

_NAN_SKIPPING = {"max": "nanmax", "min": "nanmin"}
_PAIRWISE = {"max": ("maximum", "fmax"), "min": ("minimum", "fmin")}


def position_outputs(call: ir.Call, nargout: int) -> set[int]:
    """The second output of ``max``/``min`` of one argument is a position."""
    if call.func in ("max", "min") and len(call.args) == 1 and nargout >= 2:
        return {1}
    return set()


def _colon_all(node: ir.Expr) -> ir.Expr | None:
    """``A`` when ``node`` is ``A(:)``."""
    if isinstance(node, ir.Index) and len(node.args) == 1 and isinstance(node.args[0], ir.Colon):
        return node.target
    return None


def _dim(node: ir.Call, position: int, ctx: Context) -> int | None:
    if len(node.args) <= position:
        return None
    value = ctx.info(node.args[position]).value
    if not isinstance(value, int | float) or not float(value).is_integer():
        ctx.decline(f"{node.func} along a computed dimension")
    return int(value)


def _reduce(func: str, data: ir.Expr, info: Info, dim: int | None, ctx: Context) -> ast.expr:
    """``func`` of ``data`` over ``dim`` (``None``: the default dimension)."""
    expr = ctx.expr(data)
    nan_name = _NAN_SKIPPING.get(func) if not info.finite else None
    if info.is_matrix:
        axis = 0 if dim in (None, 1) else 1 if dim == 2 else None
        if axis is None:
            ctx.decline(f"{func} along dimension {dim}")
        if nan_name:
            return ctx.call(ctx.np(nan_name), expr, axis=ast.Constant(axis))
        return ctx.method(expr, func, axis=ast.Constant(axis))
    reduces = (
        dim is None or (dim == 2 and info.kind is Kind.ROW) or (dim == 1 and info.kind is Kind.COL)
    )
    if info.is_vector and reduces:
        if func == "sum" and info.is_logical:
            return ctx.call(ctx.np("count_nonzero"), expr)
        if nan_name:
            return ctx.call(ctx.np(nan_name), expr)
        return ctx.method(expr, func)
    ctx.decline(f"{func} of a value whose shape is unknown or along a singleton dimension")


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """One reduction, or a two-argument ``max``/``min``."""
    func, args = node.func, node.args
    if not args:
        ctx.decline(f"{func} without arguments")
    if func in _PAIRWISE and len(args) == 2:
        return _pairwise(node, ctx)
    if func in ("max", "min"):
        if len(args) == 3 and not (isinstance(args[1], ir.Matrix) and not args[1].rows):
            ctx.decline(f"{func} with a second argument and a dimension")
        dim = _dim(node, 2, ctx)
    else:
        dim = _dim(node, 1, ctx)
    data = args[0]
    whole = _colon_all(data)
    if whole is not None and dim is None:
        info = ctx.info(whole)
        if not (info.is_array or info.kind is Kind.UNKNOWN):
            ctx.decline(f"{func}(x(:)) of a non-array")
        nan_name = _NAN_SKIPPING.get(func) if not info.finite else None
        if nan_name:
            return ctx.call(ctx.np(nan_name), ctx.expr(whole))
        return ctx.method(ctx.expr(whole), func)
    return _reduce(func, data, ctx.info(data), dim, ctx)


def _pairwise(node: ir.Call, ctx: Context) -> ast.expr:
    a, b = node.args
    finite = ctx.info(a).finite and ctx.info(b).finite
    # min(max(x, lo), hi) clamps x: np.clip, when no NaN can change the meaning.
    inner = a if node.func == "min" else None
    if (
        isinstance(inner, ir.Call)
        and inner.func == "max"
        and len(inner.args) == 2
        and all(ctx.info(arg).finite for arg in [*inner.args, b])
    ):
        x, lo = inner.args
        return ctx.call(ctx.np("clip"), ctx.arg(x), ctx.arg(lo), ctx.arg(b))
    plain, nan_skipping = _PAIRWISE[node.func]
    return ctx.call(ctx.np(plain if finite else nan_skipping), ctx.arg(a), ctx.arg(b))


def translate_multi(stmt: ir.Assign, ctx: Context) -> list[Line]:
    """``[m, i] = max(v)``: the value and its first position."""
    call = stmt.value
    assert isinstance(call, ir.Call)
    if call.func not in ("max", "min") or len(call.args) != 1 or len(stmt.targets) != 2:
        ctx.decline(f"{call.func} with these outputs")
    data = call.args[0]
    info = ctx.info(data)
    if not info.is_vector:
        ctx.decline(f"[m, i] = {call.func}(x) of a matrix or unknown shape")
    value_target, position_target = stmt.targets
    expr = ctx.expr(data)
    arg_func = ("nanarg" if not info.finite else "arg") + call.func
    position = ctx.call(ctx.np(arg_func), expr)
    lines: list[Line] = []
    position_name = position_target.id if isinstance(position_target, ir.Name) else None
    stored_zero_based = position_name is not None and ctx.is_index_var(position_name)
    if position_name is not None:
        if stored_zero_based:
            lines.extend(ctx.index_directive(position_name))
            lines.append(ctx.line(f"{position_name} = {ast.unparse(position)}"))
        else:
            lines.append(ctx.line(f"{position_name} = {ast.unparse(position)} + 1"))
    if isinstance(value_target, ir.Name):
        if stored_zero_based and position_name is not None:
            value = ast.Subscript(value=expr, slice=ast.Name(position_name))
        else:
            value = _reduce(call.func, data, info, None, ctx)
        lines.append(ctx.line(f"{value_target.id} = {ast.unparse(value)}"))
    elif not isinstance(value_target, ir.Ignore):
        ctx.decline("assigning a reduction into an element")
    if isinstance(position_target, ir.Index):
        ctx.decline("assigning a position into an element")
    return lines
