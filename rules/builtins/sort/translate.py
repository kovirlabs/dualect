# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/sort: ``sort(x)``, ``sort(x, dim)``, ``'descend'``, positions.

A vector sorts along itself, a matrix down its columns (``axis=0``; NumPy's
default is the last axis). Descending order is the ascending sort reversed
(``np.flip``), which puts NaN first as the source does. ``[s, i] = sort(v)`` takes the
positions from ``np.argsort(v, kind="stable")`` (the source's sort is stable;
NumPy's default quicksort is not) and stores them 0-based when ``i`` only
ever serves as a subscript, else adds 1. Descending positions must keep equal
values in their original order: ``np.argsort(-v, kind="stable")``, which is
only right for data without NaN (``-NaN`` sorts last), so other descending
positions are declined. Complex values (sorted by magnitude in the source)
and characters are declined.

From Python, ``np.sort`` is ``sort`` (a 2-D array along its last axis is
``sort(A, 2)``), and ``i = np.argsort(v, kind="stable")`` of a 1-D array is
``[~, i] = sort(v); i = i - 1`` (values stay 0-based, as in Python).
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context, Line
from dualect.infer import Elem, Info, Kind
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind, as_array


def position_outputs(call: ir.Call, nargout: int) -> set[int]:
    """The second output of ``sort`` holds positions."""
    return {1} if nargout >= 2 else set()


def _options(node: ir.Call, ctx: Context) -> tuple[int | None, bool]:
    """``(dimension or None, descending)``."""
    dim: int | None = None
    descending = False
    for arg in node.args[1:]:
        if isinstance(arg, ir.Str):
            mode = arg.value.lower()
            if mode not in ("ascend", "descend"):
                ctx.decline(f"sort with the option {arg.value!r}")
            descending = mode == "descend"
            continue
        value = ctx.info(arg).value
        if dim is not None or value not in (1, 2):
            ctx.decline("sort along a computed or third dimension")
        assert isinstance(value, int | float)
        dim = int(value)
    if len(node.args) > 3:
        ctx.decline("sort with these arguments")
    return dim, descending


def _check_data(info: Info, ctx: Context) -> None:
    if info.kind is Kind.STRING or info.elem in (Elem.CHAR, Elem.COMPLEX):
        ctx.decline("sort of characters or complex values (sorted by magnitude in the source)")
    if info.elem is Elem.UNKNOWN:
        ctx.assume("real values to sort")


def _reverse(expr: ast.expr, axis: int, ctx: Context) -> ast.expr:
    """``np.flip(expr)`` (along ``axis`` for a 2-D array)."""
    if axis < 0:
        return ctx.call(ctx.np("flip"), expr)
    return ctx.call(ctx.np("flip"), expr, axis=ast.Constant(axis))


def _sorted(data: ir.Expr, dim: int | None, descending: bool, ctx: Context) -> ast.expr:
    info = ctx.info(data)
    _check_data(info, ctx)
    if info.is_matrix:
        axis = 0 if dim in (None, 1) else 1
        result = ctx.call(ctx.np("sort"), ctx.expr(data), axis=ast.Constant(axis))
        return _reverse(result, axis, ctx) if descending else result
    along = (
        dim is None or (dim == 2 and info.kind is Kind.ROW) or (dim == 1 and info.kind is Kind.COL)
    )
    if not (info.is_vector and along):
        ctx.decline("sort of a value whose shape is unknown, or along a singleton dimension")
    result = ctx.call(ctx.np("sort"), ctx.arg(data))
    return _reverse(result, -1, ctx) if descending else result


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """The sorted values."""
    if not node.args:
        ctx.decline("sort without arguments")
    dim, descending = _options(node, ctx)
    return _sorted(node.args[0], dim, descending, ctx)


def translate_multi(stmt: ir.Assign, ctx: Context) -> list[Line]:
    """``[s, i] = sort(v)``: the sorted values and their 1-based positions."""
    call = stmt.value
    assert isinstance(call, ir.Call)
    if len(stmt.targets) != 2 or not call.args:
        ctx.decline("sort with these outputs")
    dim, descending = _options(call, ctx)
    data = call.args[0]
    info = ctx.info(data)
    _check_data(info, ctx)
    if not info.is_vector or info.kind is Kind.EMPTY or dim is not None:
        ctx.decline("[s, i] = sort(x) of a matrix, a value of unknown shape or along a dimension")
    expr = ctx.expr(data)
    if descending:
        if not info.finite or info.is_logical:
            ctx.decline(
                "descending positions of data that may hold NaN or is logical "
                "(equal values must keep their order)"
            )
        keyed: ast.expr = ast.UnaryOp(op=ast.USub(), operand=expr)
    else:
        keyed = expr
    position = ctx.call(ctx.np("argsort"), keyed, kind=ast.Constant("stable"))
    value_target, position_target = stmt.targets
    if not isinstance(position_target, ir.Name | ir.Ignore):
        ctx.decline("assigning positions into an element")
    lines: list[Line] = []
    position_name = position_target.id if isinstance(position_target, ir.Name) else None
    zero_based = position_name is not None and ctx.is_index_var(position_name)
    if position_name is not None:
        if zero_based:
            lines.extend(ctx.index_directive(position_name))
            lines.append(ctx.line(f"{position_name} = {ast.unparse(position)}"))
        else:
            lines.append(ctx.line(f"{position_name} = {ast.unparse(position)} + 1"))
    if isinstance(value_target, ir.Name):
        if zero_based and position_name is not None:
            value: ast.expr = ast.Subscript(value=expr, slice=ast.Name(position_name))
        else:
            value = _sorted(data, None, descending, ctx)
        lines.append(ctx.line(f"{value_target.id} = {ast.unparse(value)}"))
    elif not isinstance(value_target, ir.Ignore):
        ctx.decline("assigning sorted values into an element")
    return lines


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """The data's shape and type."""
    if not args:
        return None
    return replace(args[0], value=None, func=None, env=None)


def infer_outputs(node: ir.Call, args: list[Info], nargout: int) -> list[Info] | None:
    """The sorted values, then positions of the same shape."""
    if not args:
        return None
    first = replace(args[0], value=None, func=None, env=None)
    positions = replace(
        first, elem=Elem.REAL, finite=True, nonneg=True, integer=True, int_dtype=True
    )
    return [first, positions]


# --- Python -> .m ------------------------------------------------------------


def _keywords(node: ast.Call, ctx: LiftContext) -> tuple[ast.expr | None, str | None]:
    """``(axis node, kind)`` of a NumPy sort call."""
    axis_node: ast.expr | None = node.args[1] if len(node.args) > 1 else None
    kind: str | None = None
    for keyword in node.keywords:
        if keyword.arg == "axis":
            axis_node = keyword.value
        elif keyword.arg == "kind" and isinstance(keyword.value, ast.Constant):
            kind = str(keyword.value.value)
        else:
            ctx.decline(f"{keyword.arg}= in a sort")
    if len(node.args) > 2:
        ctx.decline("a sort with these arguments")
    return axis_node, kind


def _check_py_data(info: PyInfo, ctx: LiftContext) -> None:
    if info.dtype is DType.COMPLEX or info.kind is PyKind.STRING:
        ctx.decline("sorting complex values or text")
    if info.dtype is DType.UNKNOWN:
        ctx.assume("real values to sort")


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.sort(x, axis=...)``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if ctx.tag(node) == "py:call:np.argsort":
        ctx.decline("np.argsort inside an expression (it needs its own statement)")
    if not node.args:
        ctx.decline("np.sort without arguments")
    axis_node, _kind = _keywords(node, ctx)
    data = node.args[0]
    info = ctx.info(data)
    _check_py_data(info, ctx)
    axis: object = -1
    if axis_node is not None:
        axis = ctx.info(axis_node).value
        if isinstance(axis_node, ast.Constant) and axis_node.value is None:
            axis = None
    value = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    if info.is_vector or isinstance(data, ast.List):
        if axis not in (None, 0, -1):
            ctx.decline("np.sort of a 1-D array along axis 1")
        return ctx.call("sort", value, loc=loc)
    if not info.is_matrix:
        ctx.decline("np.sort of a value whose dimensions are unknown")
    if axis in (0, -2):
        return ctx.call("sort", value, loc=loc)
    if axis in (1, -1):
        return ctx.call("sort", value, ctx.num(2), loc=loc)
    ctx.decline("np.sort of a 2-D array along a flattened or computed axis")


def lift_assign(stmt: ast.Assign, ctx: LiftContext) -> list[ir.Stmt] | None:
    """``i = np.argsort(v, kind="stable")``: ``[~, i] = sort(v); i = i - 1``."""
    value = stmt.value
    target = stmt.targets[0]
    assert isinstance(value, ast.Call)
    if ctx.tag(value) != "py:call:np.argsort":
        return None
    if not value.args or not isinstance(target, ast.Name):
        ctx.decline("np.argsort with these arguments")
    axis_node, kind = _keywords(value, ctx)
    if kind not in ("stable", "mergesort"):
        ctx.decline("np.argsort without kind='stable' (equal values may come in any order)")
    data = value.args[0]
    info = ctx.info(data)
    _check_py_data(info, ctx)
    if not (info.is_vector or isinstance(data, ast.List)):
        ctx.decline("np.argsort of a value that is not 1-D")
    if axis_node is not None and ctx.info(axis_node).value not in (0, -1):
        ctx.decline("np.argsort along a computed axis")
    loc = ctx.loc(stmt)
    values = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    found = ir.Assign(
        loc=loc,
        targets=[ir.Ignore(loc=loc), ir.Name(loc=ctx.loc(target), id=target.id)],
        value=ctx.call("sort", values, loc=loc),
    )
    shifted = ir.Assign(
        loc=loc,
        targets=[ir.Name(loc=loc, id=target.id)],
        value=ctx.binop("minus", ir.Name(loc=loc, id=target.id), ctx.num(1)),
    )
    return [found, shifted]


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """``np.sort``: the data's facts (flattened without an axis); ``np.argsort``: positions."""
    if not args:
        return None
    data = as_array(args[0])
    if not data.is_array:
        return None
    if tag == "py:call:np.argsort":
        return PyInfo(
            kind=PyKind.VECTOR if data.is_vector else data.kind,
            dtype=DType.INT,
            finite=True,
            nonneg=True,
            length=data.length,
        )
    axis = keywords.get("axis", args[1] if len(args) > 1 else None)
    flattened = axis is not None and axis.kind is PyKind.NONE
    kind = PyKind.VECTOR if flattened else data.kind
    return replace(data, kind=kind, value=None, view=False)
