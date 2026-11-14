# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/diff: ``diff(x)``, ``diff(x, n)``, ``diff(x, n, dim)``.

A vector is differenced along itself, a matrix down its columns (``axis=0``,
since ``np.diff`` defaults to the last axis), and dimension ``d`` is
``axis=d - 1``. The order must be a known positive integer. When it reaches
the length of the data the source moves on to the next dimension while NumPy
returns an empty array, so an order known to be too large is declined and an
order above 1 on data of unknown length is assumed to fit, out loud. A
logical array is converted to integers first: NumPy refuses boolean
subtraction, the source differences numbers.

From Python, ``np.diff(x, n, axis)`` is the same call with the axis as a
dimension; on a 2-D array the default axis is the last one, ``dim = 2``.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind, as_array


def _constant(node: ir.Expr, ctx: Context, what: str) -> int:
    value = ctx.info(node).value
    if not isinstance(value, int | float) or not float(value).is_integer() or value < 1:
        ctx.decline(f"diff with a {what} that is not a known positive integer")
    return int(value)


def _order_and_dim(node: ir.Call, ctx: Context) -> tuple[int, int | None]:
    if not 1 <= len(node.args) <= 3:
        ctx.decline("diff with these arguments")
    order = _constant(node.args[1], ctx, "order") if len(node.args) > 1 else 1
    dim = _constant(node.args[2], ctx, "dimension") if len(node.args) > 2 else None
    if dim is not None and dim > 2:
        ctx.decline("diff along a third dimension")
    return order, dim


def _check_length(length: int | None, order: int, ctx: Context) -> None:
    if length is not None and order >= length and length > 0:
        ctx.decline(
            "diff of an order not below the length (the source moves to the next dimension)"
        )
    if length is None and order > 1:
        ctx.assume(f"more than {order} elements to difference")


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """One ``diff`` call."""
    order, dim = _order_and_dim(node, ctx)
    data = node.args[0]
    info = ctx.info(data)
    if info.elem not in (Elem.REAL, Elem.LOGICAL):
        ctx.decline("diff of values that may be complex or characters")
    expr = ctx.expr(data)
    if info.is_logical:
        expr = ctx.method(expr, "astype", ast.Name("int"))  # NumPy cannot subtract booleans
    keywords: dict[str, ast.expr] = {}
    if order != 1:
        keywords["n"] = ast.Constant(order)
    if info.is_matrix:
        axis = 0 if dim in (None, 1) else 1
        _check_length(info.rows if axis == 0 else info.cols, order, ctx)
        return ctx.call(ctx.np("diff"), expr, **keywords, axis=ast.Constant(axis))
    along = (
        dim is None or (dim == 2 and info.kind is Kind.ROW) or (dim == 1 and info.kind is Kind.COL)
    )
    if info.is_vector and info.kind is not Kind.EMPTY and along:
        _check_length(info.numel, order, ctx)
        return ctx.call(ctx.np("diff"), expr, **keywords)
    ctx.decline(
        "diff of a scalar, an empty value, a value of unknown shape or a singleton dimension"
    )


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """One element (row or column) shorter per order along the differenced dimension."""
    if not args:
        return None
    first = args[0]
    order = 1
    if len(args) > 1 and isinstance(args[1].value, int | float):
        order = int(args[1].value)
    dim = int(args[2].value) if len(args) > 2 and isinstance(args[2].value, int | float) else None
    rows, cols = first.rows, first.cols
    if first.is_matrix:
        if dim == 2:
            cols = cols - order if cols is not None else None
        else:
            rows = rows - order if rows is not None else None
    elif first.kind is Kind.ROW:
        cols = cols - order if cols is not None else None
    elif first.kind is Kind.COL:
        rows = rows - order if rows is not None else None
    else:
        return None
    return replace(
        first,
        rows=rows,
        cols=cols,
        elem=Elem.REAL,
        nonneg=False,
        value=None,
        int_dtype=first.int_dtype or first.is_logical,
        func=None,
        env=None,
    )


# --- Python -> .m ------------------------------------------------------------


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.diff(x, n=..., axis=...)``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if not node.args or len(node.args) > 3:
        ctx.decline("np.diff with these arguments")
    named = {"n": node.args[1] if len(node.args) > 1 else None}
    named["axis"] = node.args[2] if len(node.args) > 2 else None
    for keyword in node.keywords:
        if keyword.arg not in ("n", "axis"):
            ctx.decline(f"np.diff with {keyword.arg}=")
        named[keyword.arg] = keyword.value
    order, axis = 1, None
    for key, value_node in named.items():
        if value_node is None:
            continue
        value = ctx.info(value_node).value
        if isinstance(value, bool) or not isinstance(value, int):
            ctx.decline(f"np.diff with a computed {key}")
        if key == "n":
            order = value
        else:
            axis = value
    if order < 1:
        ctx.decline("np.diff of order 0")
    data = node.args[0]
    if (
        isinstance(data, ast.Call)
        and isinstance(data.func, ast.Attribute)
        and data.func.attr == "astype"
        and len(data.args) == 1
        and isinstance(data.args[0], ast.Name)
        and data.args[0].id == "int"
        and ctx.info(data.func.value).is_bool
    ):
        # m.astype(int) of a mask: the .m diff takes the logical values as numbers.
        data = data.func.value
    info = ctx.info(data)
    if info.dtype in (DType.COMPLEX, DType.BOOL) and data is node.args[0]:
        ctx.decline("np.diff of complex or boolean values")
    value = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    args: list[ir.Expr] = [value]
    if info.is_vector or isinstance(data, ast.List):
        if axis not in (None, 0, -1):
            ctx.decline("np.diff of a 1-D array along axis 1")
        length = info.length if info.is_vector else len(data.elts)  # type: ignore[attr-defined]
        if length is not None and order >= length:
            ctx.decline("np.diff of an order not below the length")
        if length is None and order > 1:
            ctx.assume(f"more than {order} elements to difference")
        if order != 1:
            args.append(ctx.num(order))
        return ctx.call("diff", *args, loc=loc)
    if not info.is_matrix:
        ctx.decline("np.diff of a value whose dimensions are unknown")
    if axis not in (None, 0, 1, -1, -2):
        ctx.decline(f"np.diff along axis {axis}")
    if order > 1:
        ctx.assume(f"more than {order} elements to difference along the axis")
    dim = 1 if axis in (0, -2) else 2
    return ctx.call("diff", value, ctx.num(order), ctx.num(dim), loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """The data's kind and element type, shorter."""
    if not args:
        return None
    data = as_array(args[0])
    if not data.is_array:
        return None
    order = args[1].value if len(args) > 1 else keywords.get("n", PyInfo(value=1)).value
    length = None
    if data.is_vector and data.length is not None and isinstance(order, int):
        length = max(data.length - order, 0)
    kind = PyKind.MATRIX if data.is_matrix else PyKind.VECTOR
    return PyInfo(kind=kind, dtype=data.dtype, finite=data.finite, length=length)
