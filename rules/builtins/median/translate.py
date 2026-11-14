# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/median: ``np.median``.

A vector reduces to a scalar, a matrix down its columns (``axis=0``) unless a
dimension is given. Both sides average the two middle values of an even
count and return NaN when the data holds one. An empty input is declined:
its result changed between Octave versions. From Python, ``np.median`` of a
1-D array is ``median(x)``, of a 2-D array without an axis ``median(A(:))``.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind, array, scalar
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind, as_array


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``median(x)`` or ``median(x, dim)``."""
    if len(node.args) not in (1, 2):
        ctx.decline("median with these arguments")
    data = ctx.info(node.args[0])
    if data.elem is not Elem.REAL:
        # Logical input: Octave keeps the class (or rejects it) where NumPy computes numbers.
        ctx.decline("median of logical, complex or unknown values")
    dim = None
    if len(node.args) == 2:
        value = ctx.info(node.args[1]).value
        if not isinstance(value, int | float) or not float(value).is_integer():
            ctx.decline("median along a computed dimension")
        dim = int(value)
    expr = ctx.arg(node.args[0])
    if data.is_scalar and dim is None:
        return ctx.call(ctx.np("median"), expr)
    axis, _count = ctx.module("builtins/std-var").reduce_axis(node, data, dim, ctx)
    if axis is None:
        return ctx.call(ctx.np("median"), expr)
    return ctx.call(ctx.np("median"), expr, axis=ast.Constant(axis))


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """A scalar for a vector, a row (or a column, for dimension 2) for a matrix."""
    if not args:
        return None
    first = args[0]
    dim = args[1].value if len(args) > 1 else None
    facts = {"finite": first.finite, "nonneg": first.nonneg}
    if first.is_vector or first.is_scalar:
        return scalar(**facts)  # type: ignore[arg-type]
    if first.is_matrix:
        if dim in (None, 1):
            return array(1, first.cols, kind=Kind.ROW, **facts)  # type: ignore[arg-type]
        if dim == 2:
            return array(first.rows, 1, kind=Kind.COL, **facts)  # type: ignore[arg-type]
    return None


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.median(x[, axis])``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if not node.args or len(node.args) > 2:
        ctx.decline("np.median with these arguments")
    data = node.args[0]
    axis_node = node.args[1] if len(node.args) > 1 else None
    for keyword in node.keywords:
        if keyword.arg != "axis":
            ctx.decline(f"np.median with {keyword.arg}=")
        axis_node = keyword.value
    axis = None if axis_node is None else ctx.info(axis_node).value
    if axis is not None and axis not in (0, 1, -1, -2):
        ctx.decline("np.median along a computed axis")
    info = ctx.info(data)
    if info.dtype in (DType.COMPLEX, DType.UNKNOWN) and not isinstance(data, ast.List):
        ctx.decline("np.median of complex data or data of unknown type")
    value = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    if info.is_scalar or info.is_vector or isinstance(data, ast.List):
        if axis not in (None, 0, -1):
            ctx.decline("np.median of a 1-D array along axis 1")
        if info.length == 0:
            ctx.decline("np.median of an empty array")
        return ctx.call("median", value, loc=loc)
    if not info.is_matrix:
        ctx.decline("np.median of a value whose dimensions are unknown")
    if axis is None:
        return ctx.call("median", ir.Index(loc=loc, target=value, args=[ir.Colon(loc=loc)]))
    dim = 1 if axis in (0, -2) else 2
    result = ctx.call("median", value, ctx.num(dim), loc=loc)
    return result if dim == 1 else ctx.transpose(result)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """A scalar, or a 1-D array when a 2-D array is reduced along an axis."""
    data = as_array(args[0]) if args else None
    if data is None or not (data.is_array or data.is_scalar):
        return None
    axis = keywords.get("axis") or (args[1] if len(args) > 1 else None)
    kind = PyKind.VECTOR if axis is not None and data.is_matrix else PyKind.SCALAR
    return PyInfo(kind=kind, dtype=DType.FLOAT, finite=data.finite, nonneg=data.nonneg)
