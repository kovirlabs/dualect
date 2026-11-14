# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/std-var: ``std`` and ``var``.

The source normalises by N - 1 by default (``std(x)``, ``std(x, 0)``) and by
N with ``std(x, 1)``; NumPy's default is N, so the translation always states
``ddof``: ``x.std(ddof=1)`` or ``x.std()``. A vector reduces to a scalar, a
matrix down its columns (``axis=0``) unless a dimension is given. With N - 1
a single element gives 0 in the source and NaN in NumPy, so data not known
to have two or more elements along the reduced dimension is assumed to,
out loud. A weight vector, complex data and a reduction along a singleton
dimension are declined.

From Python, ``ddof=1`` is the default form and no ``ddof`` is ``std(x, 1)``;
an axis becomes a dimension (``axis=1`` gives a column on the ``.m`` side,
so it is turned back into a row).
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind, array, scalar
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind, as_array


def _int(node: ir.Expr | None, ctx: Context, what: str) -> int | None:
    """A literal integer argument (``None`` when absent or ``[]``)."""
    if node is None or (isinstance(node, ir.Matrix) and not node.rows):
        return None
    value = ctx.info(node).value
    if not isinstance(value, int | float) or not float(value).is_integer():
        ctx.decline(f"{what} that is not a literal integer")
    return int(value)


def reduce_axis(
    node: ir.Call, data: Info, dim: int | None, ctx: Context
) -> tuple[int | None, int | None]:
    """``(axis, count along it)`` for a reduction of ``data`` over ``dim``.

    ``axis`` is ``None`` for a vector reduced along its length.
    """
    if data.is_matrix:
        if dim in (None, 1):
            return 0, data.rows
        if dim == 2:
            return 1, data.cols
        ctx.decline(f"{node.func} along dimension {dim}")
    if data.kind is Kind.EMPTY:
        ctx.decline(f"{node.func} of an empty array (the result depends on the version)")
    along = (
        dim is None or (dim == 2 and data.kind is Kind.ROW) or (dim == 1 and data.kind is Kind.COL)
    )
    if data.is_vector and along:
        return None, data.numel
    ctx.decline(f"{node.func} of a scalar, an unknown shape or along a singleton dimension")


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``std(x[, w[, dim]])`` or ``var(...)``."""
    if not 1 <= len(node.args) <= 3:
        ctx.decline(f"{node.func} with these arguments")
    data = ctx.info(node.args[0])
    if data.elem is not Elem.REAL:
        # Logical input: Octave keeps the class (or rejects it) where NumPy computes numbers.
        ctx.decline(f"{node.func} of logical, complex or unknown values")
    weight = _int(node.args[1] if len(node.args) > 1 else None, ctx, "a weight")
    if weight not in (None, 0, 1):
        ctx.decline(f"{node.func} with a weight vector")
    dim = _int(node.args[2] if len(node.args) > 2 else None, ctx, "a dimension")
    if data.is_scalar and dim is None:
        # One value: 0 under either weight. NumPy's N - 1 would divide by zero,
        # so the population form (ddof=0) is the one that gives 0.
        return ctx.call(ctx.np(node.func), ctx.expr(node.args[0]))
    axis, count = reduce_axis(node, data, dim, ctx)
    ddof = 0 if weight == 1 else 1
    if ddof == 1 and (count is None or count < 2):
        if count == 1:
            ctx.decline(f"{node.func} of one element (0 in the source, NaN in NumPy)")
        ctx.assume(f"at least two elements in {node.func} (one gives 0 here, NaN in NumPy)")
    keywords: dict[str, ast.expr] = {}
    if axis is not None:
        keywords["axis"] = ast.Constant(axis)
    if ddof:
        keywords["ddof"] = ast.Constant(ddof)
    return ctx.method(ctx.expr(node.args[0]), node.func, **keywords)


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """A scalar for a vector, a row (or a column, for dimension 2) for a matrix."""
    if not args:
        return None
    first = args[0]
    dim = args[2].value if len(args) > 2 else None
    facts = {"finite": first.finite, "nonneg": True}
    if first.is_vector or first.is_scalar:
        return scalar(**facts)  # type: ignore[arg-type]
    if first.is_matrix:
        if dim in (None, 1):
            return array(1, first.cols, kind=Kind.ROW, **facts)  # type: ignore[arg-type]
        if dim == 2:
            return array(first.rows, 1, kind=Kind.COL, **facts)  # type: ignore[arg-type]
    return None


# --- Python -> .m ------------------------------------------------------------


def _call_parts(node: ast.Call, ctx: LiftContext) -> tuple[str, ast.expr, list[ast.expr]]:
    """``(function, data, positional arguments after the data)``."""
    tag = ctx.tag(node) or ""
    if tag.startswith("py:method:"):
        assert isinstance(node.func, ast.Attribute)
        return tag.removeprefix("py:method:"), node.func.value, list(node.args)
    if not node.args:
        ctx.decline("a reduction without data")
    return tag.removeprefix("py:call:np."), node.args[0], list(node.args[1:])


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.std(x, ddof=1)``, ``x.var(axis=0)``, ..."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    func, data, rest = _call_parts(node, ctx)
    axis_node = rest[0] if rest else None
    ddof = 0
    if len(rest) > 1:
        ctx.decline(f"{func} with these positional arguments")
    for keyword in node.keywords:
        value = ctx.info(keyword.value).value
        if keyword.arg == "axis":
            axis_node = keyword.value
        elif keyword.arg == "ddof" and value in (0, 1) and not isinstance(value, bool):
            ddof = int(value)  # type: ignore[arg-type]
        else:
            ctx.decline(f"{func} with {keyword.arg}=")
    axis = None if axis_node is None else ctx.info(axis_node).value
    if axis is not None and axis not in (0, 1, -1, -2):
        ctx.decline(f"{func} along a computed axis")
    info = ctx.info(data)
    if info.dtype in (DType.COMPLEX, DType.UNKNOWN) and not isinstance(data, ast.List):
        ctx.decline(f"{func} of complex data or data of unknown type")
    value = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    weight = [] if ddof == 1 else [ctx.num(1)]
    if info.is_scalar and axis is None:
        if ddof == 1:
            ctx.decline(f"{func} with ddof=1 of one value (NaN here, 0 in .m code)")
        return ctx.call(func, value, loc=loc)
    if info.is_vector or isinstance(data, ast.List):
        if axis not in (None, 0, -1):
            ctx.decline(f"{func} of a 1-D array along axis 1")
        count = info.length
        if isinstance(data, ast.List):
            count = len(data.elts)
        if ddof == 1 and count is None:
            ctx.assume(f"at least two elements in {func} (one gives NaN here, 0 in .m code)")
        if ddof == 1 and count == 1:
            ctx.decline(f"{func} with ddof=1 of one element (NaN here, 0 in .m code)")
        return ctx.call(func, value, *weight, loc=loc)
    if not info.is_matrix:
        ctx.decline(f"{func} of a value whose dimensions are unknown")
    if ddof == 1:
        ctx.assume(f"at least two elements per reduction in {func}")
    if axis is None:
        flat = ir.Index(loc=loc, target=value, args=[ir.Colon(loc=loc)])
        return ctx.call(func, flat, *weight, loc=loc)
    dim = 1 if axis in (0, -2) else 2
    result = ctx.call(func, value, ctx.num(1 - ddof), ctx.num(dim), loc=loc)
    return result if dim == 1 else ctx.transpose(result)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """A scalar, or a 1-D array when a 2-D array is reduced along an axis."""
    data = receiver if receiver is not None else (as_array(args[0]) if args else None)
    if data is None or not data.is_array:
        return None
    axis = keywords.get("axis") or (args[0] if receiver is not None and args else None)
    if receiver is None and len(args) > 1:
        axis = args[1]
    facts = PyInfo(kind=PyKind.SCALAR, dtype=DType.FLOAT, finite=data.finite, nonneg=True)
    if axis is not None and data.is_matrix:
        return replace(facts, kind=PyKind.VECTOR)
    return facts
