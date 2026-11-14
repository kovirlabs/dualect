# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/cumsum-cumprod: ``cumsum`` and ``cumprod``.

A vector runs along itself (``np.cumsum(v)``), a matrix down its columns
(``axis=0``) unless a dimension is given (``d`` is ``axis=d - 1``). A
dimension along which a vector is a singleton returns it unchanged in the
source and is declined. ``cumprod`` of data the translation may hold as
integers is taken in floating point (``dtype=float``): NumPy integers wrap
around on overflow where the source multiplies doubles.

From Python, ``np.cumsum(x)`` of a 1-D array is ``cumsum(x)``; of a 2-D array
it needs an axis (without one NumPy flattens in row order, which is declined);
``axis=0`` is ``cumsum(A, 1)``, ``axis=1`` is ``cumsum(A, 2)``.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind, as_array


def _dim(node: ir.Call, ctx: Context) -> int | None:
    if len(node.args) == 1:
        return None
    value = ctx.info(node.args[1]).value
    if len(node.args) > 2 or not isinstance(value, int | float) or value not in (1, 2):
        ctx.decline(f"{node.func} with these arguments (a computed or third dimension)")
    return int(value)


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``cumsum(x)``, ``cumprod(x)``, ``cumsum(x, dim)``."""
    if not node.args:
        ctx.decline(f"{node.func} without arguments")
    dim = _dim(node, ctx)
    data = node.args[0]
    info = ctx.info(data)
    if info.elem not in (Elem.REAL, Elem.LOGICAL):
        ctx.decline(f"{node.func} of values that may be complex or characters")
    keywords: dict[str, ast.expr] = {}
    if node.func == "cumprod" and info.int_dtype:
        keywords["dtype"] = ast.Name("float")  # the source multiplies doubles
    func = ctx.np(node.func)
    if info.is_matrix:
        axis = 0 if dim in (None, 1) else 1
        return ctx.call(func, ctx.expr(data), axis=ast.Constant(axis), **keywords)
    runs = (
        dim is None or (dim == 2 and info.kind is Kind.ROW) or (dim == 1 and info.kind is Kind.COL)
    )
    if info.is_vector and runs:
        return ctx.call(func, ctx.arg(data), **keywords)
    ctx.decline(f"{node.func} of a value whose shape is unknown, or along a singleton dimension")


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """The result has the shape of the data; a logical input counts as numbers."""
    if not args or node.func not in ("cumsum", "cumprod"):
        return None
    first = args[0]
    product = node.func == "cumprod"
    return replace(
        first,
        elem=Elem.REAL if first.elem is Elem.LOGICAL else first.elem,
        finite=first.finite and not product,
        value=None,
        int_dtype=(first.int_dtype or first.is_logical) and not product,
        func=None,
        env=None,
    )


# --- Python -> .m ------------------------------------------------------------


def _axis(node: ast.Call, args: list[ast.expr], ctx: LiftContext) -> int | None:
    axis_node = args[0] if args else None
    for keyword in node.keywords:
        if keyword.arg == "axis":
            axis_node = keyword.value
        elif keyword.arg == "dtype" and isinstance(keyword.value, ast.Name):
            if keyword.value.id != "float":
                ctx.decline(f"dtype={keyword.value.id} in a running sum")
        else:
            ctx.decline(f"{keyword.arg}= in a running sum")
    if axis_node is None:
        return None
    value = ctx.info(axis_node).value
    if isinstance(value, bool) or value not in (0, 1, -1, -2):
        ctx.decline("a running sum along a computed axis")
    assert isinstance(value, int)
    return value


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.cumsum(x, axis=...)`` or ``x.cumsum(...)``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    tag = ctx.tag(node) or ""
    func = tag.split(":")[-1].removeprefix("np.")
    if tag.startswith("py:method:"):
        assert isinstance(node.func, ast.Attribute)
        data, rest = node.func.value, list(node.args)
    else:
        if not node.args:
            ctx.decline(f"{func} without arguments")
        data, rest = node.args[0], list(node.args[1:])
    axis = _axis(node, rest, ctx)
    info = ctx.info(data)
    if info.dtype is DType.COMPLEX:
        ctx.decline(f"{func} of complex values")
    if func == "cumprod" and info.dtype in (DType.INT, DType.BOOL):
        ctx.assume("no integer overflow in the Python cumprod (the .m side multiplies doubles)")
    value = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    if info.is_vector or isinstance(data, ast.List):
        if axis not in (None, 0, -1):
            ctx.decline(f"{func} of a 1-D array along axis 1")
        return ctx.call(func, value, loc=loc)
    if not info.is_matrix:
        ctx.decline(f"{func} of a value whose dimensions are unknown")
    if axis is None:
        ctx.decline(f"{func} of a 2-D array without an axis (NumPy flattens it in row order)")
    return ctx.call(func, value, ctx.num(1 if axis in (0, -2) else 2), loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """1-D unless a 2-D array keeps its shape along an axis; booleans count as integers."""
    if receiver is not None:
        data, rest = receiver, args
    elif args:
        data, rest = as_array(args[0]), args[1:]
    else:
        return None
    if not data.is_array:
        return None
    has_axis = bool(rest) or "axis" in keywords
    kind = PyKind.MATRIX if data.is_matrix and has_axis else PyKind.VECTOR
    dtype = DType.INT if data.dtype is DType.BOOL else data.dtype
    if any(isinstance(k.value, ast.Name) and k.value.id == "float" for k in node.keywords):
        dtype = DType.FLOAT
    product = tag.endswith("cumprod")
    return PyInfo(
        kind=kind,
        dtype=dtype,
        finite=data.finite and not product,
        nonneg=data.nonneg,
        length=data.length if data.is_vector else None,
    )
