# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/unique: ``unique(x)`` is ``np.unique(x)``.

Both sort the distinct values; NumPy returns a 1-D array, which stands for
the row or column the source returns. Only the plain one-output call is
translated: options (``'stable'``, ``'rows'``, ...) and the position outputs
are declined. ``np.unique`` merges NaNs, which the source keeps apart, so
data that may hold NaN is declined. Characters are declined.

From Python, ``np.unique(x)`` of 1-D data is ``unique(x)``; of a 2-D array,
which NumPy flattens into a 1-D (row) result, ``unique(A).'``.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind, array
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind, as_array


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``unique(x)``."""
    if len(node.args) != 1:
        ctx.decline("unique with options")
    info = ctx.info(node.args[0])
    if info.kind is Kind.STRING or info.elem not in (Elem.REAL, Elem.LOGICAL):
        ctx.decline("unique of characters, complex values or values of unknown type")
    if not info.finite:
        ctx.decline("unique of data that may hold NaN (np.unique merges NaNs, the source not)")
    if not (info.is_array or info.is_scalar):
        ctx.decline("unique of a value whose shape is unknown")
    return ctx.call(ctx.np("unique"), ctx.arg(node.args[0]))


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """A row for a row, a column otherwise; the elements' facts are kept."""
    if not args:
        return None
    first = args[0]
    facts = {
        "elem": first.elem,
        "finite": first.finite,
        "nonneg": first.nonneg,
        "integer": first.integer,
    }
    if first.kind is Kind.ROW:
        info = array(1, None, kind=Kind.ROW, **facts)  # type: ignore[arg-type]
    elif first.is_array:
        info = array(None, 1, kind=Kind.COL, **facts)  # type: ignore[arg-type]
    else:
        return None
    return replace(info, int_dtype=first.int_dtype)


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.unique(x)``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if len(node.args) != 1 or node.keywords:
        ctx.decline("np.unique with options or extra outputs")
    data = node.args[0]
    info = ctx.info(data)
    if info.dtype in (DType.COMPLEX, DType.UNKNOWN) and not isinstance(data, ast.List):
        ctx.decline("np.unique of complex values or values of unknown type")
    finite = info.finite
    if isinstance(data, ast.List):
        finite = all(ctx.info(e).finite for e in data.elts)
    if not finite:
        ctx.decline("np.unique of data that may hold NaN (NumPy merges NaNs, the .m side not)")
    value = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    if info.is_vector or isinstance(data, ast.List):
        return ctx.call("unique", value, loc=loc)
    if not info.is_matrix:
        ctx.decline("np.unique of a value whose dimensions are unknown")
    return ctx.transpose(ctx.call("unique", value, loc=loc))


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """A 1-D array of the data's element type."""
    if not args or node.keywords:
        return None
    data = as_array(args[0])
    if not data.is_array:
        return None
    return PyInfo(kind=PyKind.VECTOR, dtype=data.dtype, finite=data.finite, nonneg=data.nonneg)
