# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/repmat: ``repmat`` is ``np.tile`` with care for 1-D.

``repmat(A, m, n)`` (or ``repmat(A, [m n])``; ``repmat(A, n)`` is ``n x n``
copies) is ``np.tile(A, (m, n))`` for a matrix or a scalar. A row (1-D) repeated
sideways only stays a row: ``np.tile(v, n)``; stacked (``m > 1``) it becomes
a matrix, ``np.tile(v, (m, n))``. A column (also 1-D) repeated downwards stays
a column, ``np.tile(c, m)``, and sideways needs its column axis first,
``np.tile(c[:, np.newaxis], (m, n))``, because NumPy reads a 1-D array as a
row. A count that might be 1 is assumed not to be,
out loud, where that would change the result's dimensions.

From Python, ``np.tile(A, k)`` repeats along the last axis: ``repmat(A, 1,
k)``; ``np.tile(A, (m, n))`` is ``repmat(A, m, n)`` (a 1-D array is a row on
both sides).
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Info, Kind, array
from dualect.lift import LiftContext
from dualect.pyinfer import PyInfo, PyKind
from dualect.pyinfer import array as py_array


def _counts(node: ir.Call) -> list[ir.Expr] | None:
    """The repeat counts ``[m, n]``."""
    args = node.args[1:]
    if len(args) == 2:
        return list(args)
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, ir.Matrix) and len(arg.rows) == 1 and len(arg.rows[0]) == 2:
            return list(arg.rows[0])
        if not isinstance(arg, ir.Matrix):
            return [arg, arg]
    return None


def _value(node: ir.Expr, info: Info | None) -> int | None:
    if isinstance(node, ir.Num) and isinstance(node.value, int | float):
        return int(node.value) if float(node.value).is_integer() else None
    value = info.value if info is not None else None
    if isinstance(value, int | float) and float(value).is_integer():
        return int(value)
    return None


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """The tiled dimensions."""
    if node.func != "repmat" or not args:
        return None
    counts = _counts(node)
    if counts is None:
        return None
    data = args[0]
    infos = args[1:] if len(args) == 3 else [args[1], args[1]] if len(node.args) == 2 else []
    m, n = (_value(c, infos[k] if k < len(infos) else None) for k, c in enumerate(counts))
    rows = data.rows * m if data.rows is not None and m is not None else None
    cols = data.cols * n if data.cols is not None and n is not None else None
    kind = None
    if data.kind is Kind.ROW and m == 1:
        kind = Kind.ROW
    elif data.kind is Kind.COL and n == 1:
        kind = Kind.COL
    elif rows != 1 and cols != 1 and not (data.is_scalar and 1 in (m, n)):
        kind = Kind.MATRIX  # translated as 2-D (a count of 1 is assumed out)
    result = array(
        rows,
        cols,
        elem=data.elem,
        finite=data.finite,
        nonneg=data.nonneg,
        integer=data.integer,
        kind=kind,
    )
    return replace(result, int_dtype=data.int_dtype)


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``repmat(A, m, n)``."""
    counts = _counts(node)
    if counts is None:
        ctx.decline("repmat with these arguments")
    data_node = node.args[0]
    data = ctx.info(data_node)
    m_node, n_node = counts
    m, n = (_value(c, ctx.info(c)) for c in counts)
    m_expr, n_expr = ctx.expr(m_node), ctx.expr(n_node)
    both = ast.Tuple(elts=[m_expr, n_expr])
    if data.kind is Kind.STRING:
        ctx.decline("repmat of a string")
    expr = ctx.expr(data_node)
    if data.is_scalar:
        if m == 1:
            return ctx.call(ctx.np("tile"), expr, n_expr)
        if n == 1:
            return ctx.call(ctx.np("tile"), expr, m_expr)
        if m is None or n is None:
            ctx.assume("neither count is 1")
        return ctx.call(ctx.np("tile"), expr, both)
    if data.is_matrix:
        return ctx.call(ctx.np("tile"), expr, both)
    if data.kind is Kind.ROW:
        if m == 1:
            return ctx.call(ctx.np("tile"), expr, n_expr)
        if m is None:
            ctx.assume("the row is stacked more than once (m is not 1)")
        return ctx.call(ctx.np("tile"), expr, both)
    if data.kind is Kind.COL:
        if n == 1:
            return ctx.call(ctx.np("tile"), expr, m_expr)
        if n is None:
            ctx.assume("the column is repeated sideways (n is not 1)")
        column = ast.Subscript(value=expr, slice=ast.Tuple(elts=[ast.Slice(), ctx.np("newaxis")]))
        return ctx.call(ctx.np("tile"), column, both)
    ctx.decline("repmat of a value whose shape (row, column or matrix) is unknown")


# --- Python -> .m ------------------------------------------------------------


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.tile(A, reps)``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if len(node.args) != 2 or node.keywords:
        ctx.decline("np.tile with these arguments")
    data, reps = node.args
    info = ctx.info(data)
    if not (info.is_vector or info.is_matrix or info.is_scalar or isinstance(data, ast.List)):
        ctx.decline("np.tile of a value whose dimensions are unknown")
    value = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    parts = list(reps.elts) if isinstance(reps, ast.Tuple | ast.List) else [reps]
    if not all(ctx.info(p).is_scalar for p in parts) or len(parts) not in (1, 2):
        ctx.decline("np.tile to more than two dimensions")
    if len(parts) == 1:
        return ctx.call("repmat", value, ctx.num(1), ctx.expr(parts[0]), loc=loc)
    return ctx.call("repmat", value, ctx.expr(parts[0]), ctx.expr(parts[1]), loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """``np.tile``: 2-D for two counts or a 2-D input, else 1-D."""
    if tag != "py:call:np.tile" or len(args) != 2:
        return None
    data, reps = args
    two = reps.kind in (PyKind.TUPLE, PyKind.LIST) and reps.length == 2
    kind = PyKind.MATRIX if two or data.is_matrix else PyKind.VECTOR
    return py_array(kind, data.dtype, finite=data.finite, nonneg=data.nonneg)
