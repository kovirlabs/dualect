# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/concatenation-functions: ``horzcat``, ``vertcat``, ``cat``.

``horzcat(a, b, ...)`` is ``[a, b, ...]`` and ``vertcat(a, b, ...)`` is
``[a; b; ...]``; ``cat(1, ...)`` stacks vertically and ``cat(2, ...)``
horizontally. They are translated as those bracket literals, by
syntax/array-literals, so rows, columns (both 1-D in NumPy) and matrices are
joined the same way in both spellings. ``cat`` along a third or computed
dimension is declined.

From Python, ``np.hstack`` of 1-D arrays (and scalars) joins them end to end,
``[a, b]``; of 2-D arrays side by side, also ``[A, B]``. A mix of 1-D and
2-D parts is declined: NumPy would refuse it or read the 1-D part as a row.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind, array
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind
from dualect.pyinfer import array as py_array


def _layout(node: ir.Call, dim_value: object) -> tuple[str, list[ir.Expr]] | None:
    """``("horizontal" | "vertical", parts)`` of a call, or ``None``."""
    if node.func == "horzcat":
        return "horizontal", list(node.args)
    if node.func == "vertcat":
        return "vertical", list(node.args)
    if node.func == "cat" and node.args:
        if dim_value == 1:
            return "vertical", list(node.args[1:])
        if dim_value == 2:
            return "horizontal", list(node.args[1:])
    return None


def _matrix(direction: str, parts: list[ir.Expr], loc: ir.Loc) -> ir.Matrix:
    rows = [parts] if direction == "horizontal" else [[part] for part in parts]
    return ir.Matrix(loc=loc, rows=rows if parts else [])


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """The joined dimensions, when every part's are known."""
    layout = _layout(node, args[0].value if args else None)
    if layout is None:
        return None
    direction, _parts = layout
    infos = args[1:] if node.func == "cat" else args
    infos = [i for i in infos if i.kind is not Kind.EMPTY]
    if not infos:
        return array(0, 0, finite=True, nonneg=True, integer=True)
    if any(i.kind in (Kind.UNKNOWN, Kind.ARRAY, Kind.VECTOR, Kind.STRING) for i in infos):
        return None
    along = [i.cols if direction == "horizontal" else i.rows for i in infos]
    across = {i.rows if direction == "horizontal" else i.cols for i in infos}
    total = sum(a for a in along if a is not None) if None not in along else None
    other = across.pop() if len(across) == 1 else None
    rows, cols = (other, total) if direction == "horizontal" else (total, other)
    elem = Elem.LOGICAL if all(i.elem is Elem.LOGICAL for i in infos) else Elem.REAL
    if any(i.elem in (Elem.UNKNOWN, Elem.COMPLEX, Elem.CHAR) for i in infos):
        elem = Elem.UNKNOWN
    result = array(
        rows,
        cols,
        elem=elem,
        finite=all(i.finite for i in infos),
        nonneg=all(i.nonneg for i in infos),
        integer=all(i.integer for i in infos),
    )
    return replace(result, int_dtype=all(i.int_dtype for i in infos))


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``horzcat``, ``vertcat`` or ``cat(dim, ...)``, as the bracket literal."""
    dim_value = ctx.info(node.args[0]).value if node.args else None
    layout = _layout(node, dim_value)
    if layout is None:
        ctx.decline(f"{node.func} along a dimension other than 1 or 2")
    direction, parts = layout
    if not parts:
        ctx.decline(f"{node.func} of nothing")
    literals = ctx.module("syntax/array-literals")
    result = literals.translate(_matrix(direction, parts, node.loc), ctx)
    assert isinstance(result, ast.expr)
    return result


# --- Python -> .m ------------------------------------------------------------


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.hstack((a, b, ...))``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.Tuple | ast.List):
        ctx.decline("np.hstack with these arguments")
    parts = list(node.args[0].elts)
    if not parts:
        ctx.decline("np.hstack of nothing")
    infos = [ctx.info(p) for p in parts]
    literals = ctx.module_for("syntax/array-literals")
    lifted = [literals.lift_items(p, ctx) for p in parts]
    one_d = all(
        i.is_vector or i.is_scalar or isinstance(p, ast.List)
        for i, p in zip(infos, parts, strict=True)
    )
    if one_d or all(i.is_matrix for i in infos):
        return ir.Matrix(loc=loc, rows=[lifted])
    ctx.decline("np.hstack of 1-D and 2-D parts, or of unknown dimensions")


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """``np.hstack``: 1-D parts give 1-D, 2-D parts 2-D."""
    if tag != "py:call:np.hstack" or len(args) != 1 or not args[0].items:
        return None
    parts = args[0].items
    dtype = DType.BOOL
    for part in parts:
        if part.dtype is DType.UNKNOWN:
            dtype = DType.UNKNOWN
            break
        order = [DType.BOOL, DType.INT, DType.FLOAT, DType.COMPLEX]
        dtype = max(dtype, part.dtype, key=order.index)
    kind = PyKind.MATRIX if all(p.is_matrix for p in parts) else PyKind.VECTOR
    return py_array(kind, dtype, finite=all(p.finite for p in parts))
