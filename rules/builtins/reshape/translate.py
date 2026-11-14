# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/reshape: ``reshape`` fills column by column.

``reshape(x, m, n)`` (or ``reshape(x, [m n])``, with ``[]`` for the size to
be worked out) is ``np.reshape(x, (m, n), order="F")``: the elements are read
and written in column-major order, NumPy's ``order="F"``. A ``1xN`` or ``Nx1``
result is a 1-D array, ``x.ravel(order="F")`` (a vector reshaped into a vector
keeps its elements in order, so it is the vector itself). A size that may be
1 is assumed not to be, out loud, as for ``zeros``. NumPy's reshape may return
a view; when either side is later written in place the result is copied.

From Python, ``np.reshape(x, shape, order="F")`` is ``reshape(x, m, n)``; the
default row-major order is the transpose of a column-major reshape of the
transpose: ``np.reshape(A, (m, n))`` is ``reshape(A.', n, m).'``.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Info, Kind, array
from dualect.lift import LiftContext
from dualect.pyinfer import PyInfo, PyKind
from dualect.pyinfer import array as py_array


def _dims(node: ir.Call) -> list[ir.Expr] | None:
    """The size arguments: two scalars, or one ``[m n]`` literal."""
    args = node.args[1:]
    if len(args) == 2:
        return list(args)
    if len(args) == 1 and isinstance(args[0], ir.Matrix) and len(args[0].rows) == 1:
        row = args[0].rows[0]
        return list(row) if len(row) == 2 else None
    return None


def _is_placeholder(node: ir.Expr) -> bool:
    return isinstance(node, ir.Matrix) and not node.rows


def _known(node: ir.Expr, info: Info) -> int | None:
    value = info.value
    if isinstance(value, int | float) and float(value).is_integer():
        return int(value)
    return None


def _sizes(node: ir.Call, args: list[Info], data: Info) -> tuple[int | None, int | None] | None:
    """``(rows, cols)`` of the result, ``None`` entries when unknown; ``None`` if malformed."""
    dims = _dims(node)
    if dims is None:
        return None
    # Separate size arguments have facts; the items of a literal are read below.
    infos = [args[1], args[2]] if len(node.args) == 3 else [Info(), Info()]
    values: list[int | None] = []
    for dim, info in zip(dims, infos, strict=True):
        if _is_placeholder(dim):
            values.append(None)
            continue
        if isinstance(dim, ir.Num) and isinstance(dim.value, int | float):
            values.append(int(dim.value) if float(dim.value).is_integer() else None)
        else:
            values.append(_known(dim, info))
    rows, cols = values
    total = data.numel
    if total is not None:
        if _is_placeholder(dims[0]) and cols:
            rows = total // cols
        if _is_placeholder(dims[1]) and rows:
            cols = total // rows
    return rows, cols


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """The result's dimensions, as far as they are known."""
    if node.func != "reshape" or not args:
        return None
    data = args[0]
    sizes = _sizes(node, args, data)
    if sizes is None:
        return None
    rows, cols = sizes
    dims = _dims(node) or []
    kind = None
    one = [d for d in dims if isinstance(d, ir.Num) and d.value == 1]
    if not one and (rows != 1 and cols != 1):
        kind = Kind.MATRIX  # translated as 2-D (a size of 1 is assumed out)
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


def _target_mutated(node: ir.Call, ctx: Context) -> bool:
    """Whether the result or the reshaped variable is written in place later."""
    names = set()
    if isinstance(node.args[0], ir.Name):
        names.add(node.args[0].id)
    current = ctx.current
    if isinstance(current, ir.Assign) and current.value is node:
        names.update(t.id for t in current.targets if isinstance(t, ir.Name))
    return bool(names & ctx.scope.mutated)


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``reshape(x, m, n)`` / ``reshape(x, [m n])``."""
    dims = _dims(node)
    if dims is None:
        ctx.decline("reshape to other than two dimensions")
    data_node = node.args[0]
    data = ctx.info(data_node)
    if not (data.is_array or data.is_scalar) or data.kind is Kind.STRING:
        ctx.decline("reshape of a value that is not a known numeric array")
    if all(_is_placeholder(d) for d in dims):
        ctx.decline("reshape with two [] sizes")
    values = [None if _is_placeholder(d) else ctx.info(d).value for d in dims]
    expr = ctx.expr(data_node)
    if 1 in values:
        # A 1xN or Nx1 result is 1-D; read column by column.
        if data.is_vector or data.is_scalar:
            result = expr
        elif data.is_matrix:
            result = ctx.method(expr, "ravel", order=ast.Constant("F"))
        else:
            ctx.decline("reshape of a value whose shape is unknown")
    else:
        resolved = list(values)
        total = data.numel
        for k, other in ((0, 1), (1, 0)):
            known = resolved[other]
            if _is_placeholder(dims[k]) and total is not None and isinstance(known, int | float):
                resolved[k] = total / known if known else None
        if not all(isinstance(v, int | float) and v != 1 for v in resolved):
            ctx.assume("neither dimension is 1")
        sizes = [ctx.const(-1) if _is_placeholder(d) else ctx.expr(d) for d in dims]
        if not (data.is_array or data.is_scalar) or data.kind is Kind.UNKNOWN:
            ctx.decline("reshape of a value whose shape is unknown")
        result = ctx.call(ctx.np("reshape"), expr, ast.Tuple(elts=sizes), order=ast.Constant("F"))
    if _target_mutated(node, ctx) and not (
        isinstance(result, ast.Call)
        and isinstance(result.func, ast.Attribute)
        and result.func.attr == "copy"
    ):
        result = ctx.method(result, "copy")
    return result


# --- Python -> .m ------------------------------------------------------------


def _shape_parts(node: ast.expr, ctx: LiftContext) -> list[ast.expr] | None:
    if isinstance(node, ast.Tuple | ast.List):
        return list(node.elts)
    if ctx.info(node).is_scalar:
        return [node]
    return None


def _dim(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    value = ctx.info(node).value
    if value == -1:
        return ir.Matrix(loc=ctx.loc(node), rows=[])
    if isinstance(value, int | float) and value < 0:
        ctx.decline("a negative size")
    if value is None:
        ctx.assume("a size that is not -1")
    return ctx.expr(node)


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.reshape(x, shape[, order])``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    order = "C"
    for keyword in node.keywords:
        if keyword.arg != "order" or not isinstance(keyword.value, ast.Constant):
            ctx.decline(f"np.reshape with {keyword.arg}=")
        order = str(keyword.value.value)
    if len(node.args) != 2 or order not in ("C", "F"):
        ctx.decline("np.reshape with these arguments")
    data, shape = node.args
    info = ctx.info(data)
    if not (info.is_vector or info.is_matrix):
        ctx.decline("np.reshape of a value whose dimensions are unknown")
    parts = _shape_parts(shape, ctx)
    if parts is None or len(parts) not in (1, 2):
        ctx.decline("np.reshape to other than one or two dimensions")
    value = ctx.expr(data)
    empty = ir.Matrix(loc=loc, rows=[])
    if len(parts) == 1:
        # A 1-D result: a row on the .m side, read in the given order.
        if ctx.info(parts[0]).value != -1:
            ctx.assume("the size is the element count")
        source = value if order == "F" or info.is_vector else ctx.transpose(value)
        return ctx.call("reshape", source, ctx.num(1), empty, loc=loc)
    m, n = (_dim(p, ctx) for p in parts)
    if order == "F":
        return ctx.call("reshape", value, m, n, loc=loc)
    # Row-major: reshape the transpose column by column, then transpose back.
    source = value if info.is_vector else ctx.transpose(value)
    return ctx.transpose(ctx.call("reshape", source, n, m, loc=loc))


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """``np.reshape``: 1-D for one size, 2-D for two."""
    if tag != "py:call:np.reshape" or len(args) != 2:
        return None
    data, shape = args
    count = shape.length if shape.kind in (PyKind.TUPLE, PyKind.LIST) else 1
    kind = PyKind.VECTOR if count == 1 else PyKind.MATRIX if count == 2 else PyKind.UNKNOWN
    if kind is PyKind.UNKNOWN:
        return None
    return replace(py_array(kind, data.dtype, finite=data.finite, nonneg=data.nonneg), view=True)
