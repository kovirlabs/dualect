# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for indexing/grow-on-assign: assigning past the end of an array.

``.m`` -> Python, one statement at a time (the engine sends every store the
scanner tagged ``assign:grow`` here):

* ``x(end+1) = v`` on a vector (or ``[]``) with a scalar ``v`` is
  ``x = np.append(x, v)``; NumPy promotes the dtype, so nothing truncates.
* A constant subscript beyond the known size (``z(5) = 9`` on a 1x2, or
  ``A(2, 4) = 1`` on a 2x2) first pads with zeros, as the source does:
  ``z = np.pad(z, (0, 3))``, ``A = np.pad(A, ((0, 0), (0, 2)))``, then stores.
* An array that does not exist yet (``y(3) = 5``) starts as ``np.zeros``.

Declined: ``end + k`` with ``k > 1``, a computed subscript, a value that is
not a scalar, an array of unknown shape, and a variable that is defined on
some paths only. In a loop ``np.append`` copies the array each time; for long
loops a Python list and one ``np.array`` at the end is faster (a pitfall in
the rule, not a rewrite: it spans statements).

Python -> ``.m``: ``np.append(x, v)`` of a 1-D ``x`` is ``[x, v]``;
``np.pad(x, (a, b))`` is ``[zeros(1, a), x, zeros(1, b)]`` and
``np.pad(A, ((0, r), (0, c)))`` adds ``r`` zero rows and ``c`` zero columns.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context, Line
from dualect.infer import Info, Kind
from dualect.lift import LiftContext
from dualect.pyinfer import PyInfo


def translate(node: ir.Node, ctx: Context) -> ast.expr:
    """Growth only happens in a store: see :func:`translate_store`."""
    ctx.decline("growing an array outside an assignment")


def _end_plus_one(arg: ir.Expr) -> bool:
    if not (isinstance(arg, ir.BinOp) and arg.op == "plus"):
        return False
    sides = (arg.left, arg.right)
    return any(isinstance(s, ir.End) for s in sides) and any(
        isinstance(s, ir.Num) and s.value == 1 for s in sides
    )


def _constant(arg: ir.Expr, ctx: Context) -> int | None:
    value = ctx.info(arg).value
    if isinstance(value, int | float) and float(value).is_integer() and value >= 1:
        return int(value)
    return None


def _before(stmt: ir.Assign, name: str, ctx: Context) -> Info | None:
    """Facts about the array before the store; ``None`` when it is not defined."""
    env = ctx.analysis.env_before.get(id(stmt), {})
    return env.get(name)


def translate_store(stmt: ir.Assign, ctx: Context) -> list[Line]:
    """``x(end+1) = v``, ``x(n) = v`` beyond the end, ``A(i, j) = v`` beyond the edge."""
    target = stmt.targets[0]
    if not (isinstance(target, ir.Index) and isinstance(target.target, ir.Name)):
        ctx.decline("growing something other than a named array")
    name = target.target.id
    if ctx.is_index_var(name):
        ctx.decline("growing an array of positions stored 0-based")
    value = ctx.info(stmt.value)
    if not value.is_scalar:
        ctx.decline("growing an array by more than one element at a time")
    before = _before(stmt, name, ctx)
    if before is not None and before.maybe_undefined:
        ctx.decline(f"{name} is defined on some paths only")
    args = target.args
    if len(args) == 1 and _end_plus_one(args[0]):
        if before is None or not (before.is_vector or before.is_scalar):
            ctx.decline("x(end+1) = v on a value that is not known to be a vector")
        appended = ctx.call(ctx.np("append"), ast.Name(name), ctx.expr(stmt.value))
        return [ctx.line(f"{name} = {ast.unparse(appended)}")]
    rhs = ast.unparse(ctx.expr(stmt.value))
    positions = [_constant(arg, ctx) for arg in args]
    if any(p is None for p in positions) or len(args) > 2:
        ctx.decline("growing an array through a computed subscript")
    sizes = [p for p in positions if p is not None]
    if before is None or before.kind is Kind.EMPTY:
        # A new array: a row for one subscript (the source's x(3) = v is 1x3).
        shape = f"{sizes[0]}" if len(sizes) == 1 else f"({sizes[0]}, {sizes[1]})"
        start = ctx.line(f"{name} = {ast.unparse(ctx.np('zeros'))}({shape})")
        if len(sizes) == 2 and (sizes[0] == 1 or sizes[1] == 1):
            ctx.decline("a new array with one dimension 1 through two subscripts")
        return [start, *_store(name, sizes, rhs, ctx)]
    if len(sizes) == 1:
        if not before.is_vector or before.numel is None:
            ctx.decline("growing an array whose length is unknown")
        extra = sizes[0] - before.numel
        pad = f"{name} = {ast.unparse(ctx.np('pad'))}({name}, (0, {extra}))"
        return [ctx.line(pad), *_store(name, sizes, rhs, ctx)]
    if not before.is_matrix or before.rows is None or before.cols is None:
        ctx.decline("growing a matrix whose size is unknown (or a vector through two subscripts)")
    rows, cols = max(sizes[0] - before.rows, 0), max(sizes[1] - before.cols, 0)
    pad = f"{name} = {ast.unparse(ctx.np('pad'))}({name}, ((0, {rows}), (0, {cols})))"
    return [ctx.line(pad), *_store(name, sizes, rhs, ctx)]


def _store(name: str, sizes: list[int], rhs: str, ctx: Context) -> list[Line]:
    subscript = ", ".join(str(s - 1) for s in sizes)
    return [ctx.line(f"{name}[{subscript}] = {rhs}")]


# --- Python -> .m ------------------------------------------------------------


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.append(x, v)`` and ``np.pad`` with zeros."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    tag = ctx.tag(node) or ""
    literals = ctx.module_for("syntax/array-literals")
    if tag == "py:call:np.append":
        if len(node.args) != 2 or node.keywords:
            ctx.decline("np.append with an axis")
        first, second = node.args
        if not ctx.info(first).is_vector:
            ctx.decline("np.append of a value that is not 1-D (it flattens in row order)")
        added = ctx.info(second)
        if not (added.is_scalar or added.is_vector or isinstance(second, ast.List)):
            ctx.decline("np.append of a 2-D value (it flattens in row order)")
        row = [literals.lift_items(first, ctx), literals.lift_items(second, ctx)]
        return ir.Matrix(loc=loc, rows=[row])
    if tag != "py:call:np.pad":
        ctx.decline(f"{tag}")
    if len(node.args) != 2 or node.keywords:
        ctx.decline("np.pad with a mode or values other than zeros")
    data, widths = node.args
    info = ctx.info(data)
    value = literals.lift_items(data, ctx)
    if info.is_vector:
        pair = _pair(widths)
        if pair is None:
            ctx.decline("np.pad of a 1-D array without constant (before, after) widths")
        before, after = pair
        parts: list[ir.Expr] = []
        if before:
            parts.append(ctx.call("zeros", ctx.num(1), ctx.num(before), loc=loc))
        parts.append(value)
        if after:
            parts.append(ctx.call("zeros", ctx.num(1), ctx.num(after), loc=loc))
        return ir.Matrix(loc=loc, rows=[parts])
    if not info.is_matrix or not isinstance(widths, ast.Tuple) or len(widths.elts) != 2:
        ctx.decline("np.pad of a value that is not known to be 1-D or 2-D")
    row_pair, col_pair = (_pair(e) for e in widths.elts)
    if row_pair is None or col_pair is None or row_pair[0] or col_pair[0]:
        ctx.decline("np.pad other than zero rows below and zero columns to the right")
    rows, cols = row_pair[1], col_pair[1]
    height = ctx.call("size", value, ctx.num(1), loc=loc)
    width = ctx.call("size", value, ctx.num(2), loc=loc)
    top = [value] + ([ctx.call("zeros", height, ctx.num(cols), loc=loc)] if cols else [])
    grid = [top]
    if rows:
        total = ctx.binop("plus", width, ctx.num(cols), loc=loc) if cols else width
        grid.append([ctx.call("zeros", ctx.num(rows), total, loc=loc)])
    return ir.Matrix(loc=loc, rows=grid)


def _pair(node: ast.expr) -> tuple[int, int] | None:
    """``(a, b)`` of constant non-negative integers."""
    if not (isinstance(node, ast.Tuple) and len(node.elts) == 2):
        return None
    values = [e.value if isinstance(e, ast.Constant) else None for e in node.elts]
    if all(type(v) is int and v >= 0 for v in values):
        first, second = values
        assert isinstance(first, int) and isinstance(second, int)
        return first, second
    return None


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """``np.pad`` keeps the kind and element type of its array (zeros are finite)."""
    if tag != "py:call:np.pad" or not args or not args[0].is_array:
        return None
    return replace(args[0], value=None, length=None, view=False, nonneg=False)
