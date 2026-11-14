# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for control-flow/for-columns.

A ``for`` loop takes one *column* of its iterable per pass. An integer range
is a Python ``range`` (shifted to 0-based when the loop variable only serves as
a position, see :func:`dualect.emit.index_variables`); a row vector iterates
its elements; a matrix iterates ``A.T`` (``A.T.copy()`` when the body writes
into the loop variable, since each row of ``A.T`` is a view of ``A``); a column
or a scalar runs once. A range known to be empty is declined: the source then
leaves the variable as an empty array, Python leaves it unbound.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line, offset
from dualect.infer import Kind
from dualect.lift import LiftContext


def _range(stmt: ir.For, it: ir.Range, ctx: Context) -> ast.expr | None:
    ranges = ctx.module("indexing/ranges")
    step = ranges.literal_int(it.step) if it.step is not None else 1
    start = ctx.info(it.start)
    if step is None or step == 0 or not start.integer:
        return None
    first = ctx.expr(it.start)
    stop = ctx.expr(it.stop)
    if ctx.is_index_var(stmt.var.id):
        if step != 1:
            return None
        args = [offset(first, -1), stop]
    else:
        args = [first, offset(stop, 1 if step > 0 else -1)]
        if step != 1:
            args.append(ctx.const(step))
    if len(args) == 2 and ast.unparse(args[0]) == "0":
        args = args[1:]
    return ctx.call(ast.Name("range"), *args)


def translate(stmt: ir.For, ctx: Context) -> list[Line]:
    """One ``for`` loop."""
    var = stmt.var.id
    it = stmt.iterable
    info = ctx.info(it)
    lines: list[Line] = []
    iterable: ast.expr | None = None
    if isinstance(it, ir.Range):
        iterable = _range(stmt, it, ctx)
        if iterable is not None and ctx.is_index_var(var):
            lines.extend(ctx.index_directive(var))
    if iterable is None:
        if ctx.is_index_var(var):
            ctx.decline("a position loop over something other than a range")
        expr = ctx.expr(it)
        if info.kind in (Kind.ROW, Kind.EMPTY) or isinstance(it, ir.Range):
            iterable = expr
        elif info.is_matrix:
            iterable = ast.Attribute(value=expr, attr="T")
        elif info.kind is Kind.COL or info.is_scalar:
            iterable = ast.List(elts=[expr])
        else:
            ctx.decline("a loop over a value whose shape is unknown")
    if isinstance(it, ir.Range) and info.numel == 0:
        # The source leaves the variable as an empty 1x0 array; Python leaves
        # it unbound.
        ctx.decline("a loop over a range that is known to be empty")
    if info.is_matrix and var in ctx.scope.mutated:
        # Each column of A.T is a view: writing into the loop variable would
        # change A, while the source works on a copy of the column.
        iterable = ctx.method(iterable, "copy")
    lines.append(ctx.line(f"for {var} in {ast.unparse(iterable)}:"))
    with ctx.loop(stmt):
        lines.extend(ctx.block(stmt.body))
    return lines


# --- Python -> .m ------------------------------------------------------------
#
# for k in range(a, b) is for k = a:b-1 (the values stay 0-based, as in
# Python). for x in v (1-D) takes each element: for x = v. A 2-D array yields
# its rows in Python but a .m loop takes columns, and a 1-D row is a row on
# the .m side: for c in A.T is for c = A with c = c.' first in the body, and
# for r in A is for r = A.' likewise. A list of arrays, for x in [v, w], loops
# over the cell {v, w} with x = x{1} first. Writing into such a row would write into
# A in NumPy (it is a view), which is declined. An empty range is declined:
# Python leaves the variable unbound, the .m side makes it an empty array.


def lift(stmt: ast.stmt, ctx: LiftContext) -> list[ir.Stmt]:
    """One ``for`` loop (``range`` anywhere else is declined)."""
    if not isinstance(stmt, ast.For):
        ctx.decline("range() outside a for loop")
    loc = ctx.loc(stmt)
    target = stmt.target
    if not isinstance(target, ast.Name):
        ctx.decline("unpacking in a for loop")
    var = ir.Name(loc=ctx.loc(target), id=target.id)
    it = stmt.iter
    prologue: list[ir.Stmt] = []
    iterable: ir.Expr
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
        iterable = _lift_range(it, ctx)
    else:
        info = ctx.info(it)
        rows_of: ast.expr | None = None
        if isinstance(it, ast.Attribute) and it.attr == "T" and ctx.info(it.value).is_matrix:
            rows_of = it.value  # rows of A.T: the columns of A
            iterable = ctx.expr(rows_of)
        elif info.is_matrix:
            rows_of = it
            iterable = ctx.transpose(ctx.expr(it))
        elif info.is_vector or (
            isinstance(it, ast.List) and all(ctx.info(e).is_scalar for e in it.elts)
        ):
            iterable = ctx.module_for("syntax/array-literals").lift_items(it, ctx)
        elif isinstance(it, ast.List) and it.elts and all(ctx.info(e).is_array for e in it.elts):
            # A list of arrays: loop over a cell, one array per pass.
            if target.id in ctx.scope.mutated:
                ctx.decline(f"writing into {target.id}, which shares memory with a list item")
            iterable = ir.CellLit(loc=ctx.loc(it), items=[ctx.expr(e) for e in it.elts])
            unwrap = ir.Index(loc=loc, target=var, args=[ctx.num(1)])
            prologue.append(ir.Assign(loc=loc, targets=[var], value=_cell_item(unwrap)))
        else:
            ctx.decline("a loop over a value that is not a range or a known array")
        if rows_of is not None:
            if target.id in ctx.scope.mutated:
                ctx.decline(f"writing into {target.id}, a view of the array being looped over")
            prologue.append(ir.Assign(loc=loc, targets=[var], value=ctx.transpose(var)))
    with ctx.loop():
        body = prologue + ctx.block(stmt.body)
    return [ir.For(loc=loc, var=var, iterable=iterable, body=body)]


def _lift_range(call: ast.Call, ctx: LiftContext) -> ir.Expr:
    """``range(...)`` as the ``.m`` range of the same values."""
    loc = ctx.loc(call)
    args = call.args
    if call.keywords or not 1 <= len(args) <= 3:
        ctx.decline("range with these arguments")
    ranges = ctx.module_for("indexing/ranges")
    step = ranges._step(args[2] if len(args) == 3 else None, ctx)
    start = ctx.num(0) if len(args) == 1 else ctx.expr(args[0])
    stop_node = args[0] if len(args) == 1 else args[1]
    stop = ctx.shift(ctx.expr(stop_node), -1 if step > 0 else 1)
    values = [ctx.info(a).value for a in args]
    if all(isinstance(v, int) for v in values):
        count = len(range(*values))  # type: ignore[arg-type]
        if count == 0:
            ctx.decline("a loop over an empty range (the loop variable would differ)")
    if step == 1:
        return ir.Range(loc=loc, start=start, stop=stop)
    return ir.Range(loc=loc, start=start, stop=stop, step=ctx.num(step))


def _cell_item(index: ir.Index) -> ir.Expr:
    """``c{1}``: the IR has no brace index, so it is carried as verbatim ``.m`` text."""
    target = index.target
    assert isinstance(target, ir.Name)
    return ir.Opaque(loc=index.loc, construct="expr:cell-index", text=f"{target.id}{{1}}")
