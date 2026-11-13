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
