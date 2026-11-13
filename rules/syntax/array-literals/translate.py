# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for syntax/array-literals: ``[...]`` literals and concatenation.

Vectors become 1-D arrays (a row and a column alike), matrices nested lists,
a one-element literal its element, and ``[]`` an empty 1-D array. Elements
that are arrays are concatenated: a row of vectors and scalars with
``np.concatenate``, rows of equally long rows with ``np.vstack``.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Kind


def translate(node: ir.Matrix, ctx: Context) -> ast.expr:
    """One bracket literal."""
    if not node.rows:
        return ctx.call(ctx.np("array"), ast.List(elts=[]))
    infos = [[ctx.info(item) for item in row] for row in node.rows]
    flat = [info for row in infos for info in row]
    if any(info.kind is Kind.STRING for info in flat):
        ctx.decline("a bracket literal mixing strings and numbers")
    if all(info.is_scalar for info in flat):
        if len(flat) == 1:
            return ctx.expr(node.rows[0][0])
        if len(node.rows) == 1 or all(len(row) == 1 for row in node.rows):
            items = [ctx.expr(item) for row in node.rows for item in row]
            return ctx.call(ctx.np("array"), ast.List(elts=items))
        if len({len(row) for row in node.rows}) != 1:
            ctx.decline("rows of different lengths")
        rows = [ast.List(elts=[ctx.expr(item) for item in row]) for row in node.rows]
        return ctx.call(ctx.np("array"), ast.List(elts=list(rows)))
    return _concatenate(node, infos, ctx)


def _part(item: ir.Expr, ctx: Context) -> ast.expr:
    """One piece of a concatenation: a scalar becomes a one-element list."""
    if ctx.info(item).is_scalar:
        return ast.List(elts=[ctx.expr(item)])
    return ctx.expr(item)


def _concatenate(node: ir.Matrix, infos: list[list], ctx: Context) -> ast.expr:
    rows = node.rows
    if len(rows) == 1:
        if all(info.kind in (Kind.ROW, Kind.SCALAR, Kind.EMPTY) for info in infos[0]):
            parts = [_part(item, ctx) for item in rows[0]]
            return ctx.call(ctx.np("concatenate"), ast.Tuple(elts=parts))
        ctx.decline("horizontal concatenation of columns or matrices")
    if all(len(row) == 1 for row in rows):
        kinds = {info.kind for row in infos for info in row}
        if kinds <= {Kind.COL, Kind.SCALAR, Kind.EMPTY}:
            parts = [_part(row[0], ctx) for row in rows]
            return ctx.call(ctx.np("concatenate"), ast.Tuple(elts=parts))
        if kinds <= {Kind.ROW, Kind.MATRIX}:
            widths = {info.cols for row in infos for info in row}
            if None not in widths and len(widths) == 1:
                parts = [ctx.expr(row[0]) for row in rows]
                return ctx.call(ctx.np("vstack"), ast.Tuple(elts=parts))
    ctx.decline("this concatenation")
