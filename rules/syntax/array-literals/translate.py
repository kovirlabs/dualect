# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for syntax/array-literals: ``[...]`` literals and concatenation.

Vectors become 1-D arrays (a row and a column alike), matrices nested lists,
a one-element literal its element, and ``[]`` an empty 1-D array. Elements
that are arrays are concatenated: a row of vectors and scalars with
``np.concatenate``, columns and matrices side by side with
``np.column_stack`` (a 1-D vector is taken as a column there), rows of equally
long rows with ``np.vstack``.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Kind
from dualect.lift import LiftContext


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
        kinds = {info.kind for info in infos[0]}
        if kinds <= {Kind.COL, Kind.MATRIX}:
            # 1-D vectors have no orientation; column_stack reads them as columns.
            parts = [ctx.expr(item) for item in rows[0]]
            return ctx.call(ctx.np("column_stack"), ast.Tuple(elts=parts))
        ctx.decline("horizontal concatenation mixing columns with rows or scalars")
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


# --- Python -> .m ------------------------------------------------------------
#
# A 1-D NumPy array is a 1xN row on the .m side, a list of lists a matrix.
# np.concatenate of 1-D parts is [a, b]; of 2-D parts (axis 0) [A; B];
# np.vstack stacks rows, np.column_stack turns 1-D parts into columns.


def lift_items(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """A list (or tuple) of scalars as a row, of equal rows as a matrix; else the value."""
    loc = ctx.loc(node)
    if not isinstance(node, ast.List | ast.Tuple):
        return ctx.expr(node)
    elts = node.elts
    if all(isinstance(e, ast.List | ast.Tuple) for e in elts) and elts:
        rows = [[ctx.expr(x) for x in row.elts] for row in elts]  # type: ignore[attr-defined]
        if len({len(row) for row in rows}) != 1:
            ctx.decline("rows of different lengths")
        if not all(ctx.info(x).is_scalar for row in elts for x in row.elts):  # type: ignore[attr-defined]
            ctx.decline("a nested list of arrays")
        return ir.Matrix(loc=loc, rows=rows)
    if not all(ctx.info(e).is_scalar for e in elts):
        ctx.decline("a list of arrays")
    return ir.Matrix(loc=loc, rows=[[ctx.expr(e) for e in elts]] if elts else [])


def _parts(node: ast.Call, ctx: LiftContext) -> list[ast.expr]:
    if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.Tuple | ast.List):
        ctx.decline(f"{ast.unparse(node.func)} with these arguments")
    return list(node.args[0].elts)


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.array``, a list argument, and the stacking functions."""
    loc = ctx.loc(node)
    if isinstance(node, ast.List):
        return lift_items(node, ctx)
    if not isinstance(node, ast.Call):
        ctx.decline("a tuple used as a value")
    name = ctx.tag(node)
    if name == "py:call:np.array":
        if len(node.args) != 1 or node.keywords:
            ctx.decline("np.array with a dtype or other options")
        return lift_items(node.args[0], ctx)
    parts = _parts(node, ctx)
    infos = [ctx.info(p) for p in parts]
    lifted = [lift_items(p, ctx) for p in parts]
    one_d = all(
        i.is_vector or i.is_scalar or isinstance(p, ast.List)
        for i, p in zip(infos, parts, strict=True)
    )
    two_d = all(i.is_matrix for i in infos)
    if name == "py:call:np.concatenate":
        if one_d:
            return ir.Matrix(loc=loc, rows=[lifted])
        if two_d:
            return ir.Matrix(loc=loc, rows=[[part] for part in lifted])
        ctx.decline("concatenating values of unknown or mixed dimensions")
    if name == "py:call:np.vstack":
        if all(
            i.is_vector or i.is_matrix or isinstance(p, ast.List)
            for i, p in zip(infos, parts, strict=True)
        ):
            return ir.Matrix(loc=loc, rows=[[part] for part in lifted])
        ctx.decline("vstack of values of unknown dimensions")
    if name == "py:call:np.column_stack":
        columns: list[ir.Expr] = []
        for info, part, value in zip(infos, parts, lifted, strict=True):
            if info.is_matrix:
                columns.append(value)
            elif info.is_vector or isinstance(part, ast.List):
                columns.append(ctx.transpose(value))
            else:
                ctx.decline("column_stack of a value of unknown dimensions")
        return ir.Matrix(loc=loc, rows=[columns])
    ctx.decline(f"{name}")
