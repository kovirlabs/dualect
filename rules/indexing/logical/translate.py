# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for indexing/logical: masks, mask assignment, deletion.

On a vector a mask selects the same elements in the same order:
``x[mask]``. On a matrix the source walks columns first, so the selection is
taken from the transposes, ``A.T[M.T]`` (a comparison of ``A`` with a scalar
is transposed inside: ``A.T[A.T > 2]``). Deleting with ``x(m) = []`` keeps
the complement, ``x = x[~m]``; deleting positions uses ``np.delete``.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.lift import LiftContext


def _transposed_mask(mask: ir.Expr, target: ir.Expr, ctx: Context) -> ast.expr:
    """``M.T``, pushed into ``A op scalar`` when ``A`` is the indexed matrix."""
    if (
        isinstance(mask, ir.BinOp)
        and mask.op in ir.COMPARISONS
        and isinstance(target, ir.Name)
        and ctx.info(mask.right).is_scalar
        and isinstance(mask.left, ir.Name)
        and mask.left.id == target.id
    ):
        left = ast.Attribute(value=ctx.expr(mask.left), attr="T")
        compared = ctx.expr(mask)
        assert isinstance(compared, ast.Compare)
        return ast.Compare(left=left, ops=compared.ops, comparators=compared.comparators)
    return ast.Attribute(value=ctx.expr(mask), attr="T")


def select(target: ir.Expr, mask: ir.Expr, ctx: Context) -> ast.expr:
    """``target(mask)`` for a single logical subscript, as a value or a store target."""
    info = ctx.info(target)
    mask_info = ctx.info(mask)
    base = ctx.expr(target)
    if info.is_matrix:
        if not mask_info.is_matrix:
            ctx.decline("a matrix indexed with a mask that is not a matrix")
        return ast.Subscript(
            value=ast.Attribute(value=base, attr="T"), slice=_transposed_mask(mask, target, ctx)
        )
    if not info.is_vector:
        ctx.assume(f"{ast.unparse(base)} is a vector")
    return ast.Subscript(value=base, slice=ctx.expr(mask))


def translate(node: ir.Index, ctx: Context) -> ast.expr:
    """``x(mask)`` read as a value."""
    if len(node.args) != 1:
        ctx.decline("a mask in a two-subscript index")
    return select(node.target, node.args[0], ctx)


def translate_delete(stmt: ir.Assign, ctx: Context) -> list[Line]:
    """``x(m) = []``: keep everything else."""
    target = stmt.targets[0]
    assert isinstance(target, ir.Index)
    if not isinstance(target.target, ir.Name) or len(target.args) != 1:
        ctx.decline("deleting from a matrix or with two subscripts")
    name = target.target.id
    info = ctx.info(target.target)
    if not info.is_vector:
        ctx.decline("deleting from a value that is not known to be a vector")
    arg = target.args[0]
    arg_info = ctx.info(arg)
    if arg_info.is_logical:
        keep = ast.UnaryOp(op=ast.Invert(), operand=ctx.expr(arg))
        return [ctx.line(f"{name} = {name}[{ast.unparse(keep)}]")]
    positions = ctx.module("indexing/one-based").subscript(arg, ctx)
    if isinstance(positions, ast.Slice):
        ctx.decline("deleting a range")
    deleted = ctx.call(ctx.np("delete"), ast.Name(name), positions)
    return [ctx.line(f"{name} = {ast.unparse(deleted)}")]


# --- Python -> .m ------------------------------------------------------------
#
# x[mask] on a 1-D array is x(mask). A 2-D mask selects in row order in NumPy
# and in column order on the .m side, so A[A > 2] is declined; the idiom
# A.T[A.T > 2] (column order, which dualect itself writes) is A(A > 2).'.


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """Never reached directly: a mask is part of a subscript (indexing/one-based)."""
    ctx.decline("a mask outside a subscript")


def lift_transposed_mask(node: ast.Subscript, ctx: LiftContext) -> ir.Expr:
    """``A.T[A.T op s]``: the column-order selection ``A(A op s).'``."""
    value = node.value
    mask = node.slice
    assert isinstance(value, ast.Attribute)
    matrix = value.value
    if (
        isinstance(matrix, ast.Name)
        and ctx.info(matrix).is_matrix
        and isinstance(mask, ast.Compare)
        and len(mask.ops) == 1
        and isinstance(mask.left, ast.Attribute)
        and mask.left.attr == "T"
        and isinstance(mask.left.value, ast.Name)
        and mask.left.value.id == matrix.id
        and ctx.info(mask.comparators[0]).is_scalar
    ):
        compare = ast.Compare(left=mask.left.value, ops=mask.ops, comparators=mask.comparators)
        ast.copy_location(compare, mask)
        ctx.analysis.infos[id(compare)] = ctx.info(mask)
        target = ir.Index(loc=ctx.loc(node), target=ctx.expr(matrix), args=[ctx.expr(compare)])
        return ctx.transpose(target)
    ctx.decline("indexing a transposed array")
