# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for indexing/one-based: ``x(i)``, ``A(i, j)``, and stores into them.

Every subscript drops by one, with constants folded (``x(i + 1)`` is
``x[i]``); variables the emitter stores 0-based (they only hold positions) need
no shift at all. A single subscript into a matrix is a column-major linear
index: ``A.ravel(order="F")[k]``. Ranges become slices
(:mod:`indexing/ranges`), ``end`` a negative index (:mod:`indexing/end-in-index`).
A row or column vector indexed with two subscripts drops its singleton one.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line, offset
from dualect.infer import Info, Kind


def position_value(node: ir.Expr, ctx: Context) -> ast.expr:
    """The source (1-based) value of a subscript expression, in Python."""
    ends = ctx.module("indexing/end-in-index")
    if ends.end_offset(node) is not None:
        ctx.decline("end used as a number here")
    return ctx.expr(node)


def zero_based(node: ir.Expr, ctx: Context) -> ast.expr:
    """A 1-based scalar position as a 0-based Python index (``end - k`` is ``-1 - k``)."""
    ends = ctx.module("indexing/end-in-index")
    end_off = ends.end_offset(node)
    if end_off is not None:
        return ctx.const(-1 - end_off)
    k = ends.end_minus(node)
    if k is not None:
        return ast.BinOp(left=ctx.const(-1), op=ast.Sub(), right=ctx.expr(k))
    if isinstance(node, ir.Name) and ctx.is_index_var(node.id):
        return ast.Name(id=node.id)
    if _holds_positions(node, ctx):
        # An element of a 0-based position array is already a Python index.
        assert isinstance(node, ir.Index)
        return indexed(node.target, node.args, ctx, store=False)
    return offset(ctx.expr(node), -1)


def _holds_positions(node: ir.Expr, ctx: Context) -> bool:
    """``idx(k)`` where ``idx`` is stored 0-based."""
    return (
        isinstance(node, ir.Index)
        and isinstance(node.target, ir.Name)
        and ctx.is_index_var(node.target.id)
    )


def subscript(node: ir.Expr, ctx: Context) -> ast.expr:
    """One subscript as a Python index: a slice, a 0-based position or positions."""
    if isinstance(node, ir.Colon):
        return ast.Slice()
    if isinstance(node, ir.Range):
        return ctx.module("indexing/ranges").to_slice(node, ctx)
    info = ctx.info(node)
    if (
        isinstance(node, ir.Call)
        and node.func == "find"
        and len(node.args) == 1
        and ctx.info(node.args[0]).is_logical
        and ctx.info(node.args[0]).is_vector
    ):
        # x(find(mask)) selects the same elements as the mask itself.
        return ctx.expr(node.args[0])
    if info.is_logical and not info.is_scalar:
        # A mask in a two-subscript index (`A(mask, :)`) is a NumPy boolean index.
        ctx.module("indexing/logical")
        return ctx.expr(node)
    if isinstance(node, ir.Matrix) and info.is_array:
        items = [item for row in node.rows for item in row]
        if all(isinstance(i, ir.Num) and isinstance(i.value, int) for i in items):
            return ast.List(elts=[ctx.const(i.value - 1) for i in items if isinstance(i, ir.Num)])
    if info.kind is Kind.UNKNOWN or (info.elem.value == "unknown" and not info.is_scalar):
        ctx.assume(f"numeric positions in {ast.unparse(ctx.expr(node))}")
    return zero_based(node, ctx)


def _is_mask(node: ir.Expr, ctx: Context) -> bool:
    info = ctx.info(node)
    return info.is_logical and not info.is_scalar


def _is_one(node: ir.Expr) -> bool:
    return isinstance(node, ir.Num) and node.value == 1


def _vector_subscript(node: ir.Expr, ctx: Context) -> bool:
    """A subscript that selects several positions by value (fancy indexing)."""
    if isinstance(node, ir.Colon | ir.Range):
        return False
    return not ctx.info(node).is_scalar


def indexed(target: ir.Expr, args: list[ir.Expr], ctx: Context, *, store: bool) -> ast.expr:
    """``target(args)`` as a Python subscript expression."""
    info: Info = ctx.info(target)
    if isinstance(target, ir.Name) and ctx.is_index_var(target.id):
        base: ast.expr = ast.Name(id=target.id)  # index the stored positions themselves
    else:
        base = ctx.expr(target)
    name = ast.unparse(base)
    if info.kind is Kind.STRING:
        if store:
            ctx.decline("assigning into a string (Python strings are immutable)")
        if len(args) != 1:
            ctx.decline("a string with two subscripts")
        return ast.Subscript(value=base, slice=subscript(args[0], ctx))
    if info.kind in (Kind.FUNC, Kind.CELL):
        ctx.decline("indexing a function handle or cell")
    if len(args) == 1 and isinstance(args[0], ir.Colon):
        return ctx.module("indexing/colon-all").colon_all(target, ctx, store=store)
    if len(args) == 1 and _is_mask(args[0], ctx):
        return ctx.module("indexing/logical").select(target, args[0], ctx)
    if len(args) == 1:
        if info.is_matrix:
            if store:
                ctx.decline("a linear-index assignment into a matrix")
            base = ctx.method(base, "ravel", order=ast.Constant("F"))
        elif info.kind in (Kind.UNKNOWN, Kind.ARRAY):
            ctx.assume(f"{name} is a vector")
        return ast.Subscript(value=base, slice=subscript(args[0], ctx))
    if len(args) == 2:
        first, second = args
        if info.kind is Kind.ROW and _is_one(first):
            return ast.Subscript(value=base, slice=subscript(second, ctx))
        if info.kind is Kind.COL and _is_one(second):
            return ast.Subscript(value=base, slice=subscript(first, ctx))
        if info.is_vector or info.is_scalar:
            ctx.decline("two subscripts into a vector")
        if not info.is_matrix:
            ctx.assume(f"{name} is a matrix")
        if _vector_subscript(first, ctx) and _vector_subscript(second, ctx):
            ctx.decline("position vectors in both dimensions (needs np.ix_)")
        return ast.Subscript(
            value=base, slice=ast.Tuple(elts=[subscript(first, ctx), subscript(second, ctx)])
        )
    ctx.decline("more than two subscripts")


def translate(node: ir.Index, ctx: Context) -> ast.expr:
    """``x(i)`` read as a value."""
    result = indexed(node.target, node.args, ctx, store=False)
    if _holds_positions(node, ctx):
        # The stored positions are 0-based; the source value is one more.
        return ast.BinOp(left=result, op=ast.Add(), right=ast.Constant(1))
    return result


def translate_store(stmt: ir.Assign, ctx: Context) -> list[Line]:
    """``x(i) = v``: an element (or slice) assignment."""
    target = stmt.targets[0]
    assert isinstance(target, ir.Index)
    if isinstance(stmt.value, ir.Matrix) and not stmt.value.rows:
        return ctx.module("indexing/logical").translate_delete(stmt, ctx)
    before = ctx.info(target.target)
    value = ctx.info(stmt.value)
    if before.int_dtype and not (value.integer and value.is_real):
        ctx.decline("storing a possibly fractional value into an integer array would truncate")
    if len(target.args) == 1 and isinstance(target.args[0], ir.Colon):
        return ctx.module("indexing/colon-all").translate_store(stmt, ctx)
    lhs = indexed(target.target, target.args, ctx, store=True)
    rhs = ctx.arg(stmt.value)
    return [ctx.line(f"{ast.unparse(lhs)} = {ast.unparse(rhs)}")]
