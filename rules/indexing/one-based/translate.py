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
from dualect.lift import LiftContext


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


# --- Python -> .m ------------------------------------------------------------
#
# Every position rises by one: x[i] is x(i + 1), with constants folded
# (x[k - 1] is x(k)). A negative index counts from the end
# (indexing/end-in-index), a slice is a range (indexing/ranges), a boolean
# mask stays a mask (indexing/logical). A 1-D array is a row on the .m side,
# so a selection NumPy returns as 1-D from a column (A[:, j]) is transposed
# back into a row. A variable index that may be negative would count from
# the end in Python, so a non-negative index is assumed out loud unless known.


def _parts(node: ast.Subscript) -> list[ast.expr]:
    return list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]


def position(part: ast.expr, target: ir.Expr, dim: int, ndims: int, ctx: LiftContext) -> ir.Expr:
    """One Python subscript as a 1-based ``.m`` subscript."""
    if isinstance(part, ast.Slice):
        return ctx.module_for("indexing/ranges").slice_range(part, target, dim, ndims, ctx)
    info = ctx.info(part)
    if info.is_bool and not info.is_scalar:
        ctx.module_for("indexing/logical")
        return ctx.module_for("syntax/array-literals").lift_items(part, ctx)
    if _negative(part):
        return ctx.module_for("indexing/end-in-index").from_end(part, target, dim, ndims, ctx)
    if isinstance(part, ast.List):
        values = [e.value for e in part.elts if isinstance(e, ast.Constant)]
        ints = [v for v in values if type(v) is int and v >= 0]
        if len(ints) != len(part.elts):
            ctx.decline("a list subscript that is not of non-negative integer literals")
        return ir.Matrix(loc=ctx.loc(part), rows=[[ctx.num(v + 1) for v in ints]])
    if info.dtype.value not in ("int", "unknown") and not info.is_bool:
        ctx.decline("a subscript that is not an integer")
    if not info.nonneg:
        ctx.assume(f"non-negative index {ast.unparse(part)}")
    return ctx.shift(ctx.expr(part), 1)


def _negative(node: ast.expr) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub):
        return _negative(node.left)
    return False


def _is_whole(part: ast.expr) -> bool:
    return (
        isinstance(part, ast.Slice)
        and part.lower is None
        and part.upper is None
        and part.step is None
    )


def subscripted(node: ast.Subscript, ctx: LiftContext, *, store: bool) -> tuple[ir.Index, bool]:
    """``node`` as an ``.m`` index, and whether the 1-D result must be turned back into a row."""
    loc = ctx.loc(node)
    parts = _parts(node)
    value = node.value
    info = ctx.info(value)
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "ravel"
        and not value.args
        and [(k.arg, getattr(k.value, "value", None)) for k in value.keywords] == [("order", "F")]
        and ctx.info(value.func.value).is_matrix
        and len(parts) == 1
        and not isinstance(parts[0], ast.Slice)
        and ctx.info(parts[0]).is_scalar
        and not store
    ):
        # A.ravel(order="F")[k] walks the columns: the linear index A(k + 1).
        matrix = ctx.expr(value.func.value)
        index = ir.Index(loc=loc, target=matrix, args=[])
        index.args = [position(parts[0], matrix, 0, 1, ctx)]
        return index, False
    target = ctx.expr(value)
    if not isinstance(target, ir.Name):
        ctx.decline("indexing the result of an expression (needs a temporary)")
    index = ir.Index(loc=loc, target=target, args=[])
    if info.kind.value == "string":
        if len(parts) != 1 or store:
            ctx.decline("this string subscript")
        index.args = [position(parts[0], target, 0, 1, ctx)]
        return index, False
    if info.is_vector:
        if len(parts) != 1:
            ctx.decline("several subscripts into a 1-D array")
        part = parts[0]
        if _is_whole(part):
            index.args = [ir.Colon(loc=ctx.loc(part))]
            return index, not store  # x(:) is a column
        index.args = [position(part, target, 0, 1, ctx)]
        return index, False
    if not info.is_matrix:
        ctx.decline("indexing a value whose dimensions are unknown")
    if len(parts) == 1:
        part = parts[0]
        part_info = ctx.info(part)
        if part_info.is_bool and part_info.is_matrix:
            ctx.decline("a 2-D mask (NumPy selects in row order, the .m side in column order)")
        parts = [part, ast.Slice()]
    if len(parts) != 2:
        ctx.decline("more than two subscripts")
    kinds = []
    for part in parts:
        part_info = ctx.info(part)
        if isinstance(part, ast.Slice):
            kinds.append("slice")
        elif part_info.is_scalar:
            kinds.append("int")
        else:
            kinds.append("array")
    if kinds == ["array", "array"]:
        ctx.decline("position arrays in both dimensions (NumPy pairs them up)")
    index.args = [
        ir.Colon(loc=ctx.loc(p)) if _is_whole(p) else position(p, target, d, 2, ctx)
        for d, p in enumerate(parts)
    ]
    # A[i, :] is 1-D in NumPy and a row on the .m side; A[:, j] is 1-D in NumPy
    # but a column on the .m side, so it is transposed into the row it stands for.
    column = kinds[1] == "int" and kinds[0] != "int"
    return index, column and not store


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``x[i]``, ``A[i, j]`` and the other subscripts read as values."""
    assert isinstance(node, ast.Subscript)
    parts = _parts(node)
    value = node.value
    alias = ctx.numpy
    if any(
        (isinstance(p, ast.Constant) and p.value is None)
        or (
            isinstance(p, ast.Attribute)
            and isinstance(p.value, ast.Name)
            and p.value.id == alias
            and p.attr == "newaxis"
        )
        for p in parts
    ):
        return ctx.module_for("operators/arithmetic").lift_newaxis(node, ctx)
    if isinstance(value, ast.Attribute) and value.attr == "shape":
        return ctx.module_for("builtins/size-numel-length").lift_shape_index(node, ctx)
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in ctx.functions
    ):
        return ctx.module_for("functions/multiple-returns").lift_item(node, ctx)
    if isinstance(value, ast.Call) and ctx.tag(value) == "py:call:np.flatnonzero":
        return ctx.module_for("builtins/find").lift_subscript(node, ctx)
    if isinstance(value, ast.Attribute) and value.attr == "T" and len(parts) == 1:
        return ctx.module_for("indexing/logical").lift_transposed_mask(node, ctx)
    index, transpose = subscripted(node, ctx, store=False)
    return ctx.transpose(index) if transpose else index


def lift_store(stmt: ast.Assign, ctx: LiftContext) -> list[ir.Stmt]:
    """``x[i] = v``: an element, slice or mask assignment."""
    target = stmt.targets[0]
    assert isinstance(target, ast.Subscript)
    base = ctx.info(target.value)
    value = ctx.info(stmt.value)
    if base.is_bool and not value.is_bool:
        ctx.decline("storing numbers into a boolean array (NumPy converts them to True/False)")
    if base.dtype.value == "int" and value.dtype.value not in ("int", "bool"):
        ctx.decline("storing a possibly fractional value into an integer array (NumPy truncates)")
    if value.dtype.value == "complex" and base.dtype.value != "complex":
        ctx.decline("storing complex values into a real array (NumPy drops the imaginary part)")
    index, _transpose = subscripted(target, ctx, store=True)
    rhs = ctx.module_for("syntax/array-literals").lift_items(stmt.value, ctx)
    return [ir.Assign(loc=ctx.loc(stmt), targets=[index], value=rhs)]
