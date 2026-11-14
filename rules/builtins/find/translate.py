# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/find: positions of nonzero elements.

``find`` returns 1-based positions in column-major order. A variable that only
holds positions is stored 0-based by the emitter (see
:func:`dualect.emit.index_variables`); then ``find(x)`` is
``np.flatnonzero(x)``, and ``find(x, k)`` keeps the first ``k``. Elsewhere the
1-based value is ``np.flatnonzero(x) + 1``. A matrix is read column by column,
``ravel(order="F")``, and ``[r, c] = find(A)`` is ``c, r = np.nonzero(A.T)``.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.lift import LiftContext


def position_outputs(call: ir.Call, nargout: int) -> set[int]:
    """Every output of ``find`` (up to two) is a position."""
    return set(range(min(nargout, 2)))


def _flat(node: ir.Expr, ctx: Context) -> ast.expr:
    info = ctx.info(node)
    expr = ctx.arg(node)
    if info.is_matrix:
        return ctx.method(expr, "ravel", order=ast.Constant("F"))
    if not (info.is_vector or info.is_scalar):
        ctx.decline("find on a value whose shape is unknown")
    return expr


def translate_positions(call: ir.Call, ctx: Context) -> ast.expr:
    """``find(x)`` / ``find(x, k)`` as 0-based positions."""
    if len(call.args) not in (1, 2):
        ctx.decline("find with these arguments")
    result = ctx.call(ctx.np("flatnonzero"), _flat(call.args[0], ctx))
    if len(call.args) == 2:
        result = ast.Subscript(value=result, slice=ast.Slice(upper=ctx.expr(call.args[1])))
    return result


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``find`` where its 1-based values are needed."""
    return ast.BinOp(left=translate_positions(node, ctx), op=ast.Add(), right=ast.Constant(1))


def translate_multi(stmt: ir.Assign, ctx: Context) -> list[Line]:
    """``[r, c] = find(A)``: row and column positions in column-major order."""
    call = stmt.value
    assert isinstance(call, ir.Call)
    names = [t.id if isinstance(t, ir.Name) else "_" for t in stmt.targets]
    if len(names) != 2 or len(call.args) != 1:
        ctx.decline("find with these outputs")
    if not all(ctx.is_index_var(n) or n == "_" for n in names):
        ctx.decline("find positions used as values")
    if not ctx.info(call.args[0]).is_matrix:
        ctx.decline("[r, c] = find(x) of a vector")
    lines: list[Line] = []
    for n in names:
        if n != "_":
            lines.extend(ctx.index_directive(n))
    transposed = ast.Attribute(value=ctx.expr(call.args[0]), attr="T")
    rows, cols = names
    lines.append(
        ctx.line(f"{cols}, {rows} = {ast.unparse(ctx.call(ctx.np('nonzero'), transposed))}")
    )
    return lines


# --- Python -> .m ------------------------------------------------------------
#
# np.flatnonzero(x) of a 1-D array is find(x) - 1: the same positions, kept
# 0-based so the value equals the Python one. Of a 2-D array NumPy walks the
# rows, so it is find(A.').' - 1 (row order, as a row); np.flatnonzero(x)[:k]
# is find(x, k) - 1. r, c = np.nonzero(M) is [c, r] = find(M.') (find walks
# columns, NumPy rows), then each made a 0-based row.


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.flatnonzero``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if ctx.tag(node) != "py:call:np.flatnonzero" or len(node.args) != 1 or node.keywords:
        ctx.decline("np.nonzero (a tuple of position arrays)")
    arg = node.args[0]
    info = ctx.info(arg)
    value = ctx.module_for("syntax/array-literals").lift_items(arg, ctx)
    if info.is_vector or isinstance(arg, ast.List):
        found: ir.Expr = ctx.call("find", value, loc=loc)
    elif info.is_matrix:
        found = ctx.transpose(ctx.call("find", ctx.transpose(value), loc=loc))
    else:
        ctx.decline("np.flatnonzero of a value whose dimensions are unknown")
    return ctx.binop("minus", found, ctx.num(1), loc=loc)


def lift_subscript(node: ast.Subscript, ctx: LiftContext) -> ir.Expr:
    """``np.flatnonzero(x)[:k]``: the first ``k`` positions, ``find(x, k) - 1``."""
    call = node.value
    part = node.slice
    if not (
        isinstance(call, ast.Call)
        and ctx.tag(call) == "py:call:np.flatnonzero"
        and len(call.args) == 1
        and isinstance(part, ast.Slice)
        and part.lower is None
        and part.step is None
        and part.upper is not None
        and ctx.info(call.args[0]).is_vector
    ):
        ctx.decline("indexing the result of np.flatnonzero")
    if not ctx.info(part.upper).nonneg:
        ctx.decline("a count that may be negative")
    loc = ctx.loc(node)
    value = ctx.expr(call.args[0])
    found = ctx.call("find", value, ctx.expr(part.upper), loc=loc)
    return ctx.binop("minus", found, ctx.num(1), loc=loc)


def lift_multi(stmt: ast.Assign, ctx: LiftContext) -> list[ir.Stmt]:
    """``r, c = np.nonzero(M)``: row and column positions, 0-based, as rows.

    NumPy walks ``M`` in row order. ``find`` walks columns, so it is applied to
    ``M.'`` (or to ``X`` itself for ``M = X.T``), whose columns are ``M``'s
    rows; its first output then holds ``M``'s columns and its second ``M``'s rows.
    """
    target = stmt.targets[0]
    call = stmt.value
    assert isinstance(target, ast.Tuple | ast.List) and isinstance(call, ast.Call)
    if len(call.args) != 1 or call.keywords or len(target.elts) != 2:
        ctx.decline("np.nonzero with these arguments or outputs")
    names = [e.id if isinstance(e, ast.Name) else None for e in target.elts]
    if None in names:
        ctx.decline("unpacking into an element")
    rows, cols = (n for n in names if n is not None)
    if rows == cols:
        ctx.decline("the same name for both outputs")
    matrix = call.args[0]
    loc = ctx.loc(stmt)
    if (
        isinstance(matrix, ast.Attribute)
        and matrix.attr == "T"
        and ctx.info(matrix.value).is_matrix
    ):
        walked = ctx.expr(matrix.value)
    elif ctx.info(matrix).is_matrix:
        walked = ctx.transpose(ctx.expr(matrix))
    else:
        ctx.decline("np.nonzero of a value that is not 2-D")
    outputs: list[ir.Expr] = [ir.Name(loc=loc, id=cols), ir.Name(loc=loc, id=rows)]
    lifted: list[ir.Stmt] = [
        ir.Assign(loc=loc, targets=outputs, value=ctx.call("find", walked, loc=loc))
    ]
    for name in (rows, cols):
        variable = ir.Name(loc=loc, id=name)
        shifted = ctx.binop("minus", ctx.transpose(variable), ctx.num(1))
        lifted.append(ir.Assign(loc=loc, targets=[ir.Name(loc=loc, id=name)], value=shifted))
    return lifted
