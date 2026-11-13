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
