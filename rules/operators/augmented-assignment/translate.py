# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for operators/augmented-assignment: ``x += e``, ``x++``, ...

On a scalar this is Python's augmented assignment. On an array ``x += e``
would modify the array in place (and every alias of it), so it is written
``x = x + e``, which matches the source's copy semantics.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.lift import LiftContext

_AUGMENTED = {"plus": "+=", "minus": "-=", "mtimes": "*=", "mrdivide": "/="}


def translate(stmt: ir.AugAssign, ctx: Context) -> list[Line]:
    """One augmented assignment."""
    target = stmt.target
    if not isinstance(target, ir.Name):
        ctx.decline("augmented assignment to an element")
    if stmt.op not in _AUGMENTED:
        ctx.decline(f"{stmt.op}= is not supported")
    if ctx.is_index_var(target.id):
        ctx.decline("augmented assignment to a position variable")
    info = ctx.info(target)
    combined = ir.BinOp(loc=stmt.loc, op=stmt.op, left=target, right=stmt.value)
    if info.is_scalar and ctx.info(stmt.value).is_scalar:
        return [ctx.line(f"{target.id} {_AUGMENTED[stmt.op]} {ast.unparse(ctx.expr(stmt.value))}")]
    return [ctx.line(f"{target.id} = {ast.unparse(ctx.expr(combined))}")]


# --- Python -> .m ------------------------------------------------------------
#
# x += e becomes x = x + e (the .m form both dialects read). On an array,
# Python's += changes the array in place, so every alias sees it; the engine
# declines aliases of arrays that are written (dualect.lift.check_alias), and
# the operator itself is lifted by its own rule.


def lift(stmt: ast.stmt, ctx: LiftContext) -> list[ir.Stmt]:
    """One augmented assignment to a variable."""
    assert isinstance(stmt, ast.AugAssign)
    target = stmt.target
    if not isinstance(target, ast.Name):
        ctx.decline("augmented assignment to an element")
    combined = ast.BinOp(left=ast.Name(id=target.id, ctx=ast.Load()), op=stmt.op, right=stmt.value)
    ast.copy_location(combined, stmt)
    ast.fix_missing_locations(combined)
    # The synthetic node has no inferred facts of its own; its operands do.
    ctx.analysis.infos[id(combined.left)] = ctx.info(target)
    value = ctx.expr(combined)
    loc = ctx.loc(stmt)
    return [ir.Assign(loc=loc, targets=[ir.Name(loc=ctx.loc(target), id=target.id)], value=value)]
