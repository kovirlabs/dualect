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
