# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for operators/transpose: ``'`` (conjugate) and ``.'``.

A 1-D NumPy vector has no orientation, so transposing a vector changes
nothing but the facts the analysis tracks (it is a column now); only the
conjugation remains for complex data. A matrix gets ``.T`` (and ``.conj()``
first when it may be complex).
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Kind


def translate(node: ir.Transpose, ctx: Context) -> ast.expr:
    """``x'`` or ``x.'``."""
    info = ctx.info(node.operand)
    operand = ctx.expr(node.operand)
    conjugate = node.conjugate and not info.is_real
    if info.is_scalar:
        return ctx.call(ctx.np("conj"), operand) if conjugate else operand
    if info.kind in (Kind.ROW, Kind.COL, Kind.VECTOR, Kind.EMPTY):
        return ctx.method(operand, "conj") if conjugate else operand
    if info.is_matrix:
        if conjugate:
            operand = ctx.method(operand, "conj")
        return ast.Attribute(value=operand, attr="T")
    if info.kind is Kind.STRING:
        ctx.decline("transposing a char array")
    # Unknown shape: np.conj / np.transpose work on scalars and arrays alike.
    if conjugate:
        operand = ctx.call(ctx.np("conj"), operand)
    return ctx.call(ctx.np("transpose"), operand)
