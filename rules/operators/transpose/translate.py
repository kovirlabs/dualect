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
from dualect.lift import LiftContext


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


# --- Python -> .m ------------------------------------------------------------
#
# .T of a 1-D array changes nothing (it is a row on the .m side either way);
# of a 2-D array it is .' (never ', which also conjugates). x.conj().T is x'.


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``.T``, ``.conj()``, ``np.conj`` and ``np.transpose``."""
    loc = ctx.loc(node)
    if isinstance(node, ast.Attribute):  # .T
        inner = node.value
        info = ctx.info(inner)
        if info.is_vector or info.is_scalar:
            return ctx.expr(inner)
        if not info.is_matrix:
            ctx.decline(".T of a value whose dimensions are unknown")
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "conj"
            and not inner.args
        ):
            return ir.Transpose(loc=loc, operand=ctx.expr(inner.func.value), conjugate=True)
        return ir.Transpose(loc=loc, operand=ctx.expr(inner), conjugate=False)
    assert isinstance(node, ast.Call)
    name = ctx.tag(node)
    if name == "py:method:conj":
        assert isinstance(node.func, ast.Attribute)
        if node.args:
            ctx.decline("conj with arguments")
        return ctx.call("conj", ctx.expr(node.func.value), loc=loc)
    if len(node.args) != 1 or node.keywords:
        ctx.decline(f"{name} with these arguments")
    if name == "py:call:np.conj":
        return ctx.call("conj", ctx.expr(node.args[0]), loc=loc)
    info = ctx.info(node.args[0])
    if info.is_vector or info.is_scalar:
        return ctx.expr(node.args[0])
    if not info.is_matrix:
        ctx.decline("np.transpose of a value whose dimensions are unknown")
    return ir.Transpose(loc=loc, operand=ctx.expr(node.args[0]), conjugate=False)
