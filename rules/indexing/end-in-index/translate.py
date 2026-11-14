# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for indexing/end-in-index: ``end`` inside a subscript.

At the top of a subscript, ``end`` and ``end - k`` become negative Python
indices (``-1``, ``-1 - k``); :func:`end_offset` recognizes those forms for the
indexing and range translators. Anywhere else inside a subscript
(``x(round(end / 2))``) ``end`` is a number: the length of the dimension being
indexed, which :func:`translate` gives as ``x.size`` or ``x.shape[d]``.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.lift import LiftContext


def end_offset(node: ir.Expr) -> int | None:
    """``k`` when ``node`` is ``end`` (0) or ``end - k`` with a literal ``k``."""
    if isinstance(node, ir.End):
        return 0
    if (
        isinstance(node, ir.BinOp)
        and node.op == "minus"
        and isinstance(node.left, ir.End)
        and isinstance(node.right, ir.Num)
        and isinstance(node.right.value, int)
        and node.right.value >= 0
    ):
        return node.right.value
    return None


def end_minus(node: ir.Expr) -> ir.Expr | None:
    """The ``k`` expression of ``end - k`` (any ``k``), else ``None``."""
    if isinstance(node, ir.BinOp) and node.op == "minus" and isinstance(node.left, ir.End):
        return node.right
    return None


def translate(node: ir.End, ctx: Context) -> ast.expr:
    """``end`` as a number: the extent of the dimension it indexes."""
    target = ctx.expr(node.target)
    info = ctx.info(node.target)
    if node.ndims == 1:
        if info.kind.value == "string":
            return ctx.call(ast.Name(id="len"), target)
        return ast.Attribute(value=target, attr="size")
    if node.ndims == 2 and info.is_matrix:
        return ast.Subscript(
            value=ast.Attribute(value=target, attr="shape"), slice=ast.Constant(node.dim)
        )
    ctx.decline("end in a subscript of an array whose dimensions are unknown")


# --- Python -> .m ------------------------------------------------------------
#
# A negative Python index counts from the end: x[-1] is x(end), x[-k] is
# x(end - k + 1), x[-1 - n] is x(end - n).


def from_end(node: ast.expr, target: ir.Expr, dim: int, ndims: int, ctx: LiftContext) -> ir.Expr:
    """The 1-based position of a negative Python index: ``end + 1 + index``, folded."""
    end = ir.End(loc=ctx.loc(node), target=target, dim=dim, ndims=ndims)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = node.operand
        if isinstance(inner, ast.Constant) and isinstance(inner.value, int):
            return ctx.shift(end, 1 - inner.value)
        return ctx.shift(ctx.binop("minus", end, ctx.expr(inner)), 1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub):
        head = from_end(node.left, target, dim, ndims, ctx)
        op = "plus" if isinstance(node.op, ast.Add) else "minus"
        right = node.right
        if isinstance(right, ast.Constant) and isinstance(right.value, int):
            return ctx.shift(head, right.value if op == "plus" else -right.value)
        return ctx.binop(op, head, ctx.expr(right))
    ctx.decline("a negative index of this form")


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """Never reached directly: negative indices are part of a subscript."""
    ctx.decline("a negative index outside a subscript")
