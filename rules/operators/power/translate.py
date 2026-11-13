# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for operators/power: ``^`` and ``.^``.

``.^`` is ``**``. ``^`` is ``**`` between scalars and
``np.linalg.matrix_power`` for a square matrix and an integer exponent. A
negative integer exponent on an integer array is written as a float
(``x**-1.0``) because NumPy rejects negative integer powers of integers,
and an exponent above 2 (or unknown) is written as a float too, because NumPy
integers wrap around on overflow where the source computes in doubles
(``[2 3] .^ 40``); likewise ``2 .^ n`` for an integer array ``n`` is
``2.0**n``. Two logical operands are powered as numbers.
"""

import ast

from dualect import ir
from dualect.emit import Context, int_constant
from dualect.infer import Info


def _float_exponent(exponent: ast.expr) -> ast.expr:
    value = int_constant(exponent)
    if value is not None and value < 0:
        return Context.const(float(value))
    return exponent


def _small_exponent(exponent: Info) -> bool:
    """An exponent known to be 0, 1, 2 or negative: no int64 overflow from realistic data."""
    value = exponent.value
    return isinstance(value, int | float) and value <= 2


def _as_float(expr: ast.expr, info: Info, ctx: Context) -> ast.expr:
    """``expr`` computed in floating point: a float literal, ``float(n)`` or ``x.astype(float)``."""
    value = int_constant(expr)
    if value is not None:
        return Context.const(float(value))
    if isinstance(expr, ast.Constant) and isinstance(expr.value, float):
        return expr
    if info.is_scalar:
        return ctx.call(ast.Name("float"), expr)
    return ctx.method(expr, "astype", ast.Name("float"))


def translate(node: ir.BinOp, ctx: Context) -> ast.expr:
    """One power."""
    base, exponent = ctx.info(node.left), ctx.info(node.right)
    if node.op == "power":
        arithmetic = ctx.module("operators/arithmetic")
        a, b = arithmetic.expand(node, ctx)
        a = arithmetic.numeric(a, base, exponent, ctx)
        if base.is_array and base.int_dtype:
            b = _float_exponent(b)
            if not _small_exponent(exponent):
                # NumPy integers wrap around silently (3**40 overflows int64);
                # the source computes in doubles.
                b = _as_float(b, exponent, ctx)
        elif base.int_dtype and exponent.is_array and exponent.int_dtype:
            a = _as_float(a, base, ctx)  # 2 .^ (1:70): the same overflow, from the base
        return ast.BinOp(left=a, op=ast.Pow(), right=b)
    if node.op != "mpower":
        ctx.decline(f"operator {node.op}")
    a, b = ctx.expr(node.left), ctx.expr(node.right)
    if base.is_scalar and exponent.is_scalar:
        return ast.BinOp(left=a, op=ast.Pow(), right=b)
    if base.is_matrix and exponent.is_scalar:
        value = exponent.value
        if isinstance(value, int | float) and float(value).is_integer():
            return ctx.call(ctx.np("linalg.matrix_power"), a, ctx.const(int(value)))
        ctx.decline("a matrix power with a non-integer or unknown exponent")
    if exponent.is_scalar and base.kind.value in ("unknown", "array"):
        ctx.assume(f"scalar base {ast.unparse(a)}")
        return ast.BinOp(left=a, op=ast.Pow(), right=b)
    ctx.decline("a matrix power with these shapes")
