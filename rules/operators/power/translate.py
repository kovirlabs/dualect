# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for operators/power: ``^`` and ``.^``.

``.^`` is ``**``. ``^`` is ``**`` between scalars and
``np.linalg.matrix_power`` for a square matrix and an integer exponent. A
negative integer exponent on an integer array is written as a float
(``x**-1.0``) because NumPy rejects negative integer powers of integers,
and an exponent of 2 or more (or unknown) is written as a float too, because
NumPy integers wrap around on overflow where the source computes in doubles
(``[2 3] .^ 40``, and even ``x .^ 2`` for ``x`` above 3037000499); the same
holds for an element of an integer array (``np.int64``), while a scalar
literal is a Python ``int``, which never overflows. Likewise ``2 .^ n`` for
an integer array ``n`` is ``2.0**n``. Two logical operands are powered as
numbers.
"""

import ast

from dualect import ir
from dualect.emit import Context, int_constant
from dualect.infer import Info
from dualect.lift import LiftContext


def _float_exponent(exponent: ast.expr) -> ast.expr:
    value = int_constant(exponent)
    if value is not None and value < 0:
        return Context.const(float(value))
    return exponent


def _small_exponent(exponent: Info) -> bool:
    """An exponent known to be at most 1: no integer power can overflow int64.

    Even a square overflows: an integer array holding 3037000500 has a square
    beyond 2**63 - 1, and NumPy wraps it around without a word.
    """
    value = exponent.value
    return isinstance(value, int | float) and value <= 1


def _numpy_integer_element(node: ir.Expr, info: Info) -> bool:
    """An element of an integer array: a NumPy integer, which wraps on overflow.

    A scalar that comes from a literal is a Python ``int`` (unbounded); one
    read out of an integer array is ``np.int64``.
    """
    return info.is_scalar and info.int_dtype and isinstance(node, ir.Index)


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
        if (base.is_array and base.int_dtype) or _numpy_integer_element(node.left, base):
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
        if _numpy_integer_element(node.left, base) and not _small_exponent(exponent):
            b = _as_float(_float_exponent(b), exponent, ctx)  # np.int64 wraps around too
        return ast.BinOp(left=a, op=ast.Pow(), right=b)
    if base.is_matrix and exponent.is_scalar:
        value = exponent.value
        if isinstance(value, int | float) and float(value).is_integer():
            if base.int_dtype and value >= 2:
                # Repeated integer matrix products wrap around (A^3 of entries
                # near 2e6 already does); the source multiplies doubles.
                a = _as_float(a, base, ctx)
            return ctx.call(ctx.np("linalg.matrix_power"), a, ctx.const(int(value)))
        ctx.decline("a matrix power with a non-integer or unknown exponent")
    if exponent.is_scalar and base.kind.value in ("unknown", "array"):
        ctx.assume(f"scalar base {ast.unparse(a)}")
        return ast.BinOp(left=a, op=ast.Pow(), right=b)
    ctx.decline("a matrix power with these shapes")


# --- Python -> .m ------------------------------------------------------------
#
# ** is .^ (^ between scalars); np.linalg.matrix_power(A, k) is A ^ k. A
# fractional power of a negative array element is NaN in NumPy but complex on
# the .m side, so such a power is assumed to stay in the real domain.


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``**`` and ``np.linalg.matrix_power``."""
    loc = ctx.loc(node)
    if isinstance(node, ast.Call):
        if len(node.args) != 2 or node.keywords or not ctx.info(node.args[0]).is_matrix:
            ctx.decline("matrix_power with these arguments")
        return ctx.binop("mpower", ctx.expr(node.args[0]), ctx.expr(node.args[1]), loc=loc)
    assert isinstance(node, ast.BinOp)
    base, exponent = ctx.info(node.left), ctx.info(node.right)
    if base.is_bool and exponent.is_bool:
        ctx.decline("a power of booleans")
    if base.is_array and not base.nonneg and exponent.dtype.value not in ("int", "bool"):
        ctx.assume("non-negative base or integer exponent")
    op = "mpower" if base.is_scalar and exponent.is_scalar else "power"
    return ctx.binop(op, ctx.expr(node.left), ctx.expr(node.right), loc=loc)
