# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for operators/power: ``^`` and ``.^``.

``.^`` is ``**``. ``^`` is ``**`` between scalars and
``np.linalg.matrix_power`` for a square matrix and an integer exponent. A
negative integer exponent on an integer array is written as a float
(``x**-1.0``) because NumPy rejects negative integer powers of integers.
"""

import ast

from dualect import ir
from dualect.emit import Context, int_constant


def _float_exponent(exponent: ast.expr) -> ast.expr:
    value = int_constant(exponent)
    if value is not None and value < 0:
        return Context.const(float(value))
    return exponent


def translate(node: ir.BinOp, ctx: Context) -> ast.expr:
    """One power."""
    base, exponent = ctx.info(node.left), ctx.info(node.right)
    if node.op == "power":
        a, b = ctx.module("operators/arithmetic").expand(node, ctx)
        if base.is_array and base.int_dtype:
            b = _float_exponent(b)
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
