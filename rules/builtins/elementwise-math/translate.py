# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/elementwise-math.

Most names map to their NumPy function; the renamed ones are listed in
:data:`_NUMPY`. ``abs`` of a known scalar is Python's ``abs``. ``round`` rounds
halves away from zero, written out from ``np.sign`` and ``np.floor``.
``sqrt``/``log`` of a value not known to be non-negative is assumed to be in
the real domain (docs/design.md section 5.2): NumPy returns NaN where the
source returns a complex number, and verification catches that.
"""

import ast

from dualect import ir
from dualect.emit import Context

_NUMPY = {
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "atan2": "arctan2",
    "fix": "trunc",
    "rem": "fmod",
}
_DOMAIN = frozenset({"sqrt", "log", "log2", "log10", "asin", "acos"})
_SAME = frozenset(
    {"abs", "sqrt", "exp", "log", "log2", "log10", "sin", "cos", "tan", "sinh", "cosh",
     "tanh", "hypot", "floor", "ceil", "sign", "isnan", "isinf", "isfinite", "mod"}
)  # fmt: skip


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """One element-wise function call."""
    name = node.func
    if not node.args:
        ctx.decline(f"{name} without arguments")
    first = ctx.info(node.args[0])
    if name == "round":
        if len(node.args) != 1 or not isinstance(node.args[0], ir.Name):
            ctx.decline("round of an expression (half away from zero repeats it)")
        x = ctx.expr(node.args[0])
        floor = ctx.call(
            ctx.np("floor"),
            ast.BinOp(left=ctx.call(ctx.np("abs"), x), op=ast.Add(), right=ast.Constant(0.5)),
        )
        return ast.BinOp(left=ctx.call(ctx.np("sign"), x), op=ast.Mult(), right=floor)
    if name == "abs" and first.is_scalar and len(node.args) == 1:
        return ctx.call(ast.Name("abs"), ctx.expr(node.args[0]))
    if name not in _SAME and name not in _NUMPY:
        ctx.decline(f"function {name}")
    if name in _DOMAIN and not first.nonneg:
        ctx.assume(f"{name} argument in the real domain")
    args = [ctx.arg(arg) for arg in node.args]
    return ctx.call(ctx.np(_NUMPY.get(name, name)), *args)
