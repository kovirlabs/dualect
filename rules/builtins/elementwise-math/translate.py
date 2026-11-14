# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/elementwise-math.

Most names map to their NumPy function; the renamed ones are listed in
:data:`_NUMPY`. ``abs`` of a known scalar is Python's ``abs``. ``round`` rounds
halves away from zero, written out from ``np.sign`` and ``np.floor``.
``sqrt``/``log`` of a value not known to be non-negative is assumed to be in
the real domain (docs/design.md section 5.2): NumPy returns NaN where the
source returns a complex number, and verification catches that. ``mod(x, 0)``
and ``rem(x, 0)`` are ``x`` in the source but 0 or NaN in NumPy, so a zero
divisor is declined and an unknown one assumed nonzero, out loud.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.lift import LiftContext

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
    if name in ("mod", "rem") and len(node.args) == 2:
        divisor = ctx.info(node.args[1]).value
        if divisor == 0:
            # mod(x, 0) and rem(x, 0) are x in the source; NumPy gives 0 or NaN.
            ctx.decline(f"{name} by zero")
        if divisor is None:
            ctx.assume(f"nonzero divisor in {name}")
    if name in _DOMAIN and not first.nonneg:
        ctx.assume(f"{name} argument in the real domain")
    args = [ctx.arg(arg) for arg in node.args]
    return ctx.call(ctx.np(_NUMPY.get(name, name)), *args)


# --- Python -> .m ------------------------------------------------------------
#
# The same functions under their .m names (np.arcsin is asin, np.trunc fix,
# np.fmod rem, Python's % mod). np.round rounds halves to even and the .m
# round away from zero, so it is only translated for integer data. sqrt/log
# of a negative number is NaN in NumPy but complex on the .m side: assumed
# out loud unless the argument is known to be non-negative.

_TO_M = {
    "abs": "abs", "sqrt": "sqrt", "exp": "exp", "log": "log", "log2": "log2",
    "log10": "log10", "sin": "sin", "cos": "cos", "tan": "tan", "arcsin": "asin",
    "arccos": "acos", "arctan": "atan", "arctan2": "atan2", "sinh": "sinh",
    "cosh": "cosh", "tanh": "tanh", "floor": "floor", "ceil": "ceil", "trunc": "fix",
    "sign": "sign", "isnan": "isnan", "isinf": "isinf", "isfinite": "isfinite",
    "mod": "mod", "fmod": "rem", "hypot": "hypot", "round": "round",
}  # fmt: skip
_REAL_DOMAIN = frozenset({"sqrt", "log", "log2", "log10", "arcsin", "arccos"})


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.<func>(x)``, ``abs(x)`` or ``a % b``."""
    loc = ctx.loc(node)
    literals = ctx.module_for("syntax/array-literals")
    if isinstance(node, ast.BinOp):  # a % b
        divisor = ctx.info(node.right).value
        if divisor == 0:
            ctx.decline("% by zero")
        if divisor is None:
            ctx.assume("nonzero divisor in %")
        return ctx.call("mod", ctx.expr(node.left), ctx.expr(node.right), loc=loc)
    assert isinstance(node, ast.Call)
    tag = ctx.tag(node) or ""
    name = tag.split(":")[-1].removeprefix("np.")
    if node.keywords or name not in _TO_M:
        ctx.decline(f"{tag} with these arguments")
    if not node.args:
        ctx.decline(f"{name} without arguments")
    first = ctx.info(node.args[0])
    if name == "round" and (len(node.args) != 1 or first.dtype.value not in ("int", "bool")):
        ctx.decline(
            "np.round of fractions (NumPy rounds halves to even, the .m round away from zero)"
        )
    if name in ("mod", "fmod") and len(node.args) == 2:
        divisor = ctx.info(node.args[1]).value
        if divisor == 0:
            ctx.decline(f"{name} by zero (the .m side returns the dividend)")
        if divisor is None:
            ctx.assume(f"nonzero divisor in {name}")
    if name in _REAL_DOMAIN and not first.nonneg:
        ctx.assume(f"{name} argument in the real domain")
    return ctx.call(_TO_M[name], *(literals.lift_items(a, ctx) for a in node.args), loc=loc)
