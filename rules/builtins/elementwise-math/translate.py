# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/elementwise-math.

Most names map to their NumPy function; the renamed ones are listed in
:data:`_NUMPY`. ``abs`` of a known scalar is Python's ``abs``. ``round`` rounds
halves away from zero, written out from ``np.sign`` and ``np.floor``.
``floor(a / b)`` by a known nonzero scalar, of finite data, is ``a // b``.
``sqrt``/``log`` of a value not known to be non-negative is assumed to be in
the real domain (docs/design.md section 5.2): NumPy returns NaN where the
source returns a complex number, and verification catches that. ``mod(x, 0)``
and ``rem(x, 0)`` are ``x`` in the source but 0 or NaN in NumPy, so a zero
divisor is declined and an unknown one assumed nonzero, out loud.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, as_array

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
     "tanh", "hypot", "floor", "ceil", "sign", "isnan", "isinf", "isfinite", "mod", "real",
     "imag", "conj", "log1p", "expm1"}
)  # fmt: skip
_DOMAIN_LOG1P = frozenset({"log1p"})


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
    if name == "floor" and len(node.args) == 1:
        quotient = _floor_division(node.args[0], ctx)
        if quotient is not None:
            return quotient
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
    if name == "log1p" and isinstance(first.value, int | float) and first.value < -1:
        ctx.decline("log1p below -1 (complex in the source, NaN in NumPy)")
    if name == "log1p" and not first.nonneg:
        ctx.assume("log1p argument >= -1")
    args = [ctx.arg(arg) for arg in node.args]
    return ctx.call(ctx.np(_NUMPY.get(name, name)), *args)


def _floor_division(node: ir.Expr, ctx: Context) -> ast.expr | None:
    """``floor(a / b)`` as ``a // b``, when that is exactly the same value.

    Python's ``//`` floors like the source, for negative operands too, but
    ``inf // 2`` is NaN (the source gives Inf) and ``x // 0`` raises or is
    NaN, so both operands must be known finite and the divisor nonzero.
    """
    if not (isinstance(node, ir.BinOp) and node.op in ("rdivide", "mrdivide")):
        return None
    left, right = ctx.info(node.left), ctx.info(node.right)
    divisor = right.value
    if not (right.is_scalar and isinstance(divisor, int | float) and divisor != 0):
        return None
    if not (left.finite and right.finite) or not (left.is_scalar or left.is_array):
        return None
    if left.elem not in (Elem.REAL, Elem.LOGICAL) or left.is_logical:
        return None
    return ast.BinOp(left=ctx.expr(node.left), op=ast.FloorDiv(), right=ctx.expr(node.right))


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """Facts for the names the core analysis does not know (``log1p``, ``expm1``)."""
    if node.func not in ("log1p", "expm1") or len(args) != 1:
        return None
    first = args[0]
    if first.kind is Kind.UNKNOWN:
        return None
    elem = Elem.REAL if first.is_real and (node.func == "expm1" or first.nonneg) else first.elem
    if node.func == "log1p" and not first.nonneg:
        elem = Elem.UNKNOWN  # below -1 the source returns a complex number
    return replace(
        first,
        elem=elem,
        finite=first.finite and node.func == "log1p" and first.nonneg,
        nonneg=first.nonneg,
        integer=False,
        value=None,
        func=None,
        int_dtype=False,
    )


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
    "mod": "mod", "fmod": "rem", "hypot": "hypot", "round": "round", "real": "real",
    "imag": "imag", "log1p": "log1p", "expm1": "expm1",
}  # fmt: skip
_REAL_DOMAIN = frozenset({"sqrt", "log", "log2", "log10", "arcsin", "arccos", "log1p"})


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


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """``np.log1p`` and ``np.expm1``: element-wise, in floating point."""
    if tag not in ("py:call:np.log1p", "py:call:np.expm1") or len(args) != 1:
        return None
    first = as_array(args[0])
    if not (first.is_scalar or first.is_array):
        return None
    return replace(
        first,
        dtype=DType.COMPLEX if first.dtype is DType.COMPLEX else DType.FLOAT,
        finite=first.finite and first.nonneg,
        value=None,
        view=False,
    )
