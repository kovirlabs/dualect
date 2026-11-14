# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/constants: ``pi``, ``Inf``, ``NaN``, ``eps``, ``true``, ...

Only the argument-free forms are constants; ``eps(x)``, ``Inf(2, 3)`` and
``true(n)`` build arrays and are declined.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.lift import LiftContext

_NUMPY = {"pi": "pi", "e": "e", "Inf": "inf", "inf": "inf", "NaN": "nan", "nan": "nan"}


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """One constant."""
    if node.args:
        ctx.decline(f"{node.func} with arguments")
    name = node.func
    if name in _NUMPY:
        return ctx.np(_NUMPY[name])
    if name == "eps":
        return ast.Attribute(value=ctx.call(ctx.np("finfo"), ast.Name("float")), attr="eps")
    if name in ("true", "false"):
        return ast.Constant(name == "true")
    if name in ("i", "j"):
        return ast.Constant(1j)
    ctx.decline(f"constant {name}")


# --- Python -> .m ------------------------------------------------------------
#
# np.pi is pi, np.inf Inf, np.nan NaN, np.finfo(float).eps eps, 2j is 2i.
# np.e is exp(1): a variable named e (common) would hide the function e.


_FROM_NUMPY = {"np.pi": "pi", "np.inf": "Inf", "np.nan": "NaN"}


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """A NumPy constant, machine epsilon or an imaginary literal."""
    loc = ctx.loc(node)
    if isinstance(node, ast.Constant) and isinstance(node.value, complex):
        return ir.Num(loc=loc, text="", value=node.value)
    tag = ctx.tag(node)
    if tag is not None and tag.startswith("py:const:"):
        name = tag[len("py:const:") :]
        if name in _FROM_NUMPY:
            return ctx.call(_FROM_NUMPY[name], loc=loc)
        if name == "np.e":
            return ctx.call("exp", ctx.num(1), loc=loc)
        ctx.decline(f"the NumPy name {name}")
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "eps"
        and isinstance(node.value, ast.Call)
        and ctx.tag(node.value) == "py:call:np.finfo"
        and [ast.unparse(a) for a in node.value.args] in (["float"], ["np.float64"])
    ):
        return ctx.call("eps", loc=loc)
    ctx.decline(f"{ast.unparse(node)}")
