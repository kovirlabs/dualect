# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/constants: ``pi``, ``Inf``, ``NaN``, ``eps``, ``true``, ...

Only the argument-free forms are constants; ``eps(x)``, ``Inf(2, 3)`` and
``true(n)`` build arrays and are declined.
"""

import ast

from dualect import ir
from dualect.emit import Context

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
