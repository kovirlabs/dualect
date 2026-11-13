# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/linspace.

The count is always written out (the default is 100 here, 50 in NumPy), and a
count of 1 gives the *end* point, which NumPy does not.
"""

import ast

from dualect import ir
from dualect.emit import Context


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``linspace(a, b)`` or ``linspace(a, b, n)``."""
    if len(node.args) not in (2, 3):
        ctx.decline("linspace with this many arguments")
    start, stop = (ctx.expr(a) for a in node.args[:2])
    if len(node.args) == 2:
        return ctx.call(ctx.np("linspace"), start, stop, ast.Constant(100))
    count = ctx.info(node.args[2]).value
    if count == 1:
        end = ctx.info(node.args[1]).value
        if isinstance(end, int | float):
            return ctx.call(ctx.np("array"), ast.List(elts=[ctx.const(float(end))]))
        return ctx.call(ctx.np("array"), ast.List(elts=[stop]), dtype=ast.Name("float"))
    if count is None:
        ctx.assume("a point count other than 1")
    return ctx.call(ctx.np("linspace"), start, stop, ctx.expr(node.args[2]))
