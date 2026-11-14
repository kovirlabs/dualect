# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/linspace.

The count is always written out (the default is 100 here, 50 in NumPy), and a
count of 1 gives the *end* point, which NumPy does not.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.lift import LiftContext


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


# --- Python -> .m ------------------------------------------------------------
#
# np.linspace(a, b, n) is linspace(a, b, n). NumPy's default count is 50 (the
# .m default is 100), so it is always written out; a count of 1 gives the
# start in NumPy and the end on the .m side, so it is declined.


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.linspace``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    args = list(node.args)
    for keyword in node.keywords:
        if keyword.arg == "num" and len(args) == 2:
            args.append(keyword.value)
        else:
            ctx.decline(f"np.linspace with {keyword.arg}=")
    if len(args) == 2:
        count: ir.Expr = ctx.num(50)
        value: object = 50
    elif len(args) == 3:
        count = ctx.expr(args[2])
        value = ctx.info(args[2]).value
    else:
        ctx.decline("np.linspace with these arguments")
    if value == 1:
        ctx.decline("a count of 1 (NumPy gives the start point, the .m side the end point)")
    if value is None:
        ctx.assume("a point count other than 1")
    return ctx.call("linspace", ctx.expr(args[0]), ctx.expr(args[1]), count, loc=loc)
