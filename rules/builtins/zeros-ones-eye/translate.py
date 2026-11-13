# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/zeros-ones-eye.

``zeros(n)`` is square: ``np.zeros((n, n))``. A ``1xN`` or ``Nx1`` shape is a
1-D vector, ``np.zeros(n)``; ``zeros(size(A))`` is ``np.zeros(A.shape)``.
``eye`` keeps its arguments. Class arguments (``'int32'``) are declined.
"""

import ast

from dualect import ir
from dualect.emit import Context


def _is_one(node: ir.Expr, ctx: Context) -> bool:
    return ctx.info(node).value == 1


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``zeros``, ``ones`` or ``eye``."""
    args = node.args
    if any(isinstance(a, ir.Str) for a in args):
        ctx.decline(f"{node.func} with a class argument")
    func = ctx.np(node.func)
    if node.func == "eye":
        if len(args) == 2 and (_is_one(args[0], ctx) or _is_one(args[1], ctx)):
            ctx.decline("eye with a dimension of 1")
        if len(args) in (1, 2):
            return ctx.call(func, *(ctx.expr(a) for a in args))
        ctx.decline("eye with this many arguments")
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, ir.Call) and arg.func == "size" and len(arg.args) == 1:
            source = ctx.info(arg.args[0])
            if not (source.is_array or source.is_scalar):
                ctx.decline("zeros(size(x)) of an unknown x")
            return ctx.call(func, ast.Attribute(value=ctx.expr(arg.args[0]), attr="shape"))
        if not ctx.info(arg).is_scalar:
            ctx.decline(f"{node.func} with a size vector")
        n = ctx.expr(arg)
        return ctx.call(func, ast.Tuple(elts=[n, n]))
    if len(args) == 2:
        rows, cols = args
        if _is_one(rows, ctx):
            return ctx.call(func, ctx.expr(cols))
        if _is_one(cols, ctx):
            return ctx.call(func, ctx.expr(rows))
        if ctx.info(rows).value is None or ctx.info(cols).value is None:
            ctx.assume("neither dimension is 1")
        return ctx.call(func, ast.Tuple(elts=[ctx.expr(rows), ctx.expr(cols)]))
    ctx.decline(f"{node.func} with this many arguments")
