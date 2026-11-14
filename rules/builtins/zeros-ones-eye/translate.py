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
from dualect.lift import LiftContext


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


# --- Python -> .m ------------------------------------------------------------
#
# np.zeros(n) is 1-D: a 1xN row, zeros(1, n) (zeros(n) would be n x n).
# np.zeros((m, n)) is zeros(m, n), np.zeros(A.shape) zeros(size(A)).
# np.eye keeps its arguments. A dtype is declined (bool and int arrays have
# no plain .m constructor here).


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.zeros``, ``np.ones`` or ``np.eye``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    func = (ctx.tag(node) or "").removeprefix("py:call:np.")
    if node.keywords:
        ctx.decline(f"np.{func} with a dtype or other options")
    args = node.args
    if func == "eye":
        if len(args) not in (1, 2):
            ctx.decline("np.eye with these arguments")
        return ctx.call("eye", *(ctx.expr(a) for a in args), loc=loc)
    if len(args) != 1:
        ctx.decline(f"np.{func} with these arguments")
    shape = args[0]
    info = ctx.info(shape)
    if isinstance(shape, ast.Attribute) and shape.attr == "shape":
        source = ctx.info(shape.value)
        if not (source.is_matrix or source.is_vector):
            ctx.decline(f"np.{func} of the shape of an unknown value")
        return ctx.call(func, ctx.call("size", ctx.expr(shape.value), loc=loc), loc=loc)
    if isinstance(shape, ast.Tuple):
        if len(shape.elts) == 1:
            return ctx.call(func, ctx.num(1), ctx.expr(shape.elts[0]), loc=loc)
        if len(shape.elts) == 2:
            return ctx.call(func, *(ctx.expr(e) for e in shape.elts), loc=loc)
        ctx.decline(f"np.{func} with more than two dimensions")
    if info.is_scalar:
        return ctx.call(func, ctx.num(1), ctx.expr(shape), loc=loc)
    ctx.decline(f"np.{func} with a shape of unknown form")
