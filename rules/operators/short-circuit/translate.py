# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for operators/short-circuit: ``&&``, ``||``, ``&``, ``|``, ``xor``.

``&&`` / ``||`` are ``and`` / ``or``; a non-logical operand is wrapped in
``bool()`` so the result is a truth value, as in the source (Python's ``and``
returns an operand). ``&`` / ``|`` are NumPy's ``&`` / ``|`` on logical
operands and ``np.logical_and`` / ``np.logical_or`` otherwise, since ``&`` on
integers is bitwise.
"""

import ast

from dualect import ir
from dualect.emit import Context


def _truth(node: ir.Expr, ctx: Context) -> ast.expr:
    expr = ctx.expr(node)
    return expr if ctx.info(node).is_logical else ctx.call(ast.Name("bool"), expr)


def translate(node: ir.BinOp | ir.Call, ctx: Context) -> ast.expr:
    """One logical operation."""
    if isinstance(node, ir.Call):
        if node.func != "xor" or len(node.args) != 2:
            ctx.decline(f"call {node.func}")
        return ctx.call(ctx.np("logical_xor"), *(ctx.arg(a) for a in node.args))
    if node.op in ("andand", "oror"):
        op: ast.boolop = ast.And() if node.op == "andand" else ast.Or()
        return ast.BoolOp(op=op, values=[_truth(node.left, ctx), _truth(node.right, ctx)])
    left, right = ctx.info(node.left), ctx.info(node.right)
    if left.is_logical and right.is_logical:
        a, b = ctx.module("operators/arithmetic").expand(node, ctx)
        bit: ast.operator = ast.BitAnd() if node.op == "and" else ast.BitOr()
        return ast.BinOp(left=a, op=bit, right=b)
    func = "logical_and" if node.op == "and" else "logical_or"
    return ctx.call(ctx.np(func), ctx.arg(node.left), ctx.arg(node.right))
