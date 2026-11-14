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
from dualect.lift import LiftContext


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


# --- Python -> .m ------------------------------------------------------------
#
# `and`/`or` return one of their operands in Python; && and || return a
# logical. They agree when the operands are booleans, or when only the truth
# of the result is used (an if/while test). & and | are the element-wise
# operators on booleans (bitwise on integers: declined). np.logical_and/or
# are & and |, np.logical_xor is xor().


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``and``, ``or``, ``&``, ``|`` and the ``np.logical_*`` functions."""
    loc = ctx.loc(node)
    if isinstance(node, ast.BoolOp):
        truth_only = ctx.in_condition
        infos = [ctx.info(v) for v in node.values]
        if any(i.is_array for i in infos):
            ctx.decline("`and`/`or` of arrays")
        if not truth_only and not all(i.is_bool for i in infos):
            ctx.decline("`and`/`or` whose value (an operand, not a logical) is used")
        op = "andand" if isinstance(node.op, ast.And) else "oror"
        values = [ctx.expr(v) for v in node.values]
        result = values[0]
        for value in values[1:]:
            result = ctx.binop(op, result, value, loc=loc)
        return result
    literals = ctx.module_for("syntax/array-literals")
    if isinstance(node, ast.Call):
        name = ctx.tag(node)
        if len(node.args) != 2 or node.keywords:
            ctx.decline(f"{name} with these arguments")
        a, b = (literals.lift_items(arg, ctx) for arg in node.args)
        if name == "py:call:np.logical_xor":
            return ctx.call("xor", a, b, loc=loc)
        op = "and" if name == "py:call:np.logical_and" else "or"
        return ctx.binop(op, a, b, loc=loc)
    assert isinstance(node, ast.BinOp)
    if not (ctx.info(node.left).is_bool and ctx.info(node.right).is_bool):
        ctx.decline("& or | of values not known to be boolean (bitwise on integers)")
    op = "and" if isinstance(node.op, ast.BitAnd) else "or"
    return ctx.binop(op, ctx.expr(node.left), ctx.expr(node.right), loc=loc)
