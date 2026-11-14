# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for types/double: ``double(x)`` and ``logical(x)``.

``double`` of a value that is already a number is that value; of a logical
array it is ``x.astype(float)`` (of a scalar, ``float(x)``). ``logical(x)``
is ``x.astype(bool)`` (``bool(x)`` for a scalar). Character data is declined:
it converts to character codes, which Python does not do implicitly.

From Python, ``x.astype(float)`` and ``float(x)`` are ``double(x)``, and
``x.astype(bool)`` is ``logical(x)`` for data known to hold no NaN (NumPy
makes NaN true, the ``.m`` side raises). Integer types are declined: they
round and saturate on the ``.m`` side and truncate and wrap in NumPy.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Kind
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyKind


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``double(x)`` or ``logical(x)``."""
    if len(node.args) != 1:
        ctx.decline(f"{node.func} with these arguments")
    info = ctx.info(node.args[0])
    if info.kind is Kind.STRING or info.elem in (Elem.CHAR, Elem.COMPLEX):
        ctx.decline(f"{node.func} of characters or complex values")
    if info.elem is Elem.UNKNOWN or not (info.is_scalar or info.is_array):
        ctx.decline(f"{node.func} of a value of unknown type")
    expr = ctx.expr(node.args[0])
    if node.func == "double":
        if info.elem is Elem.REAL:
            return expr  # every plain number is a double already
        if info.is_scalar:
            return ctx.call(ast.Name("float"), expr)
        return ctx.method(expr, "astype", ast.Name("float"))
    if info.elem is Elem.LOGICAL:
        return expr
    if not info.finite:
        ctx.assume("no NaN in logical() (NumPy makes NaN true; the source raises)")
    if info.is_scalar:
        return ctx.call(ast.Name("bool"), expr)
    return ctx.method(expr, "astype", ast.Name("bool"))


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``x.astype(float)``, ``x.astype(bool)``, ``float(x)``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if node.keywords or len(node.args) != 1:
        ctx.decline("a conversion with these arguments")
    if isinstance(node.func, ast.Name):  # float(x)
        info = ctx.info(node.args[0])
        if info.kind is PyKind.STRING or not info.is_scalar:
            ctx.decline("float() of a string or of a value that is not a number")
        return ctx.call("double", ctx.expr(node.args[0]), loc=loc)
    assert isinstance(node.func, ast.Attribute)
    receiver = node.func.value
    target = node.args[0]
    name = target.id if isinstance(target, ast.Name) else None
    info = ctx.info(receiver)
    if not info.is_array or info.dtype is DType.COMPLEX:
        ctx.decline("astype of a value that is not a real array")
    value = ctx.module_for("syntax/array-literals").lift_items(receiver, ctx)
    if name == "float":
        return ctx.call("double", value, loc=loc)
    if name == "bool":
        if not info.finite:
            ctx.decline("astype(bool) of data that may hold NaN (the .m logical() raises)")
        return ctx.call("logical", value, loc=loc)
    ctx.decline(f"astype({ast.unparse(target)}) (integer types round and saturate in .m code)")
