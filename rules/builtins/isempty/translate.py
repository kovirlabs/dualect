# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/isempty: ``x.size == 0``, ``len(s) == 0`` for strings."""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Kind


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``isempty(x)``."""
    if len(node.args) != 1:
        ctx.decline("isempty with this many arguments")
    info = ctx.info(node.args[0])
    expr = ctx.expr(node.args[0])
    if info.kind is Kind.STRING:
        size = ctx.call(ast.Name("len"), expr)
    elif info.is_array:
        size = ast.Attribute(value=expr, attr="size")
    elif info.is_scalar:
        size = ctx.call(ctx.np("size"), expr)
    else:
        ctx.decline("isempty of a value of unknown type")
    return ast.Compare(left=size, ops=[ast.Eq()], comparators=[ast.Constant(0)])
