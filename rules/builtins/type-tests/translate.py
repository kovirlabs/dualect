# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/type-tests: ``isnumeric``, ``islogical``, ``ischar``.

Each becomes the runtime check that answers the same question of the Python
value, rather than an answer computed from what the translation knows:
``ischar(s)`` is ``isinstance(s, str)``, ``islogical(x)`` is
``np.asarray(x).dtype == bool``, ``isnumeric(x)`` is
``np.issubdtype(np.asarray(x).dtype, np.number)`` (NumPy's bool is not a
number, as logicals are not numeric in the source). A string's array is
``<U`` dtype, neither bool nor a number, so those two hold for strings too.
``ischar`` of a value of unknown type is declined (a char matrix would not be
a ``str``), and so is any test of a cell array or a function handle.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind, scalar


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """One type test."""
    if len(node.args) != 1:
        ctx.decline(f"{node.func} with these arguments")
    info = ctx.info(node.args[0])
    if info.kind in (Kind.CELL, Kind.FUNC):
        ctx.decline(f"{node.func} of a cell array or a function handle")
    expr = ctx.expr(node.args[0])
    if node.func == "ischar":
        if info.kind is Kind.UNKNOWN and info.elem is Elem.UNKNOWN:
            ctx.decline("ischar of a value of unknown type")
        return ctx.call(ast.Name("isinstance"), expr, ast.Name("str"))
    dtype = ast.Attribute(value=ctx.call(ctx.np("asarray"), expr), attr="dtype")
    if node.func == "islogical":
        return ast.Compare(left=dtype, ops=[ast.Eq()], comparators=[ast.Name("bool")])
    return ctx.call(ctx.np("issubdtype"), dtype, ctx.np("number"))


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """A logical scalar."""
    return scalar(elem=Elem.LOGICAL, nonneg=True, integer=True)
