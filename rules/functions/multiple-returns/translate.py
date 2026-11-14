# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for functions/multiple-returns (Python -> ``.m`` only, for now).

``.m`` -> Python is still ``guided``: ``nargout``-dependent bodies would have to
be collapsed into one tuple return (docs/design.md section 11).

Python -> ``.m``: ``return a, b`` gives the function the outputs ``[a, b]``
(functions/definition picks the names); ``s, p = f(v)`` is
``[s, p] = f(v)``; ``f(v)[0]`` is ``f(v)``, the first output; ``r, c =
A.shape`` is ``[r, c] = size(A)`` (builtins/size-numel-length), and
``r, c = np.nonzero(A)`` goes to builtins/find. Unpacking a
tuple literal (``a, b = b, a``) evaluates every value before assigning,
which a sequence of ``.m`` assignments does not, so it is declined.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.lift import LiftContext


def translate(node: ir.Node, ctx: Context) -> None:
    """The ``.m`` -> Python direction is guided: never called for this rule."""
    ctx.decline("multiple returns are translated by an LLM in this direction")


def _outputs(name: str, ctx: LiftContext) -> int:
    fdef = ctx.functions[name]
    definition = ctx.module_for("functions/definition")
    returns = definition._returns(fdef.body)
    counts = {len(r.value.elts) if isinstance(r.value, ast.Tuple) else 1 for r in returns}
    if len(counts) != 1:
        ctx.decline(f"{name} returns different numbers of values")
    return counts.pop()


def lift_multi(stmt: ast.Assign, ctx: LiftContext) -> list[ir.Stmt]:
    """``a, b = f(x)`` or ``r, c = A.shape``."""
    target = stmt.targets[0]
    value = stmt.value
    assert isinstance(target, ast.Tuple | ast.List)
    if isinstance(value, ast.Attribute) and value.attr == "shape":
        return ctx.module_for("builtins/size-numel-length").lift_multi(stmt, ctx)
    if isinstance(value, ast.Call) and ctx.tag(value) == "py:call:np.nonzero":
        return ctx.module_for("builtins/find").lift_multi(stmt, ctx)
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in ctx.functions
        and value.func.id not in ctx.scope.variables
    ):
        ctx.decline("unpacking a value that is not a call of a local function")
    if _outputs(value.func.id, ctx) != len(target.elts):
        ctx.decline("unpacking a different number of values than the function returns")
    names: list[ir.Expr] = []
    for element in target.elts:
        if not isinstance(element, ast.Name):
            ctx.decline("unpacking into an element")
        names.append(ir.Name(loc=ctx.loc(element), id=element.id))
    if value.keywords:
        ctx.decline("keyword arguments")
    literals = ctx.module_for("syntax/array-literals")
    call = ir.Call(
        loc=ctx.loc(value),
        func=value.func.id,
        args=[literals.lift_items(a, ctx) for a in value.args],
        local=True,
    )
    return [ir.Assign(loc=ctx.loc(stmt), targets=names, value=call)]


def lift_item(node: ast.Subscript, ctx: LiftContext) -> ir.Expr:
    """``f(x)[0]``: the first output, which a ``.m`` call returns by itself."""
    call = node.value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    if not (isinstance(node.slice, ast.Constant) and node.slice.value == 0):
        ctx.decline("an output other than the first")
    if _outputs(call.func.id, ctx) < 2:
        ctx.decline("indexing the value a function returns")
    if call.keywords:
        ctx.decline("keyword arguments")
    literals = ctx.module_for("syntax/array-literals")
    return ir.Call(
        loc=ctx.loc(node),
        func=call.func.id,
        args=[literals.lift_items(a, ctx) for a in call.args],
        local=True,
    )


def lift(node: ast.AST, ctx: LiftContext) -> ir.Expr:
    """Never reached directly: the forms above are dispatched by the engine and rules."""
    ctx.decline("this use of several return values")
