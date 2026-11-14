# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for functions/multiple-returns.

``.m`` -> Python: a function with several outputs returns them as one tuple
(functions/definition writes ``return s, p``); ``[s, p] = f(v)`` unpacks it
(``s, p = f(v)``), ``[~, p] = f(v)`` unpacks into ``_``, and taking fewer
outputs than the function has slices the tuple (``[a, b] = f(x)`` of a
three-output ``f`` is ``a, b = f(x)[:2]``); a single output is
``f(v)[0]`` (functions/definition). A function whose body depends on
``nargout`` is functions/nargout's business (guided).

Python -> ``.m``: ``return a, b`` gives the function the outputs ``[a, b]``
(functions/definition picks the names); ``s, p = f(v)`` is
``[s, p] = f(v)``; ``f(v)[0]`` is ``f(v)``, the first output; ``r, c =
A.shape`` is ``[r, c] = size(A)`` (builtins/size-numel-length), and a
built-in with several results goes to its own rule's ``lift_multi``. Unpacking
a tuple literal (``a, b = b, a``) evaluates every value before assigning,
which a sequence of ``.m`` assignments does not, so it is declined.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.lift import LiftContext


def translate(node: ir.Node, ctx: Context) -> list[Line]:
    """``[a, b] = f(x)`` for a function defined in the file."""
    if not isinstance(node, ir.Assign):
        ctx.decline("this multiple-output form")
    call = node.value
    if not (isinstance(call, ir.Call) and call.local):
        ctx.decline("several outputs of something other than a function in this file")
    fdef = ctx.functions.get(call.func)
    if fdef is None:
        ctx.decline(f"several outputs of {call.func}")
    if len(node.targets) > len(fdef.outputs):
        ctx.decline(f"more outputs than {call.func} has")
    names: list[str] = []
    for target in node.targets:
        if isinstance(target, ir.Ignore):
            names.append("_")
        elif isinstance(target, ir.Name) and not ctx.is_index_var(target.id):
            names.append(target.id)
        else:
            ctx.decline("an output assigned into an element or stored 0-based")
    # The call itself, as functions/definition writes it (defaults, None for []).
    single = ctx.module("functions/definition").translate(call, ctx)
    result = single.value if isinstance(single, ast.Subscript) else single
    if len(node.targets) < len(fdef.outputs):
        result = ast.Subscript(value=result, slice=ast.Slice(upper=ast.Constant(len(node.targets))))
    return [ctx.line(f"{', '.join(names)} = {ast.unparse(result)}")]


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
    tag = ctx.tag(value) if isinstance(value, ast.Call) else None
    rule = ctx.index.rule_for(tag) if tag is not None else None
    if rule is not None and rule.id != "functions/multiple-returns" and tag != "py:call-local":
        # A built-in with several results (np.nonzero, np.meshgrid, ...) is
        # its own rule's business: that rule's lift_multi.
        hook = getattr(ctx.module_for(rule.id), "lift_multi", None)
        if hook is None:
            ctx.decline(f"unpacking the result of {tag}")
        return list(hook(stmt, ctx))
    taken: int | None = None
    if isinstance(value, ast.Subscript):
        # f(x)[:2]: the first two outputs, which is what [a, b] = f(x) takes.
        bound = value.slice
        if not (
            isinstance(bound, ast.Slice)
            and bound.lower is None
            and bound.step is None
            and isinstance(bound.upper, ast.Constant)
            and type(bound.upper.value) is int
        ):
            ctx.decline("unpacking a slice of the outputs other than the first ones")
        taken = bound.upper.value
        value = value.value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in ctx.functions
        and value.func.id not in ctx.scope.variables
    ):
        ctx.decline("unpacking a value that is not a call of a local function")
    returned = _outputs(value.func.id, ctx)
    if (taken if taken is not None else returned) != len(target.elts) or len(
        target.elts
    ) > returned:
        ctx.decline("unpacking a different number of values than the function returns")
    names: list[ir.Expr] = []
    for element in target.elts:
        if not isinstance(element, ast.Name):
            ctx.decline("unpacking into an element")
        if element.id == "_":
            names.append(ir.Ignore(loc=ctx.loc(element)))  # a discarded output: ~
        else:
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
