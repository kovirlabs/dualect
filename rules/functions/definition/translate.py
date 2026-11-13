# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for functions/definition: ``function``, ``@(x) ...``, calls, ``return``.

A function becomes a ``def`` that returns its outputs; when the body ends by
assigning the single output, that assignment becomes the ``return``. Octave
passes arguments by value, so a parameter the body modifies in place is
copied first. Defaults filled in with ``if nargin < k`` become parameter
defaults (functions/nargin). An anonymous function becomes a ``lambda`` whose
captured variables are bound when it is created (``k=k``), because the source
captures values while a Python lambda looks names up when called.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.infer import Kind


def translate(node: ir.Node, ctx: Context) -> ast.expr | list[Line]:
    """Dispatch on the node kind."""
    if isinstance(node, ir.FunctionDef):
        return _function(node, ctx)
    if isinstance(node, ir.Lambda):
        return _lambda(node, ctx)
    if isinstance(node, ir.Call):
        return _call(node, ctx)
    if isinstance(node, ir.Return):
        outputs = ctx.scope.outputs
        if not outputs:
            return [ctx.line("return")]
        return [ctx.line(f"return {', '.join(outputs)}")]
    ctx.decline(f"{type(node).__name__}")


def _uses_nargin(fdef: ir.FunctionDef) -> bool:
    """Whether the function body reads ``nargin``."""
    for stmt in ir.walk_stmts(fdef.body):
        for expr in ir.stmt_exprs(stmt):
            for node in ir.walk_expr(expr):
                if isinstance(node, ir.Call) and node.func == "nargin":
                    return True
    return False


def _reads(name: str, body: list[ir.Stmt]) -> bool:
    """Whether ``name`` is read anywhere in ``body``."""
    for stmt in ir.walk_stmts(body):
        for expr in ir.stmt_exprs(stmt):
            if isinstance(stmt, ir.Assign) and expr in stmt.targets and isinstance(expr, ir.Name):
                continue
            for node in ir.walk_expr(expr):
                if isinstance(node, ir.Name) and node.id == name:
                    return True
                if (
                    isinstance(node, ir.Index)
                    and isinstance(node.target, ir.Name)
                    and node.target.id == name
                ):
                    return True
    return False


def _assignments(name: str, body: list[ir.Stmt]) -> int:
    count = 0
    for stmt in ir.walk_stmts(body):
        if isinstance(stmt, ir.Assign):
            count += sum(1 for t in stmt.targets if isinstance(t, ir.Name) and t.id == name)
            count += sum(
                1
                for t in stmt.targets
                if isinstance(t, ir.Index) and isinstance(t.target, ir.Name) and t.target.id == name
            )
        elif isinstance(stmt, ir.AugAssign | ir.For):
            target = stmt.target if isinstance(stmt, ir.AugAssign) else stmt.var
            if isinstance(target, ir.Name) and target.id == name:
                count += 1
    return count


def _function(fdef: ir.FunctionDef, ctx: Context) -> list[Line]:
    if "varargin" in fdef.params or "varargout" in fdef.outputs or "~" in fdef.params:
        ctx.decline("varargin, varargout or ignored parameters")
    if fdef.name in ctx.scope.variables:
        ctx.decline("a function named like a variable")
    defaults: dict[str, str] = {}
    body = fdef.body
    nargin_rule = None
    if _uses_nargin(fdef) and ctx.has_mechanical("functions/nargin"):
        nargin_rule = ctx.module("functions/nargin")
    with ctx.function_scope(fdef) as scope:
        if nargin_rule is not None:
            defaults, body = nargin_rule.extract_defaults(fdef, ctx)
        params = [f"{p}={defaults[p]}" if p in defaults else p for p in fdef.params]
        header = ctx.line(f"def {fdef.name}({', '.join(params)}):")
        copies: list[Line] = []
        facts = ctx.analysis.functions.get(fdef.name)
        for k, param in enumerate(fdef.params):
            if param not in scope.mutated:
                continue
            info = facts.params[k] if facts is not None and k < len(facts.params) else None
            if info is not None and (info.is_scalar or info.kind is Kind.STRING):
                continue
            # The source works on a private copy; NumPy would change the caller's array.
            copies.append(ctx.line(f"{param} = {param}.copy()", extra=1))
        outputs = fdef.outputs
        tail: ir.Assign | None = None
        last = body[-1] if body else None
        if (
            len(outputs) == 1
            and isinstance(last, ir.Assign)
            and len(last.targets) == 1
            and isinstance(last.targets[0], ir.Name)
            and last.targets[0].id == outputs[0]
            and outputs[0] not in fdef.params
            and _assignments(outputs[0], body) == 1
            and not _reads(outputs[0], body[:-1])
        ):
            tail = last
            body = body[:-1]
        lines = [header, *copies]
        ctx.indent += 1
        try:
            inner = ctx.statements(body)
            if tail is not None:
                tail_lines = ctx.statement(tail)
                code = [t for t in tail_lines if not t.text.startswith("#")]
                prefix = f"{outputs[0]} = "
                if (
                    len(code) == 1
                    and code[0].text.startswith(prefix)
                    and "raise" not in code[0].text
                ):
                    position = tail_lines.index(code[0])
                    tail_lines[position] = Line(
                        code[0].indent, "return " + code[0].text[len(prefix) :]
                    )
                    inner.extend(tail_lines)
                else:
                    inner.extend(tail_lines)
                    inner.append(ctx.line(f"return {outputs[0]}"))
            elif outputs:
                inner.append(ctx.line(f"return {', '.join(outputs)}"))
        finally:
            ctx.indent -= 1
        if not any(not line.text.startswith("#") for line in inner):
            inner.append(ctx.line("pass", extra=1))
        lines.extend(inner)
    return lines


def _lambda(node: ir.Lambda, ctx: Context) -> ast.expr:
    captured: list[str] = []
    for inner in ir.walk_expr(node.body):
        name = None
        if isinstance(inner, ir.Name):
            name = inner.id
        elif isinstance(inner, ir.Index) and isinstance(inner.target, ir.Name):
            name = inner.target.id
        if name is None or name in node.params or name in captured:
            continue
        if name in ctx.scope.variables:
            if name in ctx.scope.mutated or ctx.is_index_var(name):
                ctx.decline(f"a lambda capturing {name}, which is modified in place")
            captured.append(name)
    args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=p) for p in [*node.params, *captured]],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[ast.Name(id=c) for c in captured],
    )
    return ast.Lambda(args=args, body=ctx.expr(node.body))


def _call(node: ir.Call, ctx: Context) -> ast.expr:
    if node.handle:
        return ctx.call(ast.Name(node.func), *(ctx.expr(a) for a in node.args))
    fdef = ctx.functions.get(node.func)
    if fdef is None:
        ctx.decline(f"call to {node.func}")
    defaults: dict[str, str] = {}
    if _uses_nargin(fdef) and ctx.has_mechanical("functions/nargin"):
        defaults = ctx.module("functions/nargin").defaults_of(fdef, ctx)
    args: list[ast.expr] = []
    for k, arg in enumerate(node.args):
        param = fdef.params[k] if k < len(fdef.params) else None
        is_empty = isinstance(arg, ir.Matrix) and not arg.rows
        if param is not None and is_empty and defaults.get(param) == "None":
            args.append(ast.Constant(None))
        else:
            args.append(ctx.expr(arg))
    call = ctx.call(ast.Name(node.func), *args)
    if len(fdef.outputs) > 1:
        # Only the first output is used here; Python returns them all as a tuple.
        return ast.Subscript(value=call, slice=ast.Constant(0))
    return call
