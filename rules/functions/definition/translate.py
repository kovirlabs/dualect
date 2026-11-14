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
from dualect.lift import LiftContext


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
            # `if p is None: p = <default>` for defaults that are not literals.
            inner = nargin_rule.prologue(fdef, ctx) if nargin_rule is not None else []
            inner.extend(ctx.statements(body))
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


# --- Python -> .m ------------------------------------------------------------
#
# A def becomes a function whose output is the returned variable (or `out`).
# Three Python behaviours have no .m counterpart and are declined rather than
# translated as if they did not exist: a function that writes into an array
# argument changes the caller's array (unless it copies it first); a function
# that reads a module-level variable sees it, where a .m function does not;
# and a lambda looks its free variables up when called, where an .m handle
# captures their values when created (a lambda is translated when each free
# variable is assigned once, or captured with a `k=k` default).


def _returns(body: list[ast.stmt]) -> list[ast.Return]:
    """Every ``return`` of a function body (nested functions and lambdas excluded)."""
    found: list[ast.Return] = []
    for stmt in body:
        if isinstance(stmt, ast.Return):
            found.append(stmt)
        elif not isinstance(stmt, ast.FunctionDef | ast.ClassDef):
            for field in ("body", "orelse", "finalbody"):
                found.extend(_returns(getattr(stmt, field, []) or []))
            for case in getattr(stmt, "cases", []) or []:
                found.extend(_returns(case.body))
    return found


def _fresh(base: str, taken: set[str]) -> str:
    name = base
    while name in taken:
        name += "_"
    return name


def output_names(fdef: ast.FunctionDef, ctx: LiftContext) -> list[str]:
    """The ``.m`` output variables of a ``def``, from its ``return`` statements."""
    returns = _returns(fdef.body)
    values = [r.value for r in returns]
    if not returns or all(v is None for v in values):
        return []
    if any(v is None for v in values):
        ctx.decline("a function that returns a value on some paths only")
    if not isinstance(fdef.body[-1], ast.Return):
        ctx.decline("a function whose last statement is not its return")
    taken = {a.arg for a in fdef.args.args} | {
        n.id for n in ast.walk(fdef) if isinstance(n, ast.Name)
    }
    tuples = [v for v in values if isinstance(v, ast.Tuple)]
    if tuples:
        ctx.module_for("functions/multiple-returns")
        sizes = {len(v.elts) for v in values if isinstance(v, ast.Tuple)}
        if len(tuples) != len(values) or len(sizes) != 1:
            ctx.decline("returns of different arity")
        first = tuples[0].elts
        if all(
            all(isinstance(e, ast.Name) for e in t.elts)
            and [e.id for e in t.elts if isinstance(e, ast.Name)]
            == [getattr(x, "id", None) for x in first]
            for t in tuples
        ):
            return [getattr(x, "id", "") for x in first]
        return [_fresh(f"out{k + 1}", taken) for k in range(sizes.pop())]
    names = {v.id if isinstance(v, ast.Name) else None for v in values}
    if len(names) == 1 and None not in names:
        name = names.pop()
        assert name is not None
        return [name]
    return [_fresh("out", taken)]


def _bindings(name: str, body: list[ast.stmt]) -> int:
    """How many statements of a scope bind ``name``."""
    count = 0
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.FunctionDef):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name:
            count += 1
    return count


def _lift_def(fdef: ast.FunctionDef, ctx: LiftContext) -> list[ir.Stmt]:
    if ctx.function is not None or fdef not in ctx.module.body:
        ctx.decline("a function defined inside another statement")
    params = [a.arg for a in fdef.args.args]
    outputs = output_names(fdef, ctx)
    with ctx.function_scope(fdef, outputs) as scope:
        local = scope.variables
        for node in ast.walk(fdef):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in local
                and node.id in ctx.module_scope.variables
            ):
                ctx.decline(f"a function reading the module variable {node.id}")
        facts = ctx.analysis.functions.get(fdef.name)
        for k, param in enumerate(params):
            if param not in scope.mutated:
                continue
            info = facts.params[k] if facts is not None else None
            if info is not None and (info.is_scalar or info.kind.value == "string"):
                continue
            if not _copied_before_written(param, fdef.body, ctx):
                ctx.decline(f"writing into the argument {param} (NumPy changes the caller's array)")
        prologue: list[ir.Stmt] = []
        if fdef.args.defaults:
            # Defaults are `if nargin < k` blocks (functions/nargin).
            prologue = ctx.module_for("functions/nargin").lift_defaults(fdef, ctx)
        body = [*prologue, *ctx.block(fdef.body)]
    return [
        ir.FunctionDef(loc=ctx.loc(fdef), name=fdef.name, params=params, outputs=outputs, body=body)
    ]


def _writes(stmt: ast.stmt, name: str) -> bool:
    """Whether a statement writes into the array ``name`` in place."""
    for node in ast.walk(stmt):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Subscript)]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            root = target.value if isinstance(target, ast.Subscript) else target
            while isinstance(root, ast.Subscript | ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id == name:
                return True
    return False


def _copied_before_written(param: str, body: list[ast.stmt], ctx: LiftContext) -> bool:
    """``param = param.copy()`` (or any fresh value) at the top, before the first write."""
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == param
            and not ctx.info(stmt.value).view
            and not _writes(stmt, param)
        ):
            return True
        if _writes(stmt, param):
            return False
    return True


def _lift_return(stmt: ast.Return, ctx: LiftContext) -> list[ir.Stmt]:
    fdef = ctx.function
    if fdef is None:
        ctx.decline("return outside a function")
    loc = ctx.loc(stmt)
    lifted: list[ir.Stmt] = []
    values: list[ast.expr] = []
    if isinstance(stmt.value, ast.Tuple):
        values = list(stmt.value.elts)
    elif stmt.value is not None:
        values = [stmt.value]
    for output, value in zip(ctx.outputs, values, strict=True):
        if isinstance(value, ast.Name) and value.id == output:
            continue
        lifted.append(
            ir.Assign(loc=loc, targets=[ir.Name(loc=loc, id=output)], value=ctx.expr(value))
        )
    if fdef.body[-1] is not stmt:
        lifted.append(ir.Return(loc=loc))
    return lifted


def _lift_lambda(node: ast.Lambda, ctx: LiftContext) -> ir.Expr:
    args = node.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        ctx.decline("a lambda with variable or keyword-only arguments")
    names = [a.arg for a in args.args]
    first_default = len(names) - len(args.defaults)
    captured: set[str] = set()
    for name, default in zip(names[first_default:], args.defaults, strict=True):
        if not (isinstance(default, ast.Name) and default.id == name):
            ctx.decline("a lambda parameter with a default value")
        captured.add(name)
    params = names[:first_default]
    body_scope: list[ast.stmt] = ctx.function.body if ctx.function is not None else ctx.module.body
    for inner in ast.walk(node.body):
        if not isinstance(inner, ast.Name) or inner.id in params:
            continue
        name = inner.id
        if name in captured or name not in ctx.scope.variables:
            continue
        if name in ctx.scope.mutated or _bindings(name, body_scope) > 1:
            ctx.decline(
                f"a lambda reading {name}, which changes later (Python reads it when called)"
            )
    return ir.Lambda(loc=ctx.loc(node), params=params, body=ctx.expr(node.body))


def _lift_call(node: ast.Call, ctx: LiftContext) -> ir.Expr:
    loc = ctx.loc(node)
    if node.keywords:
        ctx.decline("keyword arguments")
    func = node.func
    assert isinstance(func, ast.Name)
    args = [ctx.module_for("syntax/array-literals").lift_items(a, ctx) for a in node.args]
    fdef = ctx.functions.get(func.id)
    if fdef is None or func.id in ctx.scope.variables:
        return ir.Call(loc=loc, func=func.id, args=args, handle=True)
    required = len(fdef.args.args) - len(fdef.args.defaults)
    if not required <= len(node.args) <= len(fdef.args.args):
        ctx.decline("a call with the wrong number of arguments")
    if len(node.args) < len(fdef.args.args):
        # The omitted arguments take their defaults: `if nargin < k` on the .m side.
        ctx.module_for("functions/nargin")
    returns = _returns(fdef.body)
    if any(isinstance(r.value, ast.Tuple) for r in returns):
        ctx.decline("using the whole tuple a function returns")
    return ir.Call(loc=loc, func=func.id, args=args, local=True)


def lift(node: ast.AST, ctx: LiftContext) -> ir.Expr | list[ir.Stmt]:
    """A ``def``, ``return``, ``lambda`` or a call of a local function or a lambda."""
    if isinstance(node, ast.FunctionDef):
        return _lift_def(node, ctx)
    if isinstance(node, ast.Return):
        return _lift_return(node, ctx)
    if isinstance(node, ast.Lambda):
        return _lift_lambda(node, ctx)
    if isinstance(node, ast.Call):
        return _lift_call(node, ctx)
    ctx.decline(f"{type(node).__name__}")
