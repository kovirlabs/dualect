# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for functions/nargin: the default-argument pattern.

``.m`` -> Python. A function whose body *starts* with blocks of the form::

    if nargin < k               % or: if nargin < k || isempty(p)
      p = <default>;
    end

where ``p`` is the ``k``-th parameter gets Python defaults (and the other way
round: see the Python -> .m section). A literal number or
string default goes into the header (``def f(x, factor=10)``); any other
default (one that depends on another argument, an array, or the ``isempty``
form, which lets callers skip an argument with ``[]``) becomes ``p=None`` and
``if p is None: p = <default>`` at the top of the body, and a call in the file
that passes a literal ``[]`` for such a parameter passes ``None``
(functions/definition asks :func:`defaults_of`).

Declined, so never silently wrong: ``nargin`` anywhere else; a block that does
more than fill in its parameter or fills in a different one; ``isempty(p) ||
nargin < k`` (the source would fail when ``p`` is missing); and the ``isempty``
form when a call in the file passes a value that may be empty at run time,
which Python's ``is None`` test would not catch.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.infer import Kind
from dualect.lift import LiftContext
from dualect.pyinfer import PyKind


def translate(node: ir.Node, ctx: Context) -> ast.expr:
    """``nargin`` outside the default pattern has no Python counterpart."""
    ctx.decline("nargin used other than to fill in a missing argument")


def _nargin_test(cond: ir.Expr) -> int | None:
    """``k`` of ``nargin < k`` (also ``k - 1 >= nargin``-free forms only)."""
    if (
        isinstance(cond, ir.BinOp)
        and cond.op == "lt"
        and isinstance(cond.left, ir.Call)
        and cond.left.func == "nargin"
        and isinstance(cond.right, ir.Num)
        and isinstance(cond.right.value, int)
    ):
        return cond.right.value
    return None


def _pattern(stmt: ir.Stmt, fdef: ir.FunctionDef) -> tuple[str, bool, list[ir.Stmt]] | None:
    """``(parameter, uses isempty, body)`` when ``stmt`` is one default block."""
    if not (isinstance(stmt, ir.If) and len(stmt.branches) == 1 and stmt.orelse is None):
        return None
    cond, body = stmt.branches[0]
    empty_test: str | None = None
    if isinstance(cond, ir.BinOp) and cond.op == "oror":
        right = cond.right
        if not (
            isinstance(right, ir.Call)
            and right.func == "isempty"
            and len(right.args) == 1
            and isinstance(right.args[0], ir.Name)
        ):
            return None
        empty_test = right.args[0].id
        cond = cond.left
    k = _nargin_test(cond)
    if k is None or not 1 <= k <= len(fdef.params):
        return None
    param = fdef.params[k - 1]
    if empty_test is not None and empty_test != param:
        return None
    code = [s for s in body if not isinstance(s, ir.Comment)]
    if not code or any(_uses_nargin(s) for s in code):
        return None
    last = code[-1]
    if not (
        isinstance(last, ir.Assign)
        and len(last.targets) == 1
        and isinstance(last.targets[0], ir.Name)
        and last.targets[0].id == param
    ):
        return None
    for other in code[:-1]:
        # Helper statements may only set new local names, never a parameter.
        if not isinstance(other, ir.Assign) or any(
            not isinstance(t, ir.Name) or t.id in fdef.params for t in other.targets
        ):
            return None
    return param, empty_test is not None, body


def _uses_nargin(stmt: ir.Stmt) -> bool:
    for inner in ir.walk_stmts([stmt]):
        for expr in ir.stmt_exprs(inner):
            for node in ir.walk_expr(expr):
                if isinstance(node, ir.Call) and node.func == "nargin":
                    return True
    return False


def _literal(stmt: ir.Assign) -> ast.expr | None:
    """The default as a header literal: a number or a string, nothing mutable."""
    value = stmt.value
    if isinstance(value, ir.UnOp) and value.op == "uminus" and isinstance(value.operand, ir.Num):
        number = value.operand.value
        return Context.const(-number) if isinstance(number, int | float) else None
    if isinstance(value, ir.Num) and isinstance(value.value, int | float):
        return Context.const(value.value)
    if isinstance(value, ir.Str):
        return ast.Constant(value.value)
    return None


def _blocks(fdef: ir.FunctionDef) -> tuple[list[tuple[str, bool, list[ir.Stmt]]], list[ir.Stmt]]:
    """The leading default blocks and the rest of the body."""
    found: list[tuple[str, bool, list[ir.Stmt]]] = []
    rest = list(fdef.body)
    while rest:
        head = rest[0]
        if isinstance(head, ir.Comment):
            rest = rest[1:]
            continue
        match = _pattern(head, fdef)
        if match is None:
            break
        found.append(match)
        rest = rest[1:]
    return found, rest


def _call_sites(name: str, ctx: Context) -> list[ir.Call]:
    calls: list[ir.Call] = []
    for stmt in ir.walk_stmts(ctx.scan.program.body):
        for expr in ir.stmt_exprs(stmt):
            for node in ir.walk_expr(expr):
                if isinstance(node, ir.Call) and node.local and node.func == name:
                    calls.append(node)
    return calls


def _never_empty(node: ir.Expr, ctx: Context) -> bool:
    """A literal ``[]`` (it becomes ``None``) or a value known to be non-empty."""
    if isinstance(node, ir.Matrix) and not node.rows:
        return True
    info = ctx.info(node)
    if info.is_scalar:
        return True
    if info.kind is Kind.STRING:
        return bool(info.cols)
    return bool(info.numel)


def extract_defaults(fdef: ir.FunctionDef, ctx: Context) -> tuple[dict[str, str], list[ir.Stmt]]:
    """Header defaults (parameter -> Python text) and the body without the blocks.

    A ``None`` default's assignment stays in the body as an ``if p is None``
    block; see :func:`prologue`.
    """
    blocks, rest = _blocks(fdef)
    if any(_uses_nargin(stmt) for stmt in rest):
        ctx.decline("nargin used other than to fill in a missing argument")
    seen: set[str] = set()
    defaults: dict[str, str] = {}
    for param, empty, body in blocks:
        if param in seen:
            ctx.decline(f"two default blocks for {param}")
        seen.add(param)
        code = [s for s in body if not isinstance(s, ir.Comment)]
        last = code[-1]
        assert isinstance(last, ir.Assign)
        literal = _literal(last) if len(code) == 1 and not empty else None
        if literal is not None:
            defaults[param] = ast.unparse(literal)
            continue
        defaults[param] = "None"
        if empty:
            position = fdef.params.index(param)
            for call in _call_sites(fdef.name, ctx):
                if position < len(call.args) and not _never_empty(call.args[position], ctx):
                    ctx.decline(
                        f"{fdef.name} called with a {param} that may be empty "
                        "(isempty would replace it; Python's `is None` would not)"
                    )
    # Parameters after one with a default need defaults too in Python.
    params = fdef.params
    first = min((params.index(p) for p in defaults), default=len(params))
    if any(p not in defaults for p in params[first:]):
        ctx.decline("a parameter without a default after one with a default")
    return defaults, rest


def prologue(fdef: ir.FunctionDef, ctx: Context) -> list[Line]:
    """``if p is None: p = <default>`` for every ``None`` default, in source order."""
    blocks, _rest = _blocks(fdef)
    defaults, _body = extract_defaults(fdef, ctx)
    lines: list[Line] = []
    for param, _empty, body in blocks:
        if defaults.get(param) != "None":
            continue
        lines.append(ctx.line(f"if {param} is None:"))
        lines.extend(ctx.block(body))
    return lines


def defaults_of(fdef: ir.FunctionDef, ctx: Context) -> dict[str, str]:
    """The defaults :func:`extract_defaults` gives, for translating calls."""
    defaults, _body = extract_defaults(fdef, ctx)
    return defaults


# --- Python -> .m ------------------------------------------------------------
#
# def f(x, factor=10) becomes `function y = f(x, factor)` with
# `if nargin < 2, factor = 10; end` first in the body. A None default is []
# on the .m side: `if nargin < 2, lo = []; end`, `lo is None` is isempty(lo)
# and a None passed to a function of the file is []. That is only the same
# when no caller in the file passes a value that may be empty (an empty array
# is not None in Python but isempty on the .m side), which is checked.


def lift_defaults(fdef: ast.FunctionDef, ctx: LiftContext) -> list[ir.Stmt]:
    """The ``if nargin < k`` blocks that give ``fdef``'s parameters their defaults."""
    args = fdef.args.args
    defaults = fdef.args.defaults
    first = len(args) - len(defaults)
    blocks: list[ir.Stmt] = []
    for k, (arg, default) in enumerate(zip(args[first:], defaults, strict=True), start=first + 1):
        loc = ctx.loc(default)
        value: ir.Expr
        if isinstance(default, ast.Constant) and default.value is None:
            _check_callers(fdef, k - 1, ctx)
            value = ir.Matrix(loc=loc, rows=[])
        elif (
            isinstance(default, ast.Constant)
            and isinstance(default.value, int | float | str)
            or (
                isinstance(default, ast.UnaryOp)
                and isinstance(default.op, ast.USub)
                and isinstance(default.operand, ast.Constant)
                and isinstance(default.operand.value, int | float)
            )
        ):
            value = ctx.expr(default)
        else:
            ctx.decline(f"the default of {arg.arg} (only numbers, strings and None)")
        test = ctx.binop("lt", ctx.call("nargin", loc=loc), ctx.num(k), loc=loc)
        assign = ir.Assign(loc=loc, targets=[ir.Name(loc=loc, id=arg.arg)], value=value)
        blocks.append(ir.If(loc=loc, branches=[(test, [assign])]))
    return blocks


def _check_callers(fdef: ast.FunctionDef, position: int, ctx: LiftContext) -> None:
    """Every call in the file passes None or a value known not to be empty."""
    name = fdef.args.args[position].arg
    for node in ast.walk(ctx.module):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == fdef.name
        ):
            continue
        if node.keywords:
            ctx.decline(f"{fdef.name} called with keyword arguments")
        if position >= len(node.args):
            continue
        arg = node.args[position]
        if isinstance(arg, ast.Constant) and arg.value is None:
            continue
        info = ctx.info(arg)
        known = (
            info.is_scalar
            or (info.kind is PyKind.STRING and bool(info.value))
            or (info.is_vector and bool(info.length))
        )
        if not known:
            ctx.decline(
                f"{fdef.name} called with a {name} that may be empty (it would read as "
                "the default on the .m side, where None is [])"
            )


def _none_param(node: ast.expr, ctx: LiftContext) -> bool:
    """A parameter of the enclosing ``def`` whose default is ``None``."""
    fdef = ctx.function
    if fdef is None or not isinstance(node, ast.Name):
        return False
    args = fdef.args.args
    first = len(args) - len(fdef.args.defaults)
    for arg, default in zip(args[first:], fdef.args.defaults, strict=True):
        if arg.arg == node.id:
            return isinstance(default, ast.Constant) and default.value is None
    return False


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``p is None`` / ``p is not None`` of a None-default parameter; a None argument."""
    loc = ctx.loc(node)
    if isinstance(node, ast.Compare):
        right = node.comparators[0]
        if not (
            len(node.ops) == 1
            and isinstance(right, ast.Constant)
            and right.value is None
            and _none_param(node.left, ctx)
        ):
            ctx.decline("`is` other than a None-default parameter compared with None")
        assert isinstance(node.left, ast.Name)
        if ctx.function is not None and node.left.id in ctx.scope.mutated:
            ctx.decline(f"{node.left.id} is written in place")
        empty = ctx.call("isempty", ir.Name(loc=loc, id=node.left.id), loc=loc)
        return (
            empty if isinstance(node.ops[0], ast.Is) else ir.UnOp(loc=loc, op="not", operand=empty)
        )
    # None itself: only as an argument to a function of the file, where it
    # stands for "use the default" and is [] on the .m side.
    current = ctx.current
    for inner in ast.walk(current) if current is not None else []:
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id in ctx.functions
            and any(arg is node for arg in inner.args)
        ):
            position = next(k for k, arg in enumerate(inner.args) if arg is node)
            fdef = ctx.functions[inner.func.id]
            args = fdef.args.args
            first = len(args) - len(fdef.args.defaults)
            if position >= first:
                default = fdef.args.defaults[position - first]
                if isinstance(default, ast.Constant) and default.value is None:
                    return ir.Matrix(loc=loc, rows=[])
    ctx.decline("None other than as an argument whose default is None")
