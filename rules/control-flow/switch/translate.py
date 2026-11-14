# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for control-flow/switch: ``match`` with literal patterns.

Each ``case`` value must be a literal (a number or a string); a cell of
values becomes alternatives joined by ``|``; ``otherwise`` is ``case _``.
Non-literal case values are declined: a bare name in a ``match`` pattern
captures instead of comparing.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.lift import LiftContext


def _pattern(value: ir.Expr, ctx: Context) -> str:
    if isinstance(value, ir.Num | ir.Str):
        return ast.unparse(ctx.expr(value))
    if isinstance(value, ir.UnOp) and value.op == "uminus" and isinstance(value.operand, ir.Num):
        return ast.unparse(ctx.expr(value))
    ctx.decline("a case value that is not a literal")


def translate(stmt: ir.Switch, ctx: Context) -> list[Line]:
    """One ``switch``."""
    if ctx.info(stmt.subject).is_array:
        ctx.decline("switching on an array")
    lines = [ctx.line(f"match {ast.unparse(ctx.expr(stmt.subject))}:")]
    ctx.indent += 1
    try:
        for values, body in stmt.cases:
            patterns = " | ".join(_pattern(value, ctx) for value in values)
            lines.append(ctx.line(f"case {patterns}:"))
            lines.extend(ctx.block(body))
        if stmt.otherwise is not None:
            lines.append(ctx.line("case _:"))
            lines.extend(ctx.block(stmt.otherwise))
    finally:
        ctx.indent -= 1
    return lines


# --- Python -> .m ------------------------------------------------------------
#
# match with literal patterns is switch: case 1 | 2 is case {1, 2}, a string
# pattern compares the whole text (as switch does), case _ is otherwise and
# must come last. Other patterns (captures, guards, classes) are declined.


def _values(pattern: ast.pattern, ctx: LiftContext) -> list[ir.Expr] | None:
    """The literal values of a pattern; ``None`` for the wildcard ``_``."""
    if isinstance(pattern, ast.MatchAs) and pattern.pattern is None and pattern.name is None:
        return None
    if isinstance(pattern, ast.MatchValue):
        value = pattern.value
        info = ctx.info(value)
        if not (info.is_scalar or info.kind.value == "string") or info.is_bool:
            ctx.decline("a case value that is not a number or a string")
        return [ctx.expr(value)]
    if isinstance(pattern, ast.MatchOr):
        found: list[ir.Expr] = []
        for inner in pattern.patterns:
            values = _values(inner, ctx)
            if values is None:
                ctx.decline("a wildcard inside an or-pattern")
            found.extend(values)
        return found
    ctx.decline("a case pattern that is not a literal")


def lift(stmt: ast.stmt, ctx: LiftContext) -> list[ir.Stmt]:
    """One ``match``."""
    assert isinstance(stmt, ast.Match)
    subject = ctx.info(stmt.subject)
    if not (subject.is_scalar or subject.kind.value == "string"):
        ctx.decline("matching a value that is not a number or a string")
    cases: list[tuple[list[ir.Expr], list[ir.Stmt]]] = []
    otherwise: list[ir.Stmt] | None = None
    for k, case in enumerate(stmt.cases):
        if case.guard is not None:
            ctx.decline("a case guard")
        values = _values(case.pattern, ctx)
        if values is None:
            if k != len(stmt.cases) - 1:
                ctx.decline("a wildcard case that is not last")
            otherwise = ctx.block(case.body)
        else:
            cases.append((values, ctx.block(case.body)))
    return [
        ir.Switch(
            loc=ctx.loc(stmt), subject=ctx.expr(stmt.subject), cases=cases, otherwise=otherwise
        )
    ]
