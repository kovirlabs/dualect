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
