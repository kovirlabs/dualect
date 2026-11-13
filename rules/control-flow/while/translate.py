# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for control-flow/while: ``while``, ``do ... until``, ``break``, ``continue``.

Conditions follow control-flow/if-elseif. ``do ... until c`` becomes
``while True:`` with ``if c: break`` at the end of the body. A ``continue``
inside ``do ... until`` would skip that test in Python, so it is declined.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line


def translate(stmt: ir.While | ir.DoUntil | ir.Break | ir.Continue, ctx: Context) -> list[Line]:
    """One loop or loop jump."""
    condition = ctx.module("control-flow/if-elseif").condition
    if isinstance(stmt, ir.Break):
        return [ctx.line("break")]
    if isinstance(stmt, ir.Continue):
        if ctx.loop_stack and isinstance(ctx.loop_stack[-1], ir.DoUntil):
            ctx.decline("continue inside do ... until")
        return [ctx.line("continue")]
    if isinstance(stmt, ir.While):
        header = ctx.line(f"while {ast.unparse(condition(stmt.cond, ctx))}:")
        with ctx.loop(stmt):
            body = ctx.block(stmt.body)
        return [header, *body]
    with ctx.loop(stmt):
        body = ctx.block(stmt.body)
    exit_test = ast.unparse(condition(stmt.cond, ctx))
    body = [line for line in body if line.text != "pass"]
    return [
        ctx.line("while True:"),
        *body,
        ctx.line(f"if {exit_test}:", extra=1),
        ctx.line("break", extra=2),
    ]
