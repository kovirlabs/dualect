# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for syntax/comments: ``%`` / ``#`` comments and ``%{ ... %}`` blocks.

A comment becomes ``# text``; a block becomes one ``#`` line per line (never a
string literal, which would be an expression statement); a ``%%`` cell divider
stays a cell divider, ``# %%``. Comments on a statement's line are attached by
the emitter.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.lift import LiftContext


def translate(stmt: ir.Comment, ctx: Context) -> list[Line]:
    """One comment statement."""
    lines = stmt.text.split("\n") if stmt.block else [stmt.text]
    return [ctx.line(f"# {line}".rstrip()) for line in lines]


# --- Python -> .m ------------------------------------------------------------
#
# `# text` is `% text`, on its own line or after a statement; a docstring
# becomes comment lines. The engine places comments (dualect.lift); `dualect:`
# verify directives are never copied, since they describe the source pairing.


def lift(node: ast.AST, ctx: LiftContext) -> ir.Expr:
    """Never called: comments are placed by the engine."""
    ctx.decline("a comment is not an expression")
