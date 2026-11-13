# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for syntax/comments: ``%`` / ``#`` comments and ``%{ ... %}`` blocks.

A comment becomes ``# text``; a block becomes one ``#`` line per line (never a
string literal, which would be an expression statement); a ``%%`` cell divider
stays a cell divider, ``# %%``. Comments on a statement's line are attached by
the emitter.
"""

from dualect import ir
from dualect.emit import Context, Line


def translate(stmt: ir.Comment, ctx: Context) -> list[Line]:
    """One comment statement."""
    lines = stmt.text.split("\n") if stmt.block else [stmt.text]
    return [ctx.line(f"# {line}".rstrip()) for line in lines]
