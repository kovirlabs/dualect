# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for syntax/line-continuation: ``...``.

There is nothing left to translate by the time a statement reaches the
emitter: the parser joins continued lines into one statement, and the text
after ``...`` becomes the statement's trailing comment. ruff then wraps long
Python lines inside parentheses. This module exists so the rule can be
``mechanical``; its construct never reaches :func:`translate`.
"""

from dualect import ir
from dualect.emit import Context


def translate(node: ir.Node, ctx: Context) -> None:
    """Never called: continuations do not survive parsing."""
    ctx.decline("a line continuation is not a node")
