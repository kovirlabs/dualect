# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for syntax/command-syntax: ``name word1 word2``.

Only commands whose effect is fully known are translated. ``clear a b``
deletes those variables (``del a, b``). Commands that only change how the
interpreter displays or plots (``format long``, ``more off``, ``clc``,
``close all``, ``hold on``, ``figure``) have no effect on any value, and the
translation never reproduces the interpreter's display format anyway (that is
io/display, guided), so they become a comment that says what was dropped.

Declined: ``clear`` without names (the whole workspace, or functions too),
``clear`` with options or patterns, ``pkg load`` (it changes which functions
exist) and every other command.
"""

import re

from dualect import ir
from dualect.emit import Context, Line

#: Commands with no effect on values: display, paging, figures.
_DISPLAY_ONLY = {
    "format": {"long", "short", "compact", "loose", "longg", "shortg", "longG", "shortG"},
    "more": {"on", "off"},
    "clc": set(),
    "close": {"all"},
    "hold": {"on", "off"},
    "figure": set(),
}
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def translate(node: ir.Node, ctx: Context) -> list[Line]:
    """One command."""
    if not isinstance(node, ir.Command):
        ctx.decline("command syntax in this position")
    name, words = node.name, node.words
    if name == "clear":
        if not words or not all(_NAME.match(word) for word in words):
            ctx.decline("clear without variable names, or with options or patterns")
        if any(word in ("all", "functions", "variables", "global", "classes") for word in words):
            ctx.decline(f"clear {' '.join(words)} (a keyword, not a variable name)")
        env = ctx.analysis.env_before.get(id(node), {})
        missing = [w for w in words if w not in env or env[w].maybe_undefined]
        if missing:
            ctx.decline(f"clear of {', '.join(missing)}, which may not be variables")
        return [ctx.line(f"del {', '.join(words)}")]
    allowed = _DISPLAY_ONLY.get(name)
    if allowed is None:
        ctx.decline(f"the command {name}")
    if (
        len(words) > 1
        or (words and words[0] not in allowed)
        or (not words and allowed and name != "format")
    ):
        ctx.decline(f"{name} {' '.join(words)}")
    text = " ".join([name, *words])
    return [ctx.line(f"# {text}: display only, no effect on values (dropped by dualect)")]
