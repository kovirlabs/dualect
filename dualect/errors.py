# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Named exceptions raised by dualect.

Every failure dualect knows how to describe gets its own class, so callers (the
CLI, tests, a future MCP server) can react to the *kind* of problem rather than
parsing messages. This is also how "never translate silently wrong" is enforced:
an unknown construct is an exception with a rule id, not a best guess.
"""

from pathlib import Path


class DualectError(Exception):
    """Base class for every error dualect raises on purpose."""


class RuleError(DualectError):
    """A rule folder or ``rule.yaml`` is missing, malformed or inconsistent."""


class RuntimeMissingError(DualectError):
    """The program needed to execute ``.m`` code (GNU Octave) is not available."""


class ExecutionError(DualectError):
    """Running one side of a translation pair failed.

    :param side: ``"m"`` or ``"py"``, which side failed.
    :param path: the script that was being executed.
    :param message: the interpreter's error text.
    """

    def __init__(self, *, side: str, path: Path, message: str) -> None:
        self.side = side
        self.path = path
        self.message = message
        super().__init__(f"{side} side failed ({path}): {message}")


class DirectiveError(DualectError):
    """A ``dualect:`` comment directive in a case file could not be understood."""


class UnsupportedConstructError(DualectError):
    """A source construct has no supported translation.

    Raised by the translator (milestone 4 onward) instead of emitting a guess.

    :param rule_id: the ``unsupported`` rule that documents the construct, or
        ``None`` when no rule names it yet.
    :param detail: where and what the construct is, for the report.
    """

    def __init__(self, *, rule_id: str | None, detail: str) -> None:
        self.rule_id = rule_id
        self.detail = detail
        label = rule_id if rule_id is not None else "no rule yet"
        super().__init__(f"unsupported construct ({label}): {detail}")
