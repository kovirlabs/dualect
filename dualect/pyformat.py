# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Formatting emitted Python with ruff, and comparing code structurally.

The comparison is testing layer 2 of docs/design.md section 7: a translator's
output must match the hand-written ``.py`` of each case *after formatting*.
Both sides are parsed and printed back with :func:`ast.unparse`, which is
formatting-independent and drops comments: comments in case files are
teaching notes, not code, so they are not part of the match.
"""

import ast
import subprocess

from dualect.errors import DualectError


class FormatError(DualectError):
    """ruff could not format the emitted code (a bug in a translator)."""


def _ruff() -> str:
    from ruff.__main__ import find_ruff_bin  # deferred: only needed when formatting

    return str(find_ruff_bin())


def format_python(text: str) -> str:
    """Format Python source with ruff's defaults (no project configuration).

    :param text: the source.
    :raises FormatError: if ruff rejects it (the text is not valid Python).
    """
    result = subprocess.run(
        [_ruff(), "format", "--isolated", "--stdin-filename", "dualect_output.py", "-"],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise FormatError(f"ruff format failed: {result.stderr.strip()}")
    return result.stdout


def normalized_code(text: str) -> str:
    """The code of a Python file, with formatting and comments normalized away."""
    return ast.unparse(ast.parse(text))


def same_code(a: str, b: str) -> bool:
    """Whether two Python sources are the same code, ignoring layout and comments."""
    return normalized_code(a) == normalized_code(b)
