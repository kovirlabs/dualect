# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""The verify harness: run a ``.m`` script and a ``.py`` script, compare outputs.

Every variable the ``.m`` script leaves behind (except ``ans``) must exist on the
Python side with an equal value (see :mod:`dualect.compare`). Python-only names
are allowed, since an idiomatic translation may need extra temporaries, and are
listed in the result. A pair that compares *nothing* fails: a check that checked
nothing must not read as a pass.

Case files may carry directives in comments, on either side::

    % dualect: index idx k      -- 1-based on the .m side, 0-based in Python
    % dualect: ignore tmp       -- legitimately differs (e.g. a loop counter)
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from dualect.compare import Tolerance, compare_values
from dualect.errors import DirectiveError
from dualect.runners import MRunner, run_py_script

_DIRECTIVE = re.compile(r"^\s*[%#]+\s*dualect:\s*(?P<body>.*)$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KNOWN_DIRECTIVES = ("index", "ignore")


@dataclass(frozen=True)
class VerifyOptions:
    """What to compare and how strictly.

    :param tolerance: allclose tolerances.
    :param index_vars: names holding indices (1-based in ``.m``, 0-based in Python).
    :param ignore: names excluded from comparison.
    """

    tolerance: Tolerance = field(default_factory=Tolerance)
    index_vars: frozenset[str] = frozenset()
    ignore: frozenset[str] = frozenset()


@dataclass(frozen=True)
class VarResult:
    """Outcome for one compared variable.

    :param name: variable name.
    :param ok: whether the values matched.
    :param detail: reason for a mismatch, or a note; empty when plainly equal.
    """

    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of verifying one ``.m``/``.py`` pair.

    :param m_path: the ``.m`` script.
    :param py_path: the ``.py`` script.
    :param variables: per-variable results, sorted by name.
    :param python_only: names defined only by the Python script (informational).
    """

    m_path: Path
    py_path: Path
    variables: tuple[VarResult, ...]
    python_only: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when at least one variable was compared and all matched."""
        return bool(self.variables) and all(var.ok for var in self.variables)

    def report(self) -> str:
        """Return a human-readable, multi-line summary."""
        lines = [f"verify {self.m_path} <-> {self.py_path}"]
        if not self.variables:
            lines.append("  FAIL nothing compared: the .m script left no variables")
        for var in self.variables:
            mark = "ok  " if var.ok else "FAIL"
            suffix = f"  ({var.detail})" if var.detail else ""
            lines.append(f"  {mark} {var.name}{suffix}")
        if self.python_only:
            lines.append(f"  note: Python-only names not compared: {', '.join(self.python_only)}")
        lines.append("PASS" if self.ok else "FAIL")
        return "\n".join(lines)


def parse_directives(*, text: str, path: Path) -> tuple[frozenset[str], frozenset[str]]:
    """Read ``dualect:`` comment directives from a script.

    :param text: the script source.
    :param path: used in error messages.
    :returns: ``(index_vars, ignore)``.
    :raises DirectiveError: for an unknown directive or a malformed name, so a
        typo cannot silently weaken a comparison.
    """
    found: dict[str, set[str]] = {key: set() for key in _KNOWN_DIRECTIVES}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _DIRECTIVE.match(line)
        if match is None:
            continue
        words = match.group("body").replace(",", " ").split()
        if not words or words[0] not in _KNOWN_DIRECTIVES:
            raise DirectiveError(
                f"{path}:{line_number}: unknown directive {line.strip()!r}; "
                f"expected one of {', '.join(_KNOWN_DIRECTIVES)}"
            )
        names = words[1:]
        if not names or not all(_IDENTIFIER.match(name) for name in names):
            raise DirectiveError(f"{path}:{line_number}: expected variable names after {words[0]}")
        found[words[0]].update(names)
    return frozenset(found["index"]), frozenset(found["ignore"])


def options_for_pair(
    *, m_path: Path, py_path: Path, tolerance: Tolerance, extra_index: Iterable[str] = ()
) -> VerifyOptions:
    """Build :class:`VerifyOptions` from the directives in both files.

    :param m_path: the ``.m`` script.
    :param py_path: the ``.py`` script.
    :param tolerance: tolerances to use (global default or a rule's override).
    :param extra_index: additional index names, e.g. from the command line.
    """
    index_vars: set[str] = set(extra_index)
    ignore: set[str] = set()
    for path in (m_path, py_path):
        found_index, found_ignore = parse_directives(
            text=path.read_text(encoding="utf-8"), path=path
        )
        index_vars |= found_index
        ignore |= found_ignore
    return VerifyOptions(
        tolerance=tolerance, index_vars=frozenset(index_vars), ignore=frozenset(ignore)
    )


def verify_pair(
    *,
    m_path: Path,
    py_path: Path,
    options: VerifyOptions | None = None,
    runner: MRunner | None = None,
) -> VerifyResult:
    """Run both scripts and compare every ``.m`` output with the Python value.

    :param m_path: the ``.m`` script.
    :param py_path: the ``.py`` script.
    :param options: comparison settings; by default, read from the files'
        directives with the global default tolerance.
    :param runner: an existing :class:`MRunner` to reuse (saves Octave start-up);
        one is created and closed here if not given.
    :raises ExecutionError: if either script fails to run.
    :raises RuntimeMissingError: if Octave is not installed.
    """
    if options is None:
        options = options_for_pair(m_path=m_path, py_path=py_path, tolerance=Tolerance())

    if runner is None:
        with MRunner() as own_runner:
            m_outputs = own_runner.run(path=m_path)
    else:
        m_outputs = runner.run(path=m_path)
    py_outputs = run_py_script(path=py_path)

    results = []
    for name in sorted(set(m_outputs) - options.ignore):
        if name not in py_outputs:
            results.append(VarResult(name=name, ok=False, detail="missing on the Python side"))
            continue
        reason = compare_values(
            m_value=m_outputs[name],
            py_value=py_outputs[name],
            tolerance=options.tolerance,
            index=name in options.index_vars,
        )
        note = "index, compared as 0-based" if name in options.index_vars else ""
        results.append(VarResult(name=name, ok=reason is None, detail=reason or note))

    python_only = tuple(sorted(set(py_outputs) - set(m_outputs) - options.ignore))
    return VerifyResult(
        m_path=m_path, py_path=py_path, variables=tuple(results), python_only=python_only
    )
