# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""The promotion check (docs/design.md section 4.3): does a rule earn ``mechanical``?

For each rule and each direction it declares, every case is run through the
deterministic translator and checked twice:

* **layer 2** (section 7): the translation is the same code as the case's
  hand-written target. To Python, both are compared after formatting with
  comments ignored (:func:`dualect.pyformat.same_code`). From Python, the
  generated ``.m`` and the hand-written ``.m`` are both lowered into the IR and
  compared structurally; the hand-written ``.m`` files are idiomatic sources
  written for the other direction (positions 1-based, ``x(end)`` forms), so
  this match is reported but not required from Python.
* **execution**: the translation runs and matches the source by value.

A direction *qualifies* for ``mechanical`` when the rule has its translator,
every case translates with no hole, every required layer-2 match holds and,
when verification runs, every case verifies. The check fails for a rule
marked ``mechanical`` that does not qualify, and for a ``guided`` rule that
does (it should be promoted); a rule with no case cannot qualify.
"""

import tempfile
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from dualect import ir
from dualect.dialect import Dialect
from dualect.errors import DualectError
from dualect.lower import lower_source
from dualect.pipeline import Translation, translate_file, translate_python_text, translate_text
from dualect.pyformat import same_code
from dualect.rulebook import DIRECTIONS, TO_PYTHON, CasePair, Rule, Status
from dualect.runners import MRunner
from dualect.translators import load_translator


@dataclass
class CaseCheck:
    """One case in one direction.

    :param name: the case name.
    :param complete: translated with no hole.
    :param layer2: matches the hand-written target (``None`` when not checked).
    :param verified: verified by execution (``None`` when not run).
    :param note: why it failed, briefly.
    """

    name: str
    complete: bool
    layer2: bool | None
    verified: bool | None
    note: str = ""


@dataclass
class RuleCheck:
    """One rule in one direction.

    :param rule_id: the rule.
    :param direction: ``to-python`` or ``from-python``.
    :param declared: the status the rule declares for this direction.
    :param cases: per-case results.
    :param qualifies: whether the evidence supports ``mechanical``.
    :param reasons: what keeps it from qualifying.
    """

    rule_id: str
    direction: str
    declared: Status
    cases: list[CaseCheck] = field(default_factory=list)
    qualifies: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def problem(self) -> str | None:
        """Why the declared status disagrees with the evidence, or ``None``."""
        if self.declared is Status.MECHANICAL and not self.qualifies:
            return "marked mechanical but does not qualify: " + "; ".join(self.reasons)
        if self.declared is Status.GUIDED and self.qualifies:
            return "qualifies for mechanical: promote it"
        return None

    def line(self) -> str:
        """One summary line."""
        verdict = "qualifies" if self.qualifies else "does not qualify"
        problem = self.problem
        mark = "FAIL" if problem else "ok  "
        if self.direction != TO_PYTHON and self.cases:
            # Informational from Python (see the module docstring).
            matched = sum(1 for case in self.cases if case.layer2)
            verdict += f", layer 2 {matched}/{len(self.cases)}"
        detail = (
            f"  ({problem})"
            if problem
            else f"  ({'; '.join(self.reasons)})"
            if self.reasons
            else ""
        )
        return (
            f"{mark} {self.rule_id:<32} {self.direction:<11} {self.declared.value:<11} "
            f"{verdict}{detail}"
        )


def _has_translator(rule: Rule, direction: str) -> bool:
    module = load_translator(rule)
    if module is None:
        return False
    name = "translate" if direction == TO_PYTHON else "lift"
    return callable(getattr(module, name, None))


def _fingerprint(node: Any) -> Any:
    """A structural key of IR nodes: kinds and values, without locations or comments."""
    if isinstance(node, list):
        return [_fingerprint(item) for item in node if not isinstance(item, ir.Comment)]
    if isinstance(node, tuple):
        return tuple(_fingerprint(item) for item in node)
    if is_dataclass(node) and not isinstance(node, type):
        skip = {"loc", "comment", "syntax", "source_name", "text"}
        if isinstance(node, ir.End):
            skip |= {"target"}  # a back-reference, not a child
        values = [
            (f.name, _fingerprint(getattr(node, f.name)))
            for f in fields(node)
            if f.name not in skip
        ]
        if isinstance(node, ir.Num):
            values.append(("value", complex(node.value)))
        return (type(node).__name__, tuple(values))
    return node


def same_m_code(a: str, b: str) -> bool:
    """Whether two ``.m`` sources lower to the same IR (layout and comments ignored)."""
    try:
        return _fingerprint(lower_source(text=a).body) == _fingerprint(lower_source(text=b).body)
    except (DualectError, ValueError):
        return False


def _translate(case: CasePair, direction: str, rules: list[Rule]) -> Translation:
    if direction == TO_PYTHON:
        text = case.m_path.read_text(encoding="utf-8")
        return translate_text(text=text, name=str(case.m_path), rules=rules, dialect=Dialect())
    text = case.py_path.read_text(encoding="utf-8")
    return translate_python_text(text=text, name=str(case.py_path), rules=rules)


def check_case(
    *, case: CasePair, direction: str, rules: list[Rule], runner: MRunner | None
) -> CaseCheck:
    """Translate one case in one direction; check layer 2 and (with a runner) execution.

    :param case: the case pair.
    :param direction: ``to-python`` or ``from-python``.
    :param rules: all loaded rules.
    :param runner: an Octave session for verification, or ``None`` to skip it.
    """
    try:
        result = _translate(case, direction, rules)
    except DualectError as exc:
        return CaseCheck(name=case.name, complete=False, layer2=False, verified=None, note=str(exc))
    target = case.py_path if direction == TO_PYTHON else case.m_path
    expected = target.read_text(encoding="utf-8")
    if direction == TO_PYTHON:
        layer2 = same_code(result.code, expected)
    else:
        layer2 = same_m_code(result.code, expected)
    check = CaseCheck(name=case.name, complete=result.complete, layer2=layer2, verified=None)
    if not result.complete:
        check.note = "; ".join(f"line {h.loc.line}: {h.reason}" for h in result.holes)
        return check
    if runner is None:
        return check
    source = case.m_path if direction == TO_PYTHON else case.py_path
    suffix = ".py" if direction == TO_PYTHON else ".m"
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / f"{case.name}{suffix}"
        verified = translate_file(
            source=source, output=output, rules=rules, dialect=Dialect(), runner=runner
        )
    check.verified = verified.verified
    if not verified.verified:
        check.note = verified.verify_note
    return check


def check_rule(
    *, rule: Rule, direction: str, rules: list[Rule], runner: MRunner | None
) -> RuleCheck | None:
    """Check one rule in one direction (``None`` when it declares nothing for it).

    :param rule: the rule.
    :param direction: ``to-python`` or ``from-python``.
    :param rules: all loaded rules.
    :param runner: an Octave session for verification, or ``None`` to skip it.
    """
    declared = rule.status_for(direction)
    if declared is None:
        return None
    result = RuleCheck(rule_id=rule.id, direction=direction, declared=declared)
    if not rule.cases:
        result.reasons.append("no cases")
    if not _has_translator(rule, direction):
        name = "translate()" if direction == TO_PYTHON else "lift()"
        result.reasons.append(f"no {name} in translate.py")
    else:
        result.cases = [
            check_case(case=case, direction=direction, rules=rules, runner=runner)
            for case in rule.cases
        ]
    for case in result.cases:
        if not case.complete:
            result.reasons.append(f"{case.name}: holes ({case.note})")
        elif direction == TO_PYTHON and not case.layer2:
            result.reasons.append(f"{case.name}: differs from the hand-written .py")
        elif case.verified is False:
            result.reasons.append(f"{case.name}: does not verify ({case.note})")
    result.qualifies = not result.reasons
    if result.qualifies and runner is None and declared is not Status.MECHANICAL:
        # Without execution there is no evidence enough to ask for a promotion.
        result.qualifies = False
        result.reasons.append("not verified by execution")
    return result


def check_rules(
    *,
    rules: list[Rule],
    selected: list[str] | None = None,
    directions: tuple[str, ...] = DIRECTIONS,
    verify: bool = True,
) -> list[RuleCheck]:
    """Check rules in the given directions.

    :param rules: all loaded rules (translations need every rule).
    :param selected: rule ids to check (default: all).
    :param directions: which directions.
    :param verify: run the translations (needs GNU Octave).
    :raises DualectError: for an unknown rule id.
    """
    known = {rule.id for rule in rules}
    unknown = [rule_id for rule_id in selected or [] if rule_id not in known]
    if unknown:
        raise DualectError(f"no such rule: {', '.join(unknown)}")
    chosen = [rule for rule in rules if not selected or rule.id in selected]
    results: list[RuleCheck] = []
    runner = MRunner() if verify else None
    try:
        for rule in chosen:
            for direction in directions:
                checked = check_rule(rule=rule, direction=direction, rules=rules, runner=runner)
                if checked is not None:
                    results.append(checked)
    finally:
        if runner is not None:
            runner.close()
    return results


def report(results: list[RuleCheck], *, verified: bool) -> str:
    """The check as text: one line per rule and direction, then a summary."""
    lines = [check.line() for check in results]
    problems = [check for check in results if check.problem]
    mode = "" if verified else " (not verified by execution: --no-verify)"
    lines.append(f"{len(results)} checked, {len(problems)} with a status problem{mode}")
    return "\n".join(lines)
