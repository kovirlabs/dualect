# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Loading the rule knowledge base from ``rules/<category>/<rule-id>/``.

This module only *reads* rules; checking them against the schema and the
folder conventions is :mod:`dualect.validate`. Later milestones (scanner,
context command, translators) build on :func:`load_rules`.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from dualect.errors import RuleError

RULE_FILE = "rule.yaml"
TRANSLATOR_FILE = "translate.py"
CASES_DIR = "cases"


class Status(StrEnum):
    """Lifecycle stage of a rule (docs/design.md section 4.3)."""

    UNSUPPORTED = "unsupported"
    GUIDED = "guided"
    MECHANICAL = "mechanical"


@dataclass(frozen=True)
class CasePair:
    """One paired example: a ``.m`` script and its hand-written ``.py`` twin.

    :param name: file stem shared by both files, e.g. ``"basic"``.
    :param m_path: the ``.m`` file.
    :param py_path: the ``.py`` file.
    """

    name: str
    m_path: Path
    py_path: Path


@dataclass(frozen=True)
class Rule:
    """A loaded rule folder.

    :param id: ``<category>/<rule-id>``.
    :param path: the rule folder.
    :param data: the parsed ``rule.yaml`` mapping, as written.
    :param cases: complete ``.m``/``.py`` pairs, sorted by name.
    """

    id: str
    path: Path
    data: dict[str, Any]
    cases: tuple[CasePair, ...] = field(default=())

    @property
    def category(self) -> str:
        """The category part of the id, e.g. ``"indexing"``."""
        return self.id.split("/", 1)[0]

    @property
    def title(self) -> str:
        """Short human-readable title."""
        return str(self.data.get("title", ""))

    @property
    def status(self) -> Status:
        """Lifecycle stage; raises :class:`RuleError` for an unknown value."""
        try:
            return Status(str(self.data.get("status")))
        except ValueError as exc:
            raise RuleError(f"{self.id}: unknown status {self.data.get('status')!r}") from exc

    @property
    def match(self) -> tuple[str, ...]:
        """Construct tags this rule covers (see :mod:`dualect.constructs`)."""
        return tuple(str(tag) for tag in self.data.get("match") or ())

    @property
    def has_translator(self) -> bool:
        """Whether the folder contains a deterministic ``translate.py``."""
        return (self.path / TRANSLATOR_FILE).is_file()

    @property
    def tolerance(self) -> dict[str, float]:
        """Per-rule verify tolerance override (may be empty)."""
        raw = self.data.get("tolerance") or {}
        return {key: float(value) for key, value in raw.items()}


def rule_dirs(*, rules_dir: Path) -> list[Path]:
    """Return every ``<category>/<rule-id>/`` folder that holds a ``rule.yaml``.

    :param rules_dir: root of the knowledge base.
    :returns: folders sorted by rule id, so listings and test ids are stable.
    """
    return sorted(path.parent for path in rules_dir.glob(f"*/*/{RULE_FILE}"))


def find_cases(*, rule_dir: Path) -> tuple[CasePair, ...]:
    """Return the complete ``.m``/``.py`` pairs in a rule's ``cases/`` folder.

    Unpaired files are not returned here; :mod:`dualect.validate` reports them,
    because a lone file is a mistake to fix rather than something to skip.

    :param rule_dir: the rule folder.
    """
    cases_dir = rule_dir / CASES_DIR
    pairs = []
    for m_path in sorted(cases_dir.glob("*.m")):
        py_path = m_path.with_suffix(".py")
        if py_path.is_file():
            pairs.append(CasePair(name=m_path.stem, m_path=m_path, py_path=py_path))
    return tuple(pairs)


def read_rule_yaml(*, path: Path) -> dict[str, Any]:
    """Parse a ``rule.yaml`` file into a mapping.

    :param path: the file.
    :raises RuleError: if the YAML is invalid or not a mapping.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuleError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise RuleError(f"{path}: expected a mapping at the top level")
    return data


def rule_id_for(*, rule_dir: Path) -> str:
    """Return the id a rule folder must declare: ``<category>/<folder name>``."""
    return f"{rule_dir.parent.name}/{rule_dir.name}"


def load_rule(*, rule_dir: Path) -> Rule:
    """Load one rule folder (without schema validation).

    :param rule_dir: folder containing ``rule.yaml``.
    """
    data = read_rule_yaml(path=rule_dir / RULE_FILE)
    return Rule(
        id=rule_id_for(rule_dir=rule_dir),
        path=rule_dir,
        data=data,
        cases=find_cases(rule_dir=rule_dir),
    )


def load_rules(*, rules_dir: Path, status: Status | None = None) -> list[Rule]:
    """Load every rule under ``rules_dir``.

    :param rules_dir: root of the knowledge base.
    :param status: if given, keep only rules at this lifecycle stage.
    :returns: rules sorted by id.
    """
    rules = [load_rule(rule_dir=path) for path in rule_dirs(rules_dir=rules_dir)]
    if status is not None:
        rules = [rule for rule in rules if rule.status is status]
    return rules
