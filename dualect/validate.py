# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Validating rule folders against the schema and the folder conventions.

The JSON Schema (``dualect/schema/rule.schema.json``) checks the shape of one
``rule.yaml``. The checks here add what a schema cannot see: the id matches the
folder, cases come in pairs, the status agrees with the files present, and
``related`` ids point at rules that exist.
"""

import functools
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from dualect.errors import RuleError
from dualect.paths import schema_path
from dualect.rulebook import (
    CASES_DIR,
    TRANSLATOR_FILE,
    Status,
    find_cases,
    read_rule_yaml,
    rule_dirs,
    rule_id_for,
)


@functools.cache
def _validator() -> Draft202012Validator:
    """Build the schema validator once; the schema file never changes at runtime."""
    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def schema_errors(*, data: dict[str, Any]) -> list[str]:
    """Return human-readable schema violations for one parsed ``rule.yaml``.

    :param data: the parsed mapping.
    :returns: one message per violation; empty when valid.
    """
    errors = sorted(_validator().iter_errors(data), key=lambda err: list(err.absolute_path))
    messages = []
    for err in errors:
        where = "/".join(str(part) for part in err.absolute_path) or "(top level)"
        messages.append(f"{where}: {err.message}")
    return messages


def validate_rule_dir(*, rule_dir: Path) -> list[str]:
    """Check one rule folder; return problems as ``"<rule id>: <message>"`` strings.

    :param rule_dir: folder containing ``rule.yaml``.
    """
    expected_id = rule_id_for(rule_dir=rule_dir)
    try:
        data = read_rule_yaml(path=rule_dir / "rule.yaml")
    except RuleError as exc:
        return [f"{expected_id}: {exc}"]

    problems = schema_errors(data=data)
    if data.get("id") != expected_id:
        problems.append(f"id {data.get('id')!r} does not match folder (expected {expected_id!r})")

    # Cases must come in pairs: a lone .m or .py is a half-written example, and
    # silently skipping it would hide the gap.
    cases_dir = rule_dir / CASES_DIR
    m_stems = {path.stem for path in cases_dir.glob("*.m")}
    py_stems = {path.stem for path in cases_dir.glob("*.py")}
    for stem in sorted(m_stems - py_stems):
        problems.append(f"cases/{stem}.m has no matching cases/{stem}.py")
    for stem in sorted(py_stems - m_stems):
        problems.append(f"cases/{stem}.py has no matching cases/{stem}.m")

    status = data.get("status")
    has_cases = bool(find_cases(rule_dir=rule_dir))
    if status in (Status.GUIDED, Status.MECHANICAL) and not has_cases:
        problems.append(f"status {status!r} needs at least one case pair in cases/")
    if status == Status.MECHANICAL and not (rule_dir / TRANSLATOR_FILE).is_file():
        problems.append(f"status 'mechanical' needs a {TRANSLATOR_FILE}")

    return [f"{expected_id}: {problem}" for problem in problems]


def validate_all(*, rules_dir: Path) -> list[str]:
    """Validate every rule under ``rules_dir``, including cross-rule references.

    :param rules_dir: root of the knowledge base.
    :returns: every problem found; empty when the whole knowledge base is valid.
    """
    dirs = rule_dirs(rules_dir=rules_dir)
    problems: list[str] = []
    known_ids = {rule_id_for(rule_dir=path) for path in dirs}
    for rule_dir in dirs:
        problems.extend(validate_rule_dir(rule_dir=rule_dir))
        rule_id = rule_id_for(rule_dir=rule_dir)
        try:
            related = read_rule_yaml(path=rule_dir / "rule.yaml").get("related") or []
        except RuleError:
            continue  # already reported by validate_rule_dir
        for other in related:
            if other not in known_ids:
                problems.append(f"{rule_id}: related rule {other!r} does not exist")
            elif other == rule_id:
                problems.append(f"{rule_id}: lists itself in related")
    return problems
