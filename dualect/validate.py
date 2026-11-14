# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Validating rule folders against the schema and the folder conventions.

The JSON Schema (``dualect/schema/rule.schema.json``) checks the shape of one
``rule.yaml``. The checks here add what a schema cannot see: the id matches the
folder, cases come in pairs, the status agrees with the files present, and
``related`` ids point at rules that exist.
"""

import ast
import functools
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from dualect.constructs import is_known_tag
from dualect.errors import RuleError
from dualect.paths import schema_path
from dualect.pyconstructs import is_known_py_tag
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

    for tag in data.get("match") or []:
        if isinstance(tag, str) and not is_known_tag(tag):
            problems.append(f"match: unknown construct tag {tag!r} (see dualect/constructs.py)")

    from_python = data.get("from_python")
    if isinstance(from_python, dict):
        for tag in from_python.get("match") or []:
            if isinstance(tag, str) and not is_known_py_tag(tag):
                problems.append(
                    f"from_python.match: unknown construct tag {tag!r} "
                    "(see dualect/pyconstructs.py)"
                )
        if from_python.get("status") == Status.MECHANICAL and not _defines_lift(rule_dir):
            problems.append("from_python status 'mechanical' needs lift() in translate.py")

    status = data.get("status")
    has_cases = bool(find_cases(rule_dir=rule_dir))
    if status in (Status.GUIDED, Status.MECHANICAL) and not has_cases:
        problems.append(f"status {status!r} needs at least one case pair in cases/")
    if status == Status.MECHANICAL and not (rule_dir / TRANSLATOR_FILE).is_file():
        problems.append(f"status 'mechanical' needs a {TRANSLATOR_FILE}")

    return [f"{expected_id}: {problem}" for problem in problems]


def _defines_lift(rule_dir: Path) -> bool:
    """Whether the rule's ``translate.py`` defines a top-level ``lift`` function."""
    path = rule_dir / TRANSLATOR_FILE
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(isinstance(node, ast.FunctionDef) and node.name == "lift" for node in tree.body)


def validate_all(*, rules_dir: Path) -> list[str]:
    """Validate every rule under ``rules_dir``, including cross-rule references.

    :param rules_dir: root of the knowledge base.
    :returns: every problem found; empty when the whole knowledge base is valid.
    """
    dirs = rule_dirs(rules_dir=rules_dir)
    problems: list[str] = []
    known_ids = {rule_id_for(rule_dir=path) for path in dirs}
    owners: dict[str, str] = {}
    for rule_dir in dirs:
        problems.extend(validate_rule_dir(rule_dir=rule_dir))
        rule_id = rule_id_for(rule_dir=rule_dir)
        try:
            related = read_rule_yaml(path=rule_dir / "rule.yaml").get("related") or []
        except RuleError:
            continue  # already reported by validate_rule_dir
        problems.extend(_claim_tags(rule_dir=rule_dir, rule_id=rule_id, owners=owners))
        for other in related:
            if other not in known_ids:
                problems.append(f"{rule_id}: related rule {other!r} does not exist")
            elif other == rule_id:
                problems.append(f"{rule_id}: lists itself in related")
    return problems


def _claim_tags(*, rule_dir: Path, rule_id: str, owners: dict[str, str]) -> list[str]:
    """Record the tags a rule matches; two unversioned rules may not share one.

    Variant rules (docs/design.md section 4.4) may share a tag when at least one
    of them is bounded by ``since``/``until``; the dialect filter then decides.
    """
    try:
        data = read_rule_yaml(path=rule_dir / "rule.yaml")
    except RuleError:
        return []
    problems = []
    # Python tags are not filtered by source dialect: they may never be shared.
    block = data.get("from_python")
    tags = list(block.get("match") or []) if isinstance(block, dict) else []
    if not (data.get("since") or data.get("until")):
        tags = list(data.get("match") or []) + tags
    for tag in tags:
        other = owners.setdefault(str(tag), rule_id)
        if other != rule_id:
            problems.append(f"{rule_id}: tag {tag!r} is already matched by {other}")
    return problems
