# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""The rule knowledge base: schema, conventions, and every case pair executed.

Each case pair is its own test, so a failure names exactly one rule and case.
"""

import pytest
from conftest import RULES_DIR

from dualect.compare import Tolerance
from dualect.rulebook import CasePair, Rule, Status, load_rules
from dualect.runners import MRunner, run_py_script
from dualect.scaffold import PLACEHOLDER
from dualect.validate import validate_all, validate_rule_dir
from dualect.verify import options_for_pair, verify_pair

RULES = load_rules(rules_dir=RULES_DIR)
CASES: list[tuple[Rule, CasePair]] = [(rule, case) for rule in RULES for case in rule.cases]


def _case_id(item: tuple[Rule, CasePair]) -> str:
    rule, case = item
    return f"{rule.id}:{case.name}"


def test_whole_knowledge_base_validates() -> None:
    assert validate_all(rules_dir=RULES_DIR) == []


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.id)
def test_rule_folder_is_valid(rule: Rule) -> None:
    assert validate_rule_dir(rule_dir=rule.path) == []


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.id)
def test_rule_has_no_placeholders(rule: Rule) -> None:
    # The scaffolder writes TODO markers; a rule that still has one is unfinished.
    for path in [rule.path / "rule.yaml", *(p for c in rule.cases for p in (c.m_path, c.py_path))]:
        assert PLACEHOLDER not in path.read_text(encoding="utf-8"), path


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.id)
def test_executable_rules_have_cases(rule: Rule) -> None:
    if rule.status in (Status.GUIDED, Status.MECHANICAL):
        assert rule.cases


@pytest.mark.parametrize("item", CASES, ids=_case_id)
def test_case_python_side_runs(item: tuple[Rule, CasePair]) -> None:
    # Cheap check that needs no Octave: the hand-written NumPy runs cleanly.
    _rule, case = item
    assert run_py_script(path=case.py_path)


@pytest.mark.octave
@pytest.mark.parametrize("item", CASES, ids=_case_id)
def test_case_pair_matches(item: tuple[Rule, CasePair], m_runner: MRunner) -> None:
    rule, case = item
    tolerance = Tolerance(**rule.tolerance)
    options = options_for_pair(m_path=case.m_path, py_path=case.py_path, tolerance=tolerance)
    result = verify_pair(m_path=case.m_path, py_path=case.py_path, options=options, runner=m_runner)
    assert result.ok, result.report()
