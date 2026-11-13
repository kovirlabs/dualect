# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""The ``dualect`` command line."""

from pathlib import Path

import pytest
from conftest import DATA_DIR, RULES_DIR

from dualect.cli import EXIT_FAILED, EXIT_NOT_IMPLEMENTED, EXIT_OK, main
from dualect.paths import RULES_DIR_ENV, default_rules_dir
from dualect.rulebook import load_rule
from dualect.validate import validate_rule_dir


@pytest.mark.parametrize(
    "argv", [["scan", "a.m"], ["context", "a.m"], ["translate", "a.m", "-o", "a.py"]]
)
def test_planned_commands_say_not_implemented(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == EXIT_NOT_IMPLEMENTED
    assert "not implemented yet" in capsys.readouterr().err


def test_rules_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["rules", "--rules-dir", str(tmp_path), "new", "indexing/foo"])
    capsys.readouterr()
    assert main(["rules", "--rules-dir", str(tmp_path), "list"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "indexing/foo" in out
    assert "unsupported" in out


def test_rules_list_repository(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rules", "--rules-dir", str(RULES_DIR), "list", "--status", "guided"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "indexing/end-in-index" in out
    assert "builtins/find" in out


def test_rules_list_status_filter(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rules", "--rules-dir", str(RULES_DIR), "list", "--status", "mechanical"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no rules found" in captured.err


def test_default_rules_dir_is_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RULES_DIR_ENV, raising=False)
    assert default_rules_dir() == RULES_DIR


def test_rules_new_scaffolds_a_loadable_rule(tmp_path: Path) -> None:
    assert main(["rules", "--rules-dir", str(tmp_path), "new", "indexing/foo-bar"]) == EXIT_OK
    rule_dir = tmp_path / "indexing" / "foo-bar"
    assert validate_rule_dir(rule_dir=rule_dir) == []
    rule = load_rule(rule_dir=rule_dir)
    assert rule.id == "indexing/foo-bar"
    assert [case.name for case in rule.cases] == ["basic"]


@pytest.mark.parametrize("rule_id", ["Indexing/foo", "foo", "a/b/c", "indexing/foo_bar"])
def test_rules_new_rejects_bad_ids(
    rule_id: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["rules", "--rules-dir", str(tmp_path), "new", rule_id]) == EXIT_FAILED
    assert "invalid rule id" in capsys.readouterr().err


def test_rules_new_refuses_to_overwrite(tmp_path: Path) -> None:
    assert main(["rules", "--rules-dir", str(tmp_path), "new", "syntax/x"]) == EXIT_OK
    assert main(["rules", "--rules-dir", str(tmp_path), "new", "syntax/x"]) == EXIT_FAILED


@pytest.mark.octave
def test_verify_command(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["verify", str(DATA_DIR / "beam.m"), str(DATA_DIR / "beam.py")])
    out = capsys.readouterr().out
    assert code == EXIT_OK, out
    assert out.rstrip().endswith("PASS")
