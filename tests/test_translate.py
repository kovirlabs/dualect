# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Milestone 4: deterministic translators and the translate pipeline.

Testing layer 2 (docs/design.md section 7): every case of every ``mechanical``
rule translates, with no hole, to the same code as its hand-written ``.py``
(compared after formatting, comments ignored), and the translation verifies
by execution against the ``.m`` side.
"""

import ast
import difflib
from pathlib import Path

import pytest
from conftest import RULES_DIR

from dualect.cli import EXIT_FAILED, EXIT_INCOMPLETE, EXIT_OK, main
from dualect.dialect import Dialect, parse_dialect
from dualect.pipeline import UNVERIFIED, StrictModeError, translate_file, translate_text
from dualect.pyformat import normalized_code, same_code
from dualect.rulebook import CasePair, Rule, Status, load_rules
from dualect.runners import MRunner, run_py_script

RULES = load_rules(rules_dir=RULES_DIR)
ALL_CASES = [(rule, case) for rule in RULES for case in rule.cases]
MECHANICAL = [(rule, case) for rule, case in ALL_CASES if rule.status is Status.MECHANICAL]


def _id(item: tuple[Rule, CasePair]) -> str:
    return f"{item[0].id}:{item[1].name}"


def _translate(text: str, dialect: Dialect | None = None) -> str:
    result = translate_text(text=text, name="t.m", rules=RULES, dialect=dialect or Dialect())
    return result.code


@pytest.mark.parametrize("item", MECHANICAL, ids=_id)
def test_translator_matches_hand_written_case(item: tuple[Rule, CasePair]) -> None:
    _rule, case = item
    result = translate_text(
        text=case.m_path.read_text(encoding="utf-8"),
        name=str(case.m_path),
        rules=RULES,
        dialect=Dialect(),
        strict=True,
    )
    expected = case.py_path.read_text(encoding="utf-8")
    diff = "\n".join(
        difflib.unified_diff(
            normalized_code(expected).splitlines(),
            normalized_code(result.code).splitlines(),
            "hand-written",
            "translated",
            lineterm="",
        )
    )
    assert same_code(result.code, expected), diff


@pytest.mark.octave
@pytest.mark.parametrize("item", MECHANICAL, ids=_id)
def test_translated_case_verifies(
    item: tuple[Rule, CasePair], m_runner: MRunner, tmp_path: Path
) -> None:
    _rule, case = item
    result = translate_file(
        source=case.m_path,
        output=tmp_path / f"{case.name}.py",
        rules=RULES,
        dialect=Dialect(),
        strict=True,
        runner=m_runner,
    )
    assert result.verified, result.report()


@pytest.mark.parametrize("item", ALL_CASES, ids=_id)
def test_every_case_translates_or_reports_its_holes(item: tuple[Rule, CasePair]) -> None:
    _rule, case = item
    result = translate_text(
        text=case.m_path.read_text(encoding="utf-8"),
        name=str(case.m_path),
        rules=RULES,
        dialect=Dialect(),
    )
    ast.parse(result.text())
    assert result.text().count("raise NotImplementedError(") == len(result.holes)
    if result.holes:
        assert UNVERIFIED in result.header()


def test_guided_rules_are_not_secretly_mechanical() -> None:
    # The promotion loop (docs/design.md section 4.3): a guided rule whose cases
    # all translate cleanly to the hand-written code should be promoted.
    ready = []
    for rule in RULES:
        if rule.status is not Status.GUIDED or not rule.cases:
            continue
        results = [
            translate_text(
                text=c.m_path.read_text(encoding="utf-8"),
                name="c.m",
                rules=RULES,
                dialect=Dialect(),
            )
            for c in rule.cases
        ]
        if all(
            r.complete and same_code(r.code, c.py_path.read_text(encoding="utf-8"))
            for r, c in zip(results, rule.cases, strict=True)
        ):
            ready.append(rule.id)
    assert ready == [], f"promote to mechanical: {ready}"


# --- behavior that must never be silently wrong ------------------------------


def test_holes_raise_and_say_why() -> None:
    result = translate_text(
        text="x = [1 2 3];\ny = strsplit('a b', ' ');\nz = x(2);\n",
        name="t.m",
        rules=RULES,
        dialect=Dialect(),
    )
    assert not result.complete
    [hole] = result.holes
    assert hole.status is Status.UNSUPPORTED and "call:strsplit" in hole.reason
    assert "z = x[1]" in result.code  # the rest is still translated
    namespace: dict[str, object] = {}
    with pytest.raises(NotImplementedError, match="line 2"):
        exec(compile(result.code, "t.py", "exec"), namespace)


def test_strict_stops_and_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "s.m"
    source.write_text("x = [1 2 3];\nx(end+1) = 4;\n", encoding="utf-8")
    output = tmp_path / "s.py"
    with pytest.raises(StrictModeError, match="indexing/grow-on-assign"):
        translate_file(source=source, output=output, rules=RULES, dialect=Dialect(), strict=True)
    assert not output.exists()
    assert main(["translate", str(source), "-o", str(output), "--strict"]) == EXIT_FAILED
    assert not output.exists()


def test_integer_arrays_are_not_silently_truncated() -> None:
    # np.array([1, 2, 3]) holds integers; storing 0.5 into it would give 0.
    result = translate_text(
        text="x = [1 2 3];\nx(2) = 0.5;\n", name="t.m", rules=RULES, dialect=Dialect()
    )
    [hole] = result.holes
    assert "truncate" in hole.reason
    assert "x[1] = 0" in _translate("x = [1 2 3];\nx(2) = 0;\n")


def test_assignment_copies_when_either_side_is_modified() -> None:
    code = _translate("x = [1 2 3];\ny = x;\ny(1) = 0;\nz = x;\n")
    assert "y = x.copy()" in code
    assert "z = x\n" in code  # z is never modified in place: no copy needed


def test_unknown_shapes_are_assumed_out_loud() -> None:
    code = _translate("1;\nfunction c = f(a, b)\n  c = a * b;\nend\n")
    assert "a @ b" in code and "# dualect: assumed matrix multiply" in code


def test_column_against_row_gets_an_axis() -> None:
    code = _translate("r = [1 2 3];\nc = [1; 2];\ng = c + r;\n")
    assert "c[:, np.newaxis] + r" in code


def test_octave_only_syntax_is_unsupported_under_matlab() -> None:
    result = translate_text(
        text="n = 0;\nn += 1;\n", name="t.m", rules=RULES, dialect=parse_dialect("matlab")
    )
    assert [h.status for h in result.holes] == [Status.UNSUPPORTED]


def test_float_literals_keep_their_spelling() -> None:
    assert "E = 200e9" in _translate("E = 200e9;\n")


# --- end to end against the interpreter --------------------------------------


@pytest.mark.octave
def test_precedence_survives_translation(tmp_path: Path, m_runner: MRunner) -> None:
    # The grammar nests these wrongly; a wrong nesting would change the values.
    source = tmp_path / "p.m"
    source.write_text(
        "a = -2 ^ 2;\nb = 2 ^ -2 ^ 2;\nx = [1 2 3];\nc = x == 2 & x > 1;\n"
        "d = -x .^ 2;\ne = ~x == 0;\n",
        encoding="utf-8",
    )
    result = translate_file(
        source=source, output=tmp_path / "p.py", rules=RULES, dialect=Dialect(), runner=m_runner
    )
    assert result.verified, result.report()


@pytest.mark.octave
def test_translate_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case = RULES_DIR / "control-flow" / "while" / "cases" / "basic.m"
    output = tmp_path / "out.py"
    assert main(["translate", str(case), "-o", str(output)]) == EXIT_OK
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# Translated from basic.m by dualect; verified")
    assert "verified: yes" in capsys.readouterr().out
    run_py_script(path=output)


def test_translate_command_unverified(tmp_path: Path) -> None:
    case = RULES_DIR / "control-flow" / "while" / "cases" / "basic.m"
    output = tmp_path / "out.py"
    assert main(["translate", str(case), "-o", str(output), "--no-verify"]) == EXIT_INCOMPLETE
    assert UNVERIFIED in output.read_text(encoding="utf-8").splitlines()[0]
