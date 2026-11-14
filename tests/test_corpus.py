# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Testing layer 3 (docs/design.md section 7): the combination corpus.

Each ``corpus/*.m`` file is a realistic multi-construct script run through the
whole pipeline. A file the engine cannot fully translate is an honest result,
not a failure: the test checks that every untranslated statement is reported
and left as a hole, and that a *complete* translation verifies against the
interpreter, since a complete translation that does not verify means a
mechanical rule translated something wrong.
"""

import ast
from pathlib import Path

import pytest
from conftest import REPO_ROOT, RULES_DIR

from dualect import ir
from dualect.dialect import Dialect
from dualect.pipeline import UNVERIFIED, translate_file
from dualect.rulebook import load_rules
from dualect.runners import MRunner

RULES = load_rules(rules_dir=RULES_DIR)
CORPUS = sorted((REPO_ROOT / "corpus").glob("*.m"))


def test_corpus_exists() -> None:
    names = {path.stem for path in CORPUS}
    assert {"beam_deflection", "ode_rk4", "signal_filter"} <= names


@pytest.mark.parametrize("source", CORPUS, ids=lambda path: path.stem)
def test_corpus_translation_is_honest(source: Path, tmp_path: Path) -> None:
    output = tmp_path / f"{source.stem}.py"
    result = translate_file(
        source=source, output=output, rules=RULES, dialect=Dialect(), verify=False
    )
    text = output.read_text(encoding="utf-8")
    ast.parse(text)
    # Every top-level statement is accounted for in the report.
    reported = {(s.loc.line, s.kind) for s in result.statements}
    for stmt in result.scan.program.body:
        assert (stmt.loc.line, type(stmt).__name__) in reported
    assert text.count("raise NotImplementedError(") == len(result.holes)
    assert UNVERIFIED in text.splitlines()[0]


@pytest.mark.octave
@pytest.mark.parametrize("source", CORPUS, ids=lambda path: path.stem)
def test_corpus_complete_translations_verify(
    source: Path, tmp_path: Path, m_runner: MRunner
) -> None:
    result = translate_file(
        source=source,
        output=tmp_path / f"{source.stem}.py",
        rules=RULES,
        dialect=Dialect(),
        runner=m_runner,
    )
    if result.complete:
        assert result.verified, result.report()
    else:
        assert not result.verified
        assert all(s.reason for s in result.holes)
        mechanical = len(result.statements) - len(result.holes)
        print(f"{source.name}: {mechanical}/{len(result.statements)} statements mechanical")


@pytest.mark.octave
@pytest.mark.parametrize("source", CORPUS, ids=lambda path: path.stem)
def test_corpus_sources_run(source: Path, m_runner: MRunner) -> None:
    assert m_runner.run(path=source)


def test_corpus_statements_are_all_lowered() -> None:
    for source in CORPUS:
        from dualect.lower import lower_source

        program = lower_source(text=source.read_text(encoding="utf-8"), name=source.name)
        opaque = [s for s in ir.walk_stmts(program.body) if isinstance(s, ir.OpaqueStmt)]
        assert opaque == [], source.name


#: Corpus files every statement of which has a mechanical translation (both
#: directions are then verified by the tests above and in test_reverse.py).
COMPLETE = {"beam_deflection", "ode_rk4", "signal_filter"}


@pytest.mark.parametrize("name", sorted(COMPLETE))
def test_corpus_file_translates_completely(name: str, tmp_path: Path) -> None:
    source = REPO_ROOT / "corpus" / f"{name}.m"
    result = translate_file(
        source=source, output=tmp_path / f"{name}.py", rules=RULES, dialect=Dialect(), verify=False
    )
    assert result.complete, result.report()
