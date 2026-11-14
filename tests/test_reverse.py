# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Milestone 5: Python -> ``.m`` through the shared IR.

Every rule case's ``.py`` is translated to ``.m``; a complete translation must
verify by execution against the original ``.py``, and an incomplete one must
report each hole by name. The corpus runs in reverse too: each ``corpus/*.m``
is translated to Python and that Python back to ``.m``. Python constructs
outside the subset are reported as unsupported by name, and Python behaviours
with no ``.m`` counterpart (aliasing views, writes into arguments, late-binding
lambdas, ...) are declined rather than translated as if they did not exist.
"""

from pathlib import Path

import pytest
from conftest import REPO_ROOT, RULES_DIR

from dualect.cli import EXIT_OK, main
from dualect.dialect import Dialect
from dualect.pipeline import UNVERIFIED, translate_file, translate_python_text
from dualect.rulebook import CasePair, Rule, Status, load_rules
from dualect.runners import MRunner
from dualect.scan import scan_file

RULES = load_rules(rules_dir=RULES_DIR)
ALL_CASES = [(rule, case) for rule in RULES for case in rule.cases]
CORPUS = sorted((REPO_ROOT / "corpus").glob("*.m"))


def _id(item: tuple[Rule, CasePair]) -> str:
    return f"{item[0].id}:{item[1].name}"


def _lift(text: str) -> str:
    return translate_python_text(text=text, name="t.py", rules=RULES).code


def _holes(text: str) -> list[str]:
    result = translate_python_text(text=text, name="t.py", rules=RULES)
    return [h.reason for h in result.holes]


@pytest.mark.parametrize("item", ALL_CASES, ids=_id)
def test_every_case_lifts_or_reports_its_holes(item: tuple[Rule, CasePair]) -> None:
    _rule, case = item
    result = translate_python_text(
        text=case.py_path.read_text(encoding="utf-8"), name=str(case.py_path), rules=RULES
    )
    assert result.text().count("error('dualect: line") == len(result.holes)
    assert all(h.reason for h in result.holes)
    if result.holes:
        assert UNVERIFIED in result.header()


@pytest.mark.octave
@pytest.mark.parametrize("item", ALL_CASES, ids=_id)
def test_complete_case_translations_verify(
    item: tuple[Rule, CasePair], tmp_path: Path, m_runner: MRunner
) -> None:
    _rule, case = item
    result = translate_file(
        source=case.py_path,
        output=tmp_path / f"{case.name}.m",
        rules=RULES,
        dialect=Dialect(),
        runner=m_runner,
    )
    if result.complete:
        assert result.verified, result.report()
    else:
        assert not result.verified


@pytest.mark.octave
@pytest.mark.parametrize("source", CORPUS, ids=lambda path: path.stem)
def test_corpus_round_trip(source: Path, tmp_path: Path, m_runner: MRunner) -> None:
    python = tmp_path / f"{source.stem}.py"
    forward = translate_file(
        source=source, output=python, rules=RULES, dialect=Dialect(), runner=m_runner
    )
    back = translate_file(
        source=python,
        output=tmp_path / f"{source.stem}_back.m",
        rules=RULES,
        dialect=Dialect(),
        runner=m_runner,
    )
    mechanical = len(back.statements) - len(back.holes)
    print(
        f"{source.name}: forward complete={forward.complete}; "
        f"back {mechanical}/{len(back.statements)}"
    )
    if back.complete:
        assert back.verified, back.report()
    else:
        assert all(h.reason for h in back.holes)


# --- honesty: named gaps, never a silent guess -----------------------------------


@pytest.mark.parametrize(
    ("source", "tag"),
    [
        ("import scipy\n", "py:import:scipy"),
        ("import numpy as np\ny = np.fft.fft([1, 2])\n", "py:call:np.fft.fft"),
        ("try:\n    x = 1\nexcept ValueError:\n    x = 2\n", "py:stmt:try"),
        ("xs = [k for k in range(3)]\n", "py:expr:ListComp"),
        ("a = 1 < 2 < 3\n", "py:cmp:chain"),
    ],
)
def test_unsupported_constructs_are_named(source: str, tag: str) -> None:
    reasons = _holes(source)
    assert reasons and tag in reasons[0]


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        # y is a view of x: writing into y changes x in NumPy.
        ("import numpy as np\nx = np.array([1, 2, 3])\ny = x\ny[0] = 5\n", "shares memory"),
        ("import numpy as np\nx = np.array([1, 2, 3])\ny = x[0:2]\ny[0] = 5\n", "shares memory"),
        # The caller's array changes in NumPy, never in .m code.
        (
            "def f(a):\n    a[0] = 1\n    return a\nimport numpy as np\nv = f(np.array([1, 2]))\n",
            "writing into the argument",
        ),
        # A .m function cannot see the script's variables.
        ("k = 2\ndef f(x):\n    return k * x\ny = f(3)\n", "module variable k"),
        # Python reads k when the lambda is called; a .m handle captured it.
        ("k = 2\nf = lambda t: k * t\nk = 10\ny = f(3)\n", "changes later"),
        # A variable would hide the function on the .m side.
        ("import numpy as np\nmax = 3\nx = np.array([1, 2])\ny = x.max()\n", "hide the function"),
        # NumPy truncates a fraction stored into an integer array.
        ("import numpy as np\nx = np.array([1, 2, 3])\nx[0] = 0.5\n", "truncates"),
        # bool + bool is a logical OR in NumPy.
        ("import numpy as np\nm = np.array([1, 2]) > 1\nb = m + m\n", "boolean arrays"),
        # NumPy's max propagates NaN; the .m max skips it.
        ("import numpy as np\nw = np.array([1, np.nan])\nm = w.max()\n", "NaN"),
        # linspace with one point: NumPy gives the start, the .m side the end.
        ("import numpy as np\nx = np.linspace(0, 1, 1)\n", "count of 1"),
        # `and` returns an operand in Python.
        ("a = 2\nb = 3\nc = a and b\n", "whose value"),
        # Python's ~ on integers is bitwise.
        ("x = 5\ny = ~x\n", "bitwise"),
        # np.round rounds halves to even.
        ("import numpy as np\nx = np.round(np.array([0.5, 1.5]))\n", "halves to even"),
    ],  # fmt: skip
)
def test_python_only_behaviour_is_declined(source: str, reason: str) -> None:
    reasons = _holes(source)
    assert any(reason in r for r in reasons), reasons


def test_one_d_arrays_are_rows() -> None:
    code = _lift(
        "import numpy as np\nA = np.array([[1, 2], [3, 4]])\nv = np.array([1, 1])\n"
        "y = A @ v\nu = v @ A\nd = v @ v\no = np.outer(v, v)\nc = A[:, 0]\n"
    )
    assert "y = (A * v.').';" in code
    assert "u = v * A;" in code
    assert "d = v * v.';" in code
    assert "o = v.' * v;" in code
    assert "c = A(:, 1).';" in code


def test_indices_shift_and_count_from_the_end() -> None:
    code = _lift(
        "import numpy as np\nx = np.array([1, 2, 3, 4])\na = x[0]\nb = x[-1]\nc = x[1:3]\n"
        "d = x[::-1]\ne = x[-2:]\n"
    )
    for line in (
        "a = x(1);",
        "b = x(end);",
        "c = x(2:3);",
        "d = x(end:-1:1);",
        "e = x(end - 1:end);",
    ):
        assert line in code


def test_unknown_index_sign_is_assumed_out_loud() -> None:
    code = _lift(
        "import numpy as np\ndef f(x, i):\n    return x[i]\n"
        "k = -1 + 1\ny = f(np.array([1, 2]), k)\n"
    )
    assert "% dualect: assumed non-negative index i" in code


def test_verify_directives_are_not_copied() -> None:
    code = _lift("i = 1  # dualect: index i\n# dualect: ignore i\nj = 2  # a note\n")
    assert "dualect: index" not in code and "dualect: ignore" not in code
    assert "j = 2;  % a note" in code


def test_scan_and_context_read_python(tmp_path: Path) -> None:
    source = tmp_path / "s.py"
    source.write_text("import numpy as np\nx = np.zeros(3)\nprint(x)\n", encoding="utf-8")
    result = scan_file(path=source, rules=RULES, dialect=Dialect())
    ids = {rule.id for rule in result.needed_rules()}
    assert {"builtins/zeros-ones-eye", "io/printf"} <= ids
    assert result.as_json()["direction"] == "from-python"
    assert result.status_of(next(r for r in RULES if r.id == "io/printf")) is Status.MECHANICAL


@pytest.mark.octave
def test_translate_command_from_python(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case = RULES_DIR / "control-flow" / "while" / "cases" / "basic.py"
    output = tmp_path / "out.m"
    assert main(["translate", str(case), "-o", str(output)]) == EXIT_OK
    text = output.read_text(encoding="utf-8")
    assert text.startswith("% Translated from basic.py by dualect; verified against the .py")
    assert "verified: yes" in capsys.readouterr().out
