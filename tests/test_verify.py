# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""The verify harness on hand-written pairs, and directive parsing."""

from pathlib import Path

import pytest
from conftest import DATA_DIR

from dualect.errors import DirectiveError, ExecutionError
from dualect.runners import MRunner, run_py_script, run_py_script_capturing
from dualect.verify import STDOUT, parse_directives, read_directives, verify_pair


def test_parse_directives_both_comment_styles() -> None:
    text = "% dualect: index i, j\n# dualect: ignore k\nx = 1;\n"
    index_vars, ignore = parse_directives(text=text, path=Path("t.m"))
    assert index_vars == {"i", "j"}
    assert ignore == {"k"}


@pytest.mark.parametrize(
    "line", ["% dualect: indexes i", "# dualect: ignore", "% dualect: index 1x"]
)
def test_bad_directive_is_an_error(line: str) -> None:
    with pytest.raises(DirectiveError):
        parse_directives(text=line, path=Path("t.m"))


def test_python_runner_drops_scaffolding(tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text(
        "import numpy as np\nfrom math import sqrt\n_tmp = 1\n"
        "def helper():\n    return 1\nf = lambda t: t\nx = sqrt(4.0)\n"
    )
    outputs = run_py_script(path=script)
    assert set(outputs) == {"f", "x"}


def test_python_failure_is_named(tmp_path: Path) -> None:
    script = tmp_path / "bad.py"
    script.write_text("x = 1 / 0\n")
    with pytest.raises(ExecutionError) as info:
        run_py_script(path=script)
    assert info.value.side == "py"


@pytest.mark.octave
def test_hand_written_pair_passes(m_runner: MRunner) -> None:
    result = verify_pair(m_path=DATA_DIR / "beam.m", py_path=DATA_DIR / "beam.py", runner=m_runner)
    assert result.ok, result.report()
    names = {var.name for var in result.variables}
    assert {"w", "tip", "imax"} <= names


@pytest.mark.octave
def test_wrong_translation_fails(m_runner: MRunner, tmp_path: Path) -> None:
    # The classic off-by-one: end-2:end is three elements, not two.
    m_file = tmp_path / "t.m"
    py_file = tmp_path / "t.py"
    m_file.write_text("x = 1:5;\ntail = x(end-2:end);\n")
    py_file.write_text("import numpy as np\nx = np.arange(1, 6)\ntail = x[-2:]\n")
    result = verify_pair(m_path=m_file, py_path=py_file, runner=m_runner)
    assert not result.ok
    assert [var.name for var in result.variables if not var.ok] == ["tail"]


@pytest.mark.octave
def test_missing_python_variable_fails(m_runner: MRunner, tmp_path: Path) -> None:
    m_file = tmp_path / "t.m"
    py_file = tmp_path / "t.py"
    m_file.write_text("a = 1;\nb = 2;\n")
    py_file.write_text("a = 1\n")
    result = verify_pair(m_path=m_file, py_path=py_file, runner=m_runner)
    assert not result.ok
    assert "missing" in result.report()


@pytest.mark.octave
def test_nothing_compared_fails(m_runner: MRunner, tmp_path: Path) -> None:
    m_file = tmp_path / "t.m"
    py_file = tmp_path / "t.py"
    m_file.write_text("% only a comment\n")
    py_file.write_text("# only a comment\n")
    result = verify_pair(m_path=m_file, py_path=py_file, runner=m_runner)
    assert not result.ok


@pytest.mark.octave
def test_logical_type_is_preserved(m_runner: MRunner, tmp_path: Path) -> None:
    m_file = tmp_path / "t.m"
    m_file.write_text("mask = [1 2 3] > 1;\nflag = true;\n")
    outputs = m_runner.run(path=m_file)
    assert outputs["mask"].dtype == bool
    assert outputs["flag"].dtype == bool


@pytest.mark.octave
def test_octave_failure_is_named(m_runner: MRunner, tmp_path: Path) -> None:
    m_file = tmp_path / "bad.m"
    m_file.write_text("x = [1 2 3];\ny = x(5);\n")
    with pytest.raises(ExecutionError) as info:
        m_runner.run(path=m_file)
    assert info.value.side == "m"
    # The session stays usable after an error.
    ok_file = tmp_path / "ok.m"
    ok_file.write_text("z = 2;\n")
    assert m_runner.run(path=ok_file) == {"z": 2.0}


def test_flag_directives() -> None:
    text = "% dualect: stdout\n% dualect: shape r s\nx = 1;\n"
    found = read_directives(text=text, path=Path("t.m"))
    assert found.stdout and found.shape == {"r", "s"}
    with pytest.raises(DirectiveError):
        read_directives(text="% dualect: stdout x\n", path=Path("t.m"))


def test_python_runner_captures_output(tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text("x = 1\nprint('a  ')\nprint('b', end='')\n")
    outputs, printed = run_py_script_capturing(path=script)
    assert outputs == {"x": 1} and printed == "a  \nb"


@pytest.mark.octave
def test_printed_output_is_compared_exactly(m_runner: MRunner, tmp_path: Path) -> None:
    # Trailing spaces and a last line without a newline both count.
    m_file = tmp_path / "t.m"
    m_file.write_text("x = 1;\nprintf('a  \\n');\nprintf('b');\n% dualect: stdout\n")
    good, bad, silent = (tmp_path / name for name in ("good.py", "bad.py", "silent.py"))
    good.write_text("x = 1\nprint('a  ')\nprint('b', end='')\n")
    bad.write_text("x = 1\nprint('a')\nprint('b')\n")
    silent.write_text("x = 1\n")
    assert verify_pair(m_path=m_file, py_path=good, runner=m_runner).ok
    result = verify_pair(m_path=m_file, py_path=bad, runner=m_runner)
    assert not result.ok and STDOUT in result.report()
    assert not verify_pair(m_path=m_file, py_path=silent, runner=m_runner).ok


@pytest.mark.octave
def test_output_is_not_compared_without_the_directive(m_runner: MRunner, tmp_path: Path) -> None:
    m_file = tmp_path / "t.m"
    m_file.write_text("x = 1\n")  # displays "x = 1"
    py_file = tmp_path / "t.py"
    py_file.write_text("x = 1\n")
    assert verify_pair(m_path=m_file, py_path=py_file, runner=m_runner).ok


@pytest.mark.octave
def test_shape_directive_compares_shape_only(m_runner: MRunner, tmp_path: Path) -> None:
    m_file = tmp_path / "t.m"
    m_file.write_text("r = rand(2, 3);\n% dualect: shape r\n")
    right, wrong = tmp_path / "right.py", tmp_path / "wrong.py"
    right.write_text("import numpy as np\nr = np.random.rand(2, 3)\n")
    wrong.write_text("import numpy as np\nr = np.random.rand(3, 2)\n")
    assert verify_pair(m_path=m_file, py_path=right, runner=m_runner).ok
    assert not verify_pair(m_path=m_file, py_path=wrong, runner=m_runner).ok
