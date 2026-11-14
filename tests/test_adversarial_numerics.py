# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Adversarial probes of the statistics, interpolation, fitting and text rules.

Each snippet tries to make a rule translate silently wrong: N versus N - 1,
one element, empty and logical input, NaN, decreasing or out-of-range
interpolation points, integer data, dimensions and orientation, negative
zero, magnitudes that change a format. ``VERIFIES`` (``.m`` sources) and
``PY_VERIFIES`` (Python sources) must translate completely and verify
against the interpreter; the ``DECLINES`` snippets must leave the given
(1-based) line as a named hole.
"""

from pathlib import Path

import pytest
from conftest import RULES_DIR

from dualect.dialect import Dialect
from dualect.pipeline import translate_file, translate_python_text, translate_text
from dualect.rulebook import Status, load_rules
from dualect.runners import MRunner

RULES = load_rules(rules_dir=RULES_DIR)

#: .m snippets that must translate completely and verify.
VERIFIES: dict[str, str] = {
    "std-col": "c = [1; 4; 9];\ns = std(c);\nv = var(c, 1);\n",
    "std-mat-1row": "A = [1 2 3];\nB = [A; A];\ns = std(B);\n",
    "std-nan": "s = std([1 NaN 3]);\n",
    "median-even-int": "m = median([1 2 3 4]);\n",
    "median-mat-dim2": "A = [1 2 3; 4 5 9];\nm = median(A, 2);\n",
    "interp-func-x": "1;\nfunction y = f(x, v, q)\n  y = interp1(x, v, q);\nend\nr = f([1 2 3], [1 4 9], 2.5);\n",
    "interp-nan-query": "y = interp1([1 2 3], [1 4 9], [NaN 2]);\n",
    "interp-edges": "y = interp1([1 2 3], [1 4 9], [1 3]);\n",
    "interp-col-y": "y = interp1([1; 2; 3], [1; 4; 9], [1.5 2.5]);\n",
    "interp-loopvar-x": "x = 1:3;\nfor k = 1:2\n  x = x(end:-1:1);\nend\ny = interp1(x, [1 2 3], 1.5);\n",
    "polyfit-col": "x = [0; 1; 2; 3];\np = polyfit(x, x.^2, 2);\nq = polyval(p, [0.5; 1.5]);\n",
    "polyfit-int": "p = polyfit([1 2 3 4], [2 4 6 9], 1);\n",
    "polyval-mat": "A = [1 2; 3 4];\nv = polyval([1 0 -1], A);\n",
    "polyfit-degvar": "n = 2;\np = polyfit([1 2 3], [1 4 9], n);\n",
    "trapz-col": "c = [1; 2; 4];\nt = trapz(c);\nt2 = trapz([0; 1; 3], c);\n",
    "trapz-xrow-ycol": "t = trapz([0 1 3], [1; 2; 4]);\n",
    "trapz-int": "t = trapz([1 2 3], [2 4 8]);\n",
    "num2str-floor": "x = floor(7.9);\ns = num2str(x);\n",
    "num2str-prec-int": "s = num2str(42, 4);\ns2 = num2str(123456, 3);\n",
    "num2str-fmt-pad": "s = num2str(3, '  %d  ');\n",
    "num2str-negzero": "s = num2str(-0);\n",
    "floordiv-neg": "a = [-7 -0.5 3];\nq = floor(a / 2);\nq2 = floor(-1 / 3);\n",
    "floordiv-inf": "a = [Inf 3];\nq = floor(a / 2);\n",
    "floordiv-negdiv": "q = floor(7 / -2);\nq2 = floor([1 -1] ./ -3);\n",
    "floordiv-logical": "m = [1 0 1] > 0;\nq = floor(m / 2);\n",
    "expm1-int": "y = expm1([0 1 2]);\n",
    "true-var": "n = 2;\nm = 3;\nT = true(n, m);\nc = sum(T(:));\n",
    "true-one": "n = 1;\nT = true(n, 3);\n",
    "inf-n1": "x = Inf(3, 1);\ny = x(2);\n",
    "false-scalar": "f = false(1);\n",
}

#: Python snippets that must translate to .m completely and verify.
PY_VERIFIES: dict[str, str] = {
    "py-std-default": "import numpy as np\nx = np.array([1.0, 2.0, 4.0])\ns = x.std()\nv = np.var(x, ddof=1)\n",
    "py-std-axis1": "import numpy as np\nA = np.array([[1.0, 2.0, 4.0], [3.0, 3.0, 9.0]])\ns = A.std(axis=1)\nv = np.var(A, axis=0, ddof=1)\nw = A.var()\n",
    "py-median-2d": "import numpy as np\nA = np.array([[1.0, 5.0], [3.0, 2.0], [9.0, 0.0]])\nm = np.median(A)\nr = np.median(A, axis=1)\n",
    "py-interp-nan": "import numpy as np\nx = np.array([1.0, 2.0, 3.0])\nq = np.interp(np.array([0.0, 2.5]), x, x**2, left=np.nan, right=np.nan)\n",
    "py-interp-clamp": "import numpy as np\nx = np.array([1.0, 2.0, 3.0])\ny = np.array([1.0, 4.0, 9.0])\nq = np.interp([0.0, 2.5, 7.0], x, y)\n",
    "py-trapz-2d": "import numpy as np\nA = np.array([[1.0, 2.0, 4.0], [3.0, 3.0, 9.0]])\nt = np.trapezoid(A)\nt0 = np.trapezoid(A, axis=0)\ntx = np.trapezoid(A, x=[0.0, 1.0, 3.0])\n",
    "py-polyfit": "import numpy as np\nx = np.arange(5)\np = np.polyfit(x, x**2 + 1.0, 2)\nv = np.polyval(p, 2.5)\n",
    "py-strip": "s = '  ab \\t'\nt = s.strip()\n",
    "py-int-neg": "x = -2.5\nk = int(x)\n",
    "py-full": "import numpy as np\nA = np.full((2, 3), np.inf)\nv = np.full(4, np.nan)\nm = np.full((2, 2), True)\nz = np.full(3, False)\n",
    "py-ndim": "import numpy as np\nA = np.ones((2, 3))\nv = np.ones(3)\na = A.ndim\nb = v.ndim\n",
    "py-log1p": "import numpy as np\nx = np.array([1e-10, 0.5])\ny = np.log1p(x)\nz = np.expm1(x)\nr = np.real(np.array([1+2j]))\n",
    "py-floordiv": "import numpy as np\na = np.array([-7.0, 7.5])\nq = a // 2\n",
}

#: .m snippets whose given line must become a hole, not a guess.
DECLINES: dict[str, tuple[str, int]] = {
    "std-logical": ("m = [1 5 2 8] > 3;\ns = std(m);\n", 2),
    "std-weightvec": ("s = std([1 2 3], [1 1 2]);\n", 1),
    "median-empty": ("m = median([]);\n", 1),
    "median-logical": ("m = median([1 0 1] > 0);\n", 1),
    "interp-decr": ("y = interp1([3 2 1], [30 20 10], [1.5 2.5]);\n", 1),
    "interp-var-decr": ("x = [3 2 1];\ny = interp1(x, [30 20 10], 1.5);\n", 2),
    "interp-nearest": ("y = interp1([1 2 3], [1 4 9], 2.4, 'nearest');\n", 1),
    "interp-mat-y": ("y = interp1([1 2 3], [1 2; 3 4; 5 6], 2.5);\n", 1),
    "polyfit-S": ("[p, S] = polyfit([1 2 3], [1 4 9], 2);\n", 1),
    "trapz-scalar": ("t = trapz(5);\n", 1),
    "num2str-float": ("s = num2str(0.1);\n", 1),
    "num2str-prec-inf": ("s = num2str(Inf, 4);\n", 1),
    "num2str-arr": ("s = num2str([1 2 3]);\n", 1),
    "log1p-neg": ("y = log1p(-2);\n", 1),
    "nan-sizevec": ("N = NaN([2 3]);\n", 1),
}

#: Python snippets whose given line must become a hole.
PY_DECLINES: dict[str, tuple[str, int]] = {
    "py-std-one": ("import numpy as np\nx = np.array([5.0])\ns = x.std(ddof=1)\n", 3),
    "py-std-ddof2": ("import numpy as np\nx = np.array([5.0, 6.0, 9.0])\ns = x.std(ddof=2)\n", 3),
    "py-interp-left0": (
        "import numpy as np\nq = np.interp(5.0, [1.0, 2.0], [3.0, 4.0], left=0)\n",
        2,
    ),
    "py-trapz-dx": ("import numpy as np\nt = np.trapezoid([1.0, 2.0], dx=0.5)\n", 2),
    "py-str-float": ("x = 0.1\ns = str(x)\n", 2),
    "py-full-other": ("import numpy as np\nv = np.full(3, 7.0)\n", 2),
}


@pytest.mark.parametrize("name", sorted(VERIFIES))
def test_probe_translates_completely(name: str) -> None:
    result = translate_text(text=VERIFIES[name], name="p.m", rules=RULES, dialect=Dialect())
    assert result.complete, result.report()


@pytest.mark.parametrize("name", sorted(PY_VERIFIES))
def test_python_probe_translates_completely(name: str) -> None:
    result = translate_python_text(text=PY_VERIFIES[name], name="p.py", rules=RULES)
    assert result.complete, result.report()


@pytest.mark.octave
@pytest.mark.parametrize("name", sorted(VERIFIES))
def test_probe_verifies(name: str, tmp_path: Path, m_runner: MRunner) -> None:
    source = tmp_path / "probe.m"
    source.write_text(VERIFIES[name], encoding="utf-8")
    result = translate_file(
        source=source, output=tmp_path / "probe.py", rules=RULES, dialect=Dialect(), runner=m_runner
    )
    assert result.verified, result.report()


@pytest.mark.octave
@pytest.mark.parametrize("name", sorted(PY_VERIFIES))
def test_python_probe_verifies(name: str, tmp_path: Path, m_runner: MRunner) -> None:
    source = tmp_path / "probe.py"
    source.write_text(PY_VERIFIES[name], encoding="utf-8")
    result = translate_file(
        source=source, output=tmp_path / "probe.m", rules=RULES, dialect=Dialect(), runner=m_runner
    )
    assert result.verified, result.report()


@pytest.mark.parametrize("name", sorted(DECLINES))
def test_probe_is_a_named_hole(name: str) -> None:
    text, line = DECLINES[name]
    result = translate_text(text=text, name="p.m", rules=RULES, dialect=Dialect())
    holes = [h for h in result.holes if h.loc.line == line]
    assert holes, result.report()
    assert all(h.status is not Status.MECHANICAL and h.reason for h in holes)


@pytest.mark.parametrize("name", sorted(PY_DECLINES))
def test_python_probe_is_a_named_hole(name: str) -> None:
    text, line = PY_DECLINES[name]
    result = translate_python_text(text=text, name="p.py", rules=RULES)
    holes = [h for h in result.holes if h.loc.line == line]
    assert holes, result.report()
    assert all(h.status is not Status.MECHANICAL and h.reason for h in holes)


def test_interp1_never_clamps() -> None:
    # np.interp repeats the end values outside the samples; interp1 gives NaN.
    code = translate_text(
        text="y = interp1([1 2 3], [1 4 9], 5);\n", name="p.m", rules=RULES, dialect=Dialect()
    ).code
    assert "left=np.nan, right=np.nan" in code


def test_std_states_ddof() -> None:
    text = "x = [1 2 4];\ns = std(x);\nt = std(x, 1);\n"
    code = translate_text(text=text, name="p.m", rules=RULES, dialect=Dialect()).code
    assert "s = x.std(ddof=1)" in code and "t = x.std()" in code
