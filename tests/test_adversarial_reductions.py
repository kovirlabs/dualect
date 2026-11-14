# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Adversarial probes of the cumsum, diff, any/all, sort, unique and flip rules.

Same promise as tests/test_adversarial.py: a translation either verifies by
execution or the statement is a named hole. The ``.py`` snippets probe the
Python -> ``.m`` direction.
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
    # One function called with a row and with a column: no orientation needed.
    "cumsum-both-orientations": (
        "1;\nfunction y = f(x)\n  y = cumsum(x);\nend\na = f([1 2 3]);\nb = f([1; 2; 3]);\n"
    ),
    "sort-both-orientations": (
        "1;\nfunction y = f(x)\n  y = sort(x, 'descend');\nend\na = f([3 1 2]);\n"
        "b = f([3; 1; 2]);\n"
    ),
    "unique-both-orientations": (
        "1;\nfunction y = f(x)\n  y = unique(x);\nend\na = f([3 1 3]);\nb = f([3; 1; 3]);\n"
    ),
    # Writing into a result must not write into the data (copies, views).
    "written-results": (
        "v = [1 2 3];\nc = cumsum(v);\nc(2) = 0;\nr = fliplr(v);\nv(1) = 9;\n"
        "A = [1 2; 3 4];\nB = flipud(A);\nB(1, 1) = 0;\nw = [3 1 2];\n"
        "d = sort(w, 'descend');\nd(1) = 0;\n"
    ),
    "sort-positions": (
        "v = [3 1 3 2];\n[s, i] = sort(v, 'descend');\nt = v(i);\n[~, k] = sort(v);\n"
        "u = v(k);\n[s2, j] = sort(v);\nj2 = j * 2;\n"
    ),
    "mixed": (
        "x = [1 NaN 2];\nc = cumsum(x);\nm = [1 0 1] > 0;\ns = cumsum(m) + 0.5;\n"
        "A = [0 1; 0 0];\nr = any(A, 2);\nn = sum(r);\nq = diff([1; 4; 9]);\n"
        "z = sort([]);\nst = fliplr('abc');\n"
    ),
}

#: .m snippets whose given line must become a hole.
DECLINES: dict[str, tuple[str, int]] = {
    # The source ignores NaN in any(); NumPy counts it as true.
    "any-of-nan": ("x = [0 NaN];\na = any(x);\n", 2),
    # diff of an order that reaches the length moves to the next dimension.
    "diff-order-too-high": ("x = [1 4 9];\nd = diff(x, 3);\n", 2),
    # Along a singleton dimension the source returns the data unchanged.
    "cumsum-singleton-dim": ("v = [1 2 3];\nc = cumsum(v, 1);\n", 2),
    "sort-singleton-dim": ("v = [3 1 2];\ns = sort(v, 1);\n", 2),
    # Descending positions of NaN: -NaN sorts last in NumPy.
    "sort-desc-positions-nan": ("v = [3 NaN 2];\n[s, i] = sort(v, 'descend');\n", 2),
    "sort-text": ("s = sort('hello');\n", 1),
    "unique-nan": ("u = unique([1 NaN NaN]);\n", 1),
    "unique-positions": ("[u, i, j] = unique([3 1 3]);\n", 1),
    "fliplr-unknown-orientation": (
        "1;\nfunction y = f(x)\n  y = fliplr(x);\nend\na = f([1 2 3]);\nb = f([1; 2; 3]);\n",
        3,
    ),
}

#: .py snippets that must lift completely and verify.
PY_VERIFIES: dict[str, str] = {
    "axes": (
        "import numpy as np\nA = np.array([[1, 2, 4], [3, 5, 9]])\nc = A.cumsum(axis=1)\n"
        "p = np.cumprod(A, axis=0)\nd = np.diff(A)\ne = np.diff(A, axis=0)\n"
        "r = A.any(axis=1)\nk = np.all(A, axis=0)\nt = A.all()\ns = np.sort(A)\n"
        "s0 = np.sort(A, axis=0)\nu = np.unique(A)\nf = np.flip(A, axis=1)\n"
        "g = np.flipud(A)\nv = np.array([1, 2, 3])\nw = np.flipud(v)\n"
        "q = np.array([2, 3]).prod(dtype=float)\n"
    ),
    "argsort-stable": (
        "import numpy as np\nv = np.array([2, 1, 2, 1])\ni = np.argsort(v, kind='stable')\nw = v[i]\n"
    ),
}

#: .py snippets whose given line must become a hole.
PY_DECLINES: dict[str, tuple[str, int]] = {
    # NumPy flattens a 2-D array in row order without an axis.
    "cumsum-2d-no-axis": (
        "import numpy as np\nA = np.array([[1, 2], [3, 4]])\nc = np.cumsum(A)\n",
        3,
    ),
    "any-of-nan": ("import numpy as np\nx = np.array([0, np.nan])\na = x.any()\n", 3),
    "argsort-unstable": ("import numpy as np\nv = np.array([2, 1, 2])\ni = np.argsort(v)\n", 3),
    "sort-flattened": (
        "import numpy as np\nA = np.array([[3, 1], [2, 5]])\nu = np.sort(A, axis=None)\n",
        3,
    ),
    "unique-nan": ("import numpy as np\nu = np.unique(np.array([1, np.nan, np.nan]))\n", 2),
    "flip-both-axes": ("import numpy as np\nA = np.array([[1, 2], [3, 4]])\nb = np.flip(A)\n", 3),
    # np.flip is a view: a later write into v shows through w in NumPy only.
    "flip-view-of-written": (
        "import numpy as np\nv = np.array([1, 2, 3])\nw = np.flip(v)\nv[0] = 9\n",
        3,
    ),
    "diff-order-too-high": ("import numpy as np\nx = np.array([1, 4, 9])\nd = np.diff(x, 3)\n", 3),
    "reduction-integer-dtype": (
        "import numpy as np\nx = np.array([2, 3])\ns = x.sum(dtype=int)\n",
        3,
    ),
}


@pytest.mark.parametrize("name", sorted(VERIFIES))
def test_probe_translates_completely(name: str) -> None:
    result = translate_text(text=VERIFIES[name], name="p.m", rules=RULES, dialect=Dialect())
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


@pytest.mark.parametrize("name", sorted(DECLINES))
def test_probe_is_a_named_hole(name: str) -> None:
    text, line = DECLINES[name]
    result = translate_text(text=text, name="p.m", rules=RULES, dialect=Dialect())
    holes = [h for h in result.holes if h.loc.line == line]
    assert holes, result.report()
    assert all(h.status is not Status.MECHANICAL and h.reason for h in holes)


@pytest.mark.octave
@pytest.mark.parametrize("name", sorted(PY_VERIFIES))
def test_python_probe_verifies(name: str, tmp_path: Path, m_runner: MRunner) -> None:
    source = tmp_path / "probe.py"
    source.write_text(PY_VERIFIES[name], encoding="utf-8")
    result = translate_file(
        source=source, output=tmp_path / "probe.m", rules=RULES, dialect=Dialect(), runner=m_runner
    )
    assert result.verified, result.report()


@pytest.mark.parametrize("name", sorted(PY_DECLINES))
def test_python_probe_is_a_named_hole(name: str) -> None:
    text, line = PY_DECLINES[name]
    result = translate_python_text(text=text, name="p.py", rules=RULES)
    holes = [h for h in result.holes if h.loc.line == line]
    assert holes, result.report()
    assert all(h.status is not Status.MECHANICAL and h.reason for h in holes)
