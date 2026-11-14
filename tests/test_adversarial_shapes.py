# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Adversarial probes of the shape and linear-algebra built-ins, both directions.

``reshape``, ``repmat``, ``horzcat``/``vertcat``/``cat``, ``meshgrid``,
``rand``/``randn``, ``norm``, ``dot``/``cross``/``kron`` and
``inv``/``det``/``trace``/``rank``: each snippet here either must translate
completely and verify against the interpreter, or must leave the named
statement as a hole. A source starting with ``#py`` is Python (translated to
``.m``; hole lines count from the line after the marker); the rest are ``.m``.
"""

from pathlib import Path

import pytest
from conftest import RULES_DIR

from dualect.dialect import Dialect
from dualect.pipeline import (
    Translation,
    translate_file,
    translate_python_text,
    translate_text,
)
from dualect.rulebook import Status, load_rules
from dualect.runners import MRunner

RULES = load_rules(rules_dir=RULES_DIR)
PY = "#py\n"

#: Snippets that must translate completely and verify by execution.
VERIFIES: dict[str, str] = {
    "reshape-forms": (
        "v = 1:6;\nc = reshape(v, [], 1);\nd = c(2);\nA = [1 2 3];\nB = reshape(A, 3, []);\n"
        "t = sum(B);\nM = reshape(1:6, 3, 2);\nx = M(4);\ny = M(end);\ns = sum(M);\n"
        "n = 1;\nr = reshape([1 2; 3 4], n, []);\n"
    ),
    "reshape-copy": "A = [1 2; 3 4];\nr = reshape(A, 4, 1);\nr(1) = 9;\na = A(1);\n",
    "reshape-from-python": (
        PY + "import numpy as np\nv = np.arange(6)\nA = np.reshape(v, (2, 3))\n"
        "B = np.reshape(A, (3, 2))\nc = np.reshape(A, -1)\nD = np.reshape(A, (-1, 2))\n"
        'E = np.reshape(A, (3, 2), order="F")\n'
    ),
    "repmat-orientation": (
        "v = [1 2 3];\nc = [4; 5];\nA = repmat(v, 2, 2);\nB = repmat(c, 2, 3);\n"
        "d = repmat(c, 2, 1);\ns = sum(B);\nm = [1 2 3] > 1;\nM = repmat(m, 2, 1);\n"
        "k = sum(M(:));\na = repmat(5, 2, 3);\nb = repmat(true, 1, 3);\n"
    ),
    "tile-from-python": (
        PY + "import numpy as np\nv = np.array([1, 2, 3])\nA = np.array([[1, 2], [3, 4]])\n"
        "B = np.tile(v, 2)\nC = np.tile(v, (2, 1))\nD = np.tile(A, 2)\nE = np.tile(A, (2, 3))\n"
    ),
    "concatenation-functions": (
        "c = [1; 2];\nd = [3; 4];\nM = cat(2, c, d);\nv = cat(1, c, d);\nx = horzcat(1, 2, 3);\n"
        "y = vertcat(1, 2);\nz = cat(1, [1 2], [3 4]);\n"
    ),
    "hstack-from-python": (
        PY + "import numpy as np\na = np.array([1, 2])\nb = np.array([3, 4, 5])\n"
        "A = np.array([[1, 2], [3, 4]])\nc = np.hstack((a, b))\nC = np.hstack((A, A))\n"
        "d = np.hstack((a, 7))\n"
    ),
    # np.meshgrid(x) is one grid; meshgrid(x) is two: the vector is passed twice.
    "meshgrid-forms": (
        "X = meshgrid(1:3, 1:2);\n[~, Y] = meshgrid(1:3, [4 5]);\ns = sum(Y(:));\n"
        "x = [1; 2; 3];\n[P, Q] = meshgrid(x);\n[G, H] = meshgrid(1:3, 1:2);\nZ = G .^ 50;\n"
    ),
    "meshgrid-from-python": (
        PY + "import numpy as np\nx = np.linspace(0, 1, 4)\ny = np.array([1.0, 2.0])\n"
        "X, Y = np.meshgrid(x, y)\nZ = X * Y\n"
    ),
    "random-shapes": "a = rand(1);\nb = randn(1, 1);\nr = rand([2 3]);\nc = randn(3, 1);\n",
    "random-from-python": (
        PY + "import numpy as np\na = np.random.default_rng().random((2, 3))\n"
        "b = np.random.default_rng().standard_normal(4)\nc = np.random.rand(3)\n"
        "d = np.random.default_rng().random()\n"
    ),
    "norms": (
        "v = [1 -2 3];\nc = [1; -2; 3];\na = norm(v);\nb = norm(c, 1);\nd = norm(v, -Inf);\n"
        "e = norm(v, 3);\nf = norm(v, 'fro');\ng = norm(c, 'inf');\nA = [1 2; 3 4; 5 6];\n"
        "h = norm(A);\nk = norm(A, 'fro');\nm = norm(A, 1);\nn = norm(A, Inf);\n"
        "o = norm([1+2i 3]);\n"
    ),
    "norms-from-python": (
        PY
        + "import numpy as np\nv = np.array([3.0, -4.0])\nA = np.array([[1.0, 2.0], [3.0, 4.0]])\n"
        "a = np.linalg.norm(v)\nb = np.linalg.norm(A)\nc = np.linalg.norm(A, 2)\n"
        "d = np.linalg.norm(v, np.inf)\ne = np.linalg.norm(A, ord=1)\n"
    ),
    # A column and a row: their Kronecker product is a matrix, not a long vector.
    "kron-orientation": (
        "r = [1 2 3];\nc = [1; 2];\nA = kron(r, c);\nB = kron(c, r);\ns = sum(A);\nd = [3; 4];\n"
        "k = kron(c, d);\nk2 = k(3);\nM = [1 2; 3 4];\nK = kron(M, c);\nL = kron([1 2], M);\n"
        "N = kron(2, [1 2 3]);\nP = kron([2 3], [4 5]) .^ 40;\nw = cross([1; 0; 0], [0; 1; 0]);\n"
    ),
    "products-from-python": (
        PY + "import numpy as np\nu = np.array([1.0, 2.0, 3.0])\nv = np.array([4.0, 5.0, 6.0])\n"
        "d = np.dot(u, v)\nw = np.cross(u, v)\nk = np.kron(u, v)\nK = np.kron(np.eye(2), u)\n"
    ),
    "linear-algebra-integers": (
        "A = [2 1; 1 3];\nB = inv(A);\nd = det(A);\nt = trace(A);\nr = rank(A);\n"
        "B(1, 1) = 0.5;\ne = trace(zeros(0, 0));\n"
    ),
}

#: Snippets whose given (1-based) line must become a hole, not a guess.
DECLINES: dict[str, tuple[str, int]] = {
    # np.meshgrid keeps integer dtypes: a fraction stored into a grid would truncate.
    "meshgrid-integer-store": ("[X, Y] = meshgrid(1:3, 1:2);\nX(1, 1) = 0.5;\n", 2),
    "kron-integer-store": ("K = kron([1 2], [3 4]);\nK(1) = 0.5;\n", 2),
    "cross-integer-store": ("w = cross([1 0 0], [0 1 0]);\nw(1) = 0.5;\n", 2),
    "meshgrid-one-point": ("[X, Y] = meshgrid([1 2 3], 5);\n", 1),
    "meshgrid-of-matrix": ("A = [1 2; 3 4];\n[X, Y] = meshgrid(A, [1 2]);\n", 2),
    "cat-third-dimension": ("A = [1 2; 3 4];\nB = cat(3, A, A);\n", 2),
    "horzcat-row-and-column": ("a = [1 2];\nc = [3; 4];\nx = horzcat(a, c);\n", 3),
    "rand-seed": ("rand('seed', 1);\n", 1),
    "norm-matrix-p3": ("A = [1 2; 3 4];\nn = norm(A, 3);\n", 2),
    "norm-unknown-shape": (
        "1;\nfunction n = f(x)\n  n = norm(x);\nend\na = f([1 2 3]);\nb = f([1 2; 3 4]);\n", 3
    ),
    "dot-complex": ("u = [1+1i 2];\nv = [1 1];\nd = dot(u, v);\n", 3),
    "dot-of-matrices": ("A = [1 2; 3 4];\nd = dot(A, A);\n", 2),
    "cross-row-and-column": ("w = cross([1 0 0], [0; 1; 0]);\n", 1),
    "kron-unknown-orientation": (
        "1;\nfunction k = f(a, b)\n  k = kron(a, b);\nend\nx = f([1 2], [1; 2]);\n"
        "y = f([1; 2], [1 2]);\n",
        3,
    ),
    "trace-not-square": ("W = [1 2 3; 4 5 6];\nt = trace(W);\n", 2),
    "inv-of-scalar": ("x = inv(4);\n", 1),
    "py-meshgrid-ij": (
        PY + 'import numpy as np\nX, Y = np.meshgrid(np.arange(3), np.arange(2), indexing="ij")\n',
        2,
    ),
    "py-meshgrid-one-vector": (PY + "import numpy as np\nX, Y = np.meshgrid(np.arange(3))\n", 2),
    "py-hstack-mixed": (
        PY + "import numpy as np\na = np.array([1, 2])\nA = np.array([[1, 2], [3, 4]])\n"
        "C = np.hstack((A, a))\n",
        4,
    ),
    "py-norm-matrix-minus-inf": (
        PY + "import numpy as np\nA = np.array([[1.0, 2.0], [3.0, 4.0]])\n"
        "n = np.linalg.norm(A, -np.inf)\n",
        3,
    ),
    "py-dot-of-matrices": (
        PY + "import numpy as np\nA = np.array([[1.0, 2.0], [3.0, 4.0]])\nB = np.dot(A, A)\n", 3
    ),
    "py-cross-2d-vectors": (
        PY + "import numpy as np\na = np.array([1.0, 2.0])\nb = np.array([3.0, 4.0])\n"
        "c = np.cross(a, b)\n",
        4,
    ),
    "py-kron-integer-store": (
        PY + "import numpy as np\nK = np.kron(np.array([1, 2]), np.array([3, 4]))\nK[0] = 0.5\n",
        3,
    ),
    "py-seeded-generator": (PY + "import numpy as np\na = np.random.default_rng(1).random(3)\n", 2),
}  # fmt: skip


def _translate(text: str) -> Translation:
    if text.startswith(PY):
        return translate_python_text(text=text.removeprefix(PY), name="p.py", rules=RULES)
    return translate_text(text=text, name="p.m", rules=RULES, dialect=Dialect())


@pytest.mark.parametrize("name", sorted(VERIFIES))
def test_probe_translates_completely(name: str) -> None:
    result = _translate(VERIFIES[name])
    assert result.complete, result.report()


@pytest.mark.octave
@pytest.mark.parametrize("name", sorted(VERIFIES))
def test_probe_verifies(name: str, tmp_path: Path, m_runner: MRunner) -> None:
    text = VERIFIES[name]
    is_python = text.startswith(PY)
    source = tmp_path / ("probe.py" if is_python else "probe.m")
    source.write_text(text.removeprefix(PY), encoding="utf-8")
    output = tmp_path / ("probe_out.m" if is_python else "probe_out.py")
    result = translate_file(
        source=source, output=output, rules=RULES, dialect=Dialect(), runner=m_runner
    )
    assert result.verified, result.report()


@pytest.mark.parametrize("name", sorted(DECLINES))
def test_probe_is_a_named_hole(name: str) -> None:
    text, line = DECLINES[name]
    result = _translate(text)
    holes = [h for h in result.holes if h.loc.line == line]
    assert holes, result.report()
    assert all(h.status is not Status.MECHANICAL and h.reason for h in holes)
