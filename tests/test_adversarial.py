# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Adversarial probes of the ``.m`` -> Python engine: never silently wrong.

Each snippet here once translated to code that ran but computed something
else (or is a near neighbour of one that did). The rule is the project's
central promise: a translation is either right, verified by running both
sides, or it is a named hole (guided/unsupported) in the report. The
``VERIFIES`` snippets must translate completely and verify against the
interpreter; the ``DECLINES`` snippets must leave the named statement as a hole.
"""

from pathlib import Path

import pytest
from conftest import RULES_DIR

from dualect.dialect import Dialect
from dualect.pipeline import translate_file, translate_python_text, translate_text
from dualect.rulebook import Status, load_rules
from dualect.runners import MRunner

RULES = load_rules(rules_dir=RULES_DIR)

#: Snippets that must translate completely and verify by execution.
VERIFIES: dict[str, str] = {
    # 1-D vectors: orientation matters for products, transposes, concatenation.
    "vectors": (
        "b = [1; 2; 3];\nr = [4 5 6];\nA = [1 2 3; 4 5 6; 7 8 10];\n"
        "outer_bb = b * b';\ninner_bb = b' * b;\nrt = r';\nrr = [r; r];\nbb = [b, b];\n"
        "Ab_ = [A, b];\nsb = size(b);\nsr = size(r);\nbc = b(:);\nrc = r(:);\n"
        "Ab = A * b;\nAsb = A \\ b;\nrA = r * A;\na = b' * A;\nc = A * r';\nd = r * r';\n"
        "k = A(:, 1) * A(1, :);\nm = A(1, :) * A(:, 1);\ns = A(:, 1) + A(1, :);\n"
        "v = A \\ r';\nw = r / A;\n"
    ),
    "vector-into-function": (
        "1;\nfunction y = f(v)\n  y = v * v';\nend\nb = [1; 2; 3];\nyb = f(b);\n"
    ),
    "precedence": (
        "x = [1 2 3];\ny = [4 5 6];\na = -2^-2;\nb = 2^3';\nc = x.^-1;\nd = ~x == 0;\n"
        "e = 1:3+2;\nf = -x';\ng = [1 -2];\nh = [1 - 2];\nk = [x' y'];\nn = 2^-1^2;\n"
        "o = -x.^2;\np = x(2)^2;\nq = ~x(1) + 1;\ns = 1 - - 1;\nt = [1 -2 +3];\n"
        "u = [x -1];\nv = 3 - 2 - 1;\nw = 12 / 3 / 2;\nz = x .* y + x ./ y - 2;\n"
    ),
    "number-dot-operators": (
        "x = [2 3];\nb = 2.^x;\nA = [1 2; 3 4];\nB = 2.^A;\nC = 3.*A;\nD = 1.5.^x;\nE = 2./x;\n"
    ),
    "end-in-nested-index": (
        "x = [10 20 30 40 50];\ny = [1 2 3];\nA = [1 2 3; 4 5 6; 7 8 9];\n"
        "a = x(y(end));\nb = x(min(end, 4));\nc = x(end-1);\nd = A(end, end-1);\n"
        "e = A(end);\nf = x(end:-1:1);\ng = x(end:-2:1);\nh = x(3:-1:1);\nk = A(:, end);\n"
        "m = x(2:end-1);\no = A(end, :);\np = A(2:end, 1);\nq = x(end-2:end);\n"
        "r = x([1 end]);\ns = x(y(end-1):end);\nt = A(end-1:end, end-1:end);\n"
    ),
    "ranges": (
        "a = 5:-1:1;\nb = 1:0.5:3;\nc = 0:0.1:1;\nd = 1:0;\ne = 5:1;\nf = 1:-1:5;\n"
        "h = -3:-1;\nk = 10:-3:1;\nn = 0:-0.25:-1;\no = numel(1:0);\np = 3:3;\n"
    ),
    "logical-arithmetic": (
        "x = [1 2 3];\na = true + 1;\nm = x > 1;\nb = m + m;\nc = m * 2;\nd = sum(m);\n"
        "e = ~[1 0 2];\nf = true & 1;\ng = m .* m;\nh = (x > 1) + 0;\nn = m | false;\n"
        "p = 3 > 2;\nq = p + p;\nr = x(m);\ns = ~p;\nt = -m;\nu = m(2) + m(3);\n"
        "v = m .^ 2;\nw = m' * m;\n"
    ),
    "integer-overflow": (
        "x = [2 3];\na = x .^ 40;\nb = x / 2;\nc = 7 / 2;\nd = floor(7 / 2);\n"
        "e = mod(-7, 3);\nf = rem(-7, 3);\ng = prod([100000 100000 100000 100000]);\n"
        "h = x .^ 70;\nk = 2 .^ (1:70);\nn = 3;\np = x .^ n;\n"
    ),
    "loops": (
        "s = 0;\nfor k = 1:3\n  k = k * 2;\n  s = s + k;\nend\nA = [1 2; 3 4];\nt = 0;\n"
        "for c = A\n  t = t + c(1) * 10 + c(2);\nend\nB = [1 2; 3 4];\nfor c = B\n"
        "  c(1) = 0;\nend\nw = 0;\nfor v = [3 1 2]\n  w = w * 10 + v;\nend\n"
        "cnt = 0;\nfor i = 5:-1:1\n  cnt = cnt + i;\nend\n"
    ),
    "lambda-same-shapes": ("f = @(x) x * x';\nb = [1; 2; 3];\na1 = f(b);\na2 = f(2 * b);\n"),
    # Printing is compared character for character (`# dualect: stdout`).
    "printf-formats": (
        "x = 3.14159;\nfprintf('%#g %#.3e %+05d [% +d] [%-05d]\\n', 2, 3, 4, 3, 3);\n"
        "fprintf('%e %E %G\\n', 1.5e-300, 2.5e10, 1e-10);\ns = 'abcdef';\n"
        "fprintf('[%8.3s][%-8.3s]\\n', s, s);\nfprintf('{x} = %d {} 50%%\\n', 3);\n"
        "fprintf('a\\\\b\\n');\nfprintf(\"tab\\there %d\\n\", 4);\nz = -0;\n"
        "fprintf('%d|%.1f\\n', z, z);\nn = 2^62;\nfprintf('%d\\n', n);\n"
        "m = 3 > 2;\nfprintf('%d %f\\n', m, m);\n"
    ),
    "printf-recycling": (
        "A = [1 2; 3 4];\nfprintf('%d,', A);\nfprintf('\\n');\nc = [1; 2; 3];\n"
        "fprintf('%d\\n', c);\nt = sprintf('%d;', A);\nfor k = 1:3\n"
        "  printf('%d:%d\\n', k, k^2);\nend\nfprintf('%d %d\\n', 7, 8, 9);\n"
    ),
    "filter-forms": (
        "y1 = filter([1 1], 2, [1 2 3]);\nX = [1 2 3; 4 5 6];\nY = filter(1, [1 -0.5], X);\n"
        "1;\nfunction y = f(x)\n  y = filter([1 1], 1, x);\nend\na = f([1 2 3]);\n"
        "b = f([1; 2; 3]);\n"
    ),
}

#: Snippets whose given (1-based) line must become a hole, not a guess.
DECLINES: dict[str, tuple[str, int]] = {
    # A lambda translated once cannot follow whichever call came last.
    "lambda-row-and-column": ("f = @(x) x * x';\nr = [1 2 3];\nb = [1; 2; 3];\n"
                              "a1 = f(r);\na2 = f(b);\n", 1),
    "lambda-size-row-and-column": ("f = @(x) size(x, 1);\nb = [1; 2; 3];\nr = [1 2 3];\n"
                                   "a1 = f(b);\na2 = f(r);\n", 1),
    # A function called with both orientations: `u * v` has no single `@` form.
    "function-row-and-column": ("1;\nfunction y = op(u, v)\n  y = u * v;\nend\nb = [1; 2; 3];\n"
                                "r = [4 5 6];\no1 = op(b, r);\no2 = op(r, b);\n", 3),
    "mod-by-zero": ("m = mod(-1, 0);\n", 1),
    "empty-loop": ("z = 0;\nfor j = 1:0\n  z = z + 1;\nend\n", 2),
    # %d of a fraction prints it in another format, not rounded or truncated.
    "printf-d-of-fraction": ("x = 7 / 2;\nfprintf('%d\\n', x);\n", 2),
    "printf-s-of-number": ("fprintf('%s\\n', 65);\n", 1),
    "printf-d-of-string": ("fprintf('%d\\n', 'ab');\n", 1),
    "printf-file-id": ("fid = 2;\nfprintf(fid, 'x\\n');\n", 2),
    "printf-hex": ("fprintf('%x\\n', 255);\n", 1),
    "printf-inf": ("fprintf('%f %g\\n', Inf, -Inf);\n", 1),
    "printf-huge-integer": ("n = 1e20;\nfprintf('%d\\n', n);\n", 2),
    "printf-empty-array": ("s = sprintf('x%dy', []);\n", 1),
    "printf-mixed-array": ("v = [1 2];\nfprintf('%d %d\\n', 3, v);\n", 2),
    "printf-rows-per-pass": ("A = [1 2 3; 4 5 6];\nfprintf('%d %d %d\\n', A);\n", 2),
    "disp-of-number": ("disp(0.1)\n", 1),
    "filter-scalar-data": ("y = filter(2, 1, 5);\n", 1),
    "filter-state": ("[y, z] = filter([1 1], 1, [1 2 3]);\n", 1),
}  # fmt: skip


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


def test_lambda_body_follows_every_call() -> None:
    # The body is annotated for the join of all calls, not the last one.
    code = translate_text(
        text=VERIFIES["lambda-same-shapes"], name="p.m", rules=RULES, dialect=Dialect()
    ).code
    assert "np.outer(x, x)" in code


def test_logical_operands_become_numbers() -> None:
    code = translate_text(
        text="m = [1 2 3] > 1;\nb = m + m;\n", name="p.m", rules=RULES, dialect=Dialect()
    ).code
    assert "b = m.astype(int) + m" in code


def test_loop_over_matrix_copies_a_written_column() -> None:
    code = translate_text(
        text="B = [1 2; 3 4];\nfor c = B\n  c(1) = 0;\nend\n",
        name="p.m",
        rules=RULES,
        dialect=Dialect(),
    ).code
    assert "for c in B.T.copy():" in code


def test_assumptions_stay_on_their_own_statement() -> None:
    code = translate_text(
        text="1;\nfunction c = f(a, b)\n  c = a * b;\nend\n",
        name="p.m",
        rules=RULES,
        dialect=Dialect(),
    ).code
    header = next(line for line in code.splitlines() if line.startswith("def f"))
    assert "assumed" not in header


#: Probes of the rules promoted to mechanical (grow-on-assign, nargin,
#: multiple-returns, command syntax) and of integer powers.
PROMOTED_VERIFIES: dict[str, str] = {
    "promoted-grow-col": "c = [1; 2];\nc(end+1) = 3;\nn = size(c, 1);\n",
    "promoted-grow-empty-row": "y = [];\ny(end+1) = 2;\ny(end+1) = 3;\ns = size(y);\n",
    "promoted-grow-new": "w(3) = 5;\n",
    "promoted-grow-new-2d": "W(2, 3) = 1;\n",
    "promoted-grow-frac-append": "x = [1 2];\nx(end+1) = 2.5;\n",
    "promoted-grow-matrix-rows": "A = [1 2; 3 4];\nA(3, 1) = 7;\n",
    "promoted-grow-loop-col": "c = zeros(0, 1);\nfor k = 1:3\n  c(end+1) = k;\nend\n",
    "promoted-grow-in-func": "1;\nfunction y = f(n)\n  y = [];\n  for k = 1:n\n    y(end+1) = k;\n  end\nend\na = f(3);\nb = f(0);\n",
    "promoted-grow-scalar": "s = 5;\ns(end+1) = 6;\n",
    "promoted-nargin-nonlast": "1;\nfunction y = g(x, k)\n  if nargin < 2\n    k = 3;\n  end\n  y = x * k;\nend\na = g(2);\nb = g(2, 5);\n",
    "promoted-nargin-array-default": "1;\nfunction y = g(x, w)\n  if nargin < 2\n    w = [1 2];\n  end\n  y = x * w;\nend\nr = g(2);\nt = g(2, [3 4]);\n",
    "promoted-multi-fewer": "1;\nfunction [a, b, c] = h(x)\n  a = x;\n  b = 2 * x;\n  c = 3 * x;\nend\n[p, q] = h(1);\n[~, ~, r] = h(2);\n",
    "promoted-multi-position": "1;\nfunction [a, b] = h(x)\n  a = x;\n  b = 2;\nend\nz = [10 20 30];\n[m, i] = h(1);\nw = z(i);\n",
    "promoted-cmd-format-loop": "for k = 1:2\n  format long\nend\nx = 1;\n",
    # A tall A \\ b is a least-squares solve (np.linalg.lstsq), not a decline.
    "least-squares-backslash": "t = [0 1 2 3 4];\nA = [ones(5, 1), t'];\ny = [1.1; 2.9; 5.2; 7.1; 8.8];\nc = A \\ y;\nr = y - A * c;\n",
    "integer-squares": "x = [3037000500 1];\ny = x .^ 2;\nz = x(1) ^ 2;\nA = [2000000 1; 1 1];\nB = A ^ 3;\n",
}
PROMOTED_DECLINES: dict[str, tuple[str, int]] = {
    "promoted-grow-int-then-frac": ("x = [1 2];\nx(end+1) = 3;\nx(1) = 0.5;\n", 3),
    "promoted-grow-branch": ("if rand() > 2\n  q = [1 2];\nend\nq(3) = 1;\n", 4),
    "promoted-grow-end2": ("x = [1 2];\nx(end+2) = 5;\n", 2),
    "promoted-grow-matrix-end": ("A = [1 2; 3 4];\nA(end+1, :) = [5 6];\n", 2),
    "promoted-grow-vector-value": ("x = [1 2];\nx(end+1) = [3 4];\n", 2),
    "promoted-nargin-empty-var": (
        "1;\nfunction y = g(x, lo)\n  if nargin < 2 || isempty(lo)\n    lo = 0;\n  end\n  y = x + lo;\nend\ne = zeros(1, 0);\na = g(1, e);\n",
        2,
    ),
    "promoted-nargin-wrong-param": (
        "1;\nfunction y = g(x, a, b)\n  if nargin < 2\n    b = 1;\n  end\n  y = x;\nend\nr = g(1);\n",
        2,
    ),
    "promoted-nargin-elsewhere": (
        "1;\nfunction y = g(x, k)\n  if nargin < 2\n    k = 3;\n  end\n  y = x * k + nargin;\nend\na = g(2);\n",
        2,
    ),
    "promoted-nargin-reversed-or": (
        "1;\nfunction y = g(x, k)\n  if isempty(k) || nargin < 2\n    k = 3;\n  end\n  y = x * k;\nend\na = g(2, 1);\n",
        2,
    ),
    "promoted-multi-into-element": (
        "1;\nfunction [a, b] = h(x)\n  a = x;\n  b = 2 * x;\nend\nv = [0 0];\n[v(1), v(2)] = h(1);\n",
        7,
    ),
    "promoted-multi-too-many": ("1;\nfunction a = h(x)\n  a = x;\nend\n[p, q] = h(1);\n", 5),
    "promoted-cmd-clear-undefined": ("a = 1;\nclear a b\n", 2),
    "promoted-cmd-clear-all": ("a = 1;\nclear\n", 2),
    "promoted-cmd-pkg": ("pkg load statistics\nx = 1;\n", 1),
}
#: Python sources whose given line must become a hole on the way to .m.
PY_DECLINES: dict[str, tuple[str, int]] = {
    "py-nargin-empty": (
        "import numpy as np\n\n\ndef g(x, lo=None):\n    if lo is None:\n        lo = 0\n    return x + lo\n\n\ne = np.zeros(0)\na = g(np.array([1.0]), e)\n",
        4,
    ),
    "py-pad": ("import numpy as np\n\nA = np.eye(2)\nB = np.pad(A, ((1, 0), (0, 1)))\n", 4),
    "py-append-2d": ("import numpy as np\n\nA = np.eye(2)\nb = np.append(A, 3)\n", 4),
}


@pytest.mark.parametrize("name", sorted(PROMOTED_VERIFIES))
def test_promoted_probe_translates_completely(name: str) -> None:
    result = translate_text(
        text=PROMOTED_VERIFIES[name], name="p.m", rules=RULES, dialect=Dialect()
    )
    assert result.complete, result.report()


@pytest.mark.octave
@pytest.mark.parametrize("name", sorted(PROMOTED_VERIFIES))
def test_promoted_probe_verifies(name: str, tmp_path: Path, m_runner: MRunner) -> None:
    source = tmp_path / "probe.m"
    source.write_text(PROMOTED_VERIFIES[name], encoding="utf-8")
    result = translate_file(
        source=source, output=tmp_path / "probe.py", rules=RULES, dialect=Dialect(), runner=m_runner
    )
    assert result.verified, result.report()


@pytest.mark.parametrize("name", sorted(PROMOTED_DECLINES))
def test_promoted_probe_is_a_named_hole(name: str) -> None:
    text, line = PROMOTED_DECLINES[name]
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
