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
from dualect.pipeline import translate_file, translate_text
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
