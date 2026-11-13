# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Milestone 3: parsing (with the Octave dialect), lowering, scanning, context."""

from pathlib import Path

import pytest
from conftest import RULES_DIR

from dualect import ir
from dualect.cli import EXIT_OK, main
from dualect.context import build_context
from dualect.dialect import Dialect, DialectError, parse_dialect
from dualect.lower import lower_source
from dualect.rulebook import CasePair, Rule, load_rules
from dualect.scan import OCTAVE_ONLY_TAG, RuleIndex, scan_file, scan_text
from dualect.source import normalize

RULES = load_rules(rules_dir=RULES_DIR)
CASES: list[tuple[Rule, CasePair]] = [(rule, case) for rule in RULES for case in rule.cases]
INDEX = RuleIndex(rules=RULES, dialect=Dialect())


def _scan(text: str, dialect: Dialect | None = None) -> set[str]:
    index = INDEX if dialect is None else RuleIndex(rules=RULES, dialect=dialect)
    result = scan_text(text=text, name="t.m", index=index)
    return {rule.id for rule in result.needed_rules()}


def _tags(text: str) -> set[str]:
    result = scan_text(text=text, name="t.m", index=INDEX)
    return {finding.tag for finding in result.findings}


@pytest.mark.parametrize("item", CASES, ids=lambda item: f"{item[0].id}:{item[1].name}")
def test_scanning_a_rule_case_finds_that_rule(item: tuple[Rule, CasePair]) -> None:
    rule, case = item
    result = scan_file(path=case.m_path, rules=RULES, dialect=Dialect())
    assert rule.id in {found.id for found in result.needed_rules()}, result.report()


@pytest.mark.parametrize("item", CASES, ids=lambda item: f"{item[0].id}:{item[1].name}")
def test_rule_cases_parse_without_errors(item: tuple[Rule, CasePair]) -> None:
    # Every case uses only syntax the parser (plus normalization) understands.
    _rule, case = item
    result = scan_file(path=case.m_path, rules=RULES, dialect=Dialect())
    assert "parse:error" not in {f.tag for f in result.findings}, result.report()


# --- the Octave dialect in front of a grammar that only knows the other one ---


def test_normalize_keeps_lines_and_marks_features() -> None:
    source = "x = 1; # note\nif a != b, c = !d; endif\nk += 2;\nn++;\n#{\nhidden\n#}\n"
    normalized = normalize(source)
    assert normalized.text.count("\n") == source.count("\n")
    assert "#" not in normalized.text and "!" not in normalized.text
    assert "endif" not in normalized.text
    names = {feature.name for feature in normalized.features}
    assert {"# comment", "!=", "! (not)", "endif", "+=", "++", "#{ #} block comment"} <= names


def test_normalize_leaves_strings_and_transposes_alone() -> None:
    source = "s = '# not a comment != ok';\nt = \"it's #1\";\nu = x';\n"
    assert normalize(source).text == source


def test_do_until_is_a_loop_not_two_commands() -> None:
    program = lower_source(text="n = 0;\ndo\n  n++;\nuntil n >= 3\n")
    loop = program.body[1]
    assert isinstance(loop, ir.DoUntil)
    assert isinstance(loop.body[0], ir.AugAssign)
    assert isinstance(loop.cond, ir.BinOp) and loop.cond.op == "ge"
    assert loop.loc.line == 2 and loop.cond.loc.line == 4


def test_stray_until_is_a_parse_error() -> None:
    assert "parse:error" in _tags("x = 1;\nuntil x > 0\n")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The grammar nests these wrongly; the lowering rebuilds them.
        ("y = -2 ^ 2;", "uminus(mpower(2, 2))"),
        ("y = 2 ^ -2 ^ 2;", "mpower(mpower(2, uminus(2)), 2)"),
        ("y = a == b & c;", "and(eq(a, b), c)"),
        ("y = a.^b';", "ctranspose(power(a, b))"),
        ("y = -x';", "uminus(ctranspose(x))"),
        ("y = ~a == b;", "eq(not(a), b)"),
        ("y = a | b && c;", "andand(or(a, b), c)"),
        ("y = a - -b * c;", "minus(a, mtimes(uminus(b), c))"),
    ],
)
def test_operator_precedence(text: str, expected: str) -> None:
    program = lower_source(text="a = 1; b = 2; c = 3; x = 4;\n" + text)
    stmt = program.body[-1]
    assert isinstance(stmt, ir.Assign)
    assert _render(stmt.value) == expected


def _render(node: ir.Expr) -> str:
    match node:
        case ir.BinOp(op=op, left=left, right=right):
            return f"{op}({_render(left)}, {_render(right)})"
        case ir.UnOp(op=op, operand=operand):
            return f"{op}({_render(operand)})"
        case ir.Transpose(operand=operand, conjugate=conjugate):
            return f"{'ctranspose' if conjugate else 'transpose'}({_render(operand)})"
        case ir.Name(id=name):
            return name
        case ir.Num(text=text):
            return text
    raise AssertionError(node)


def test_unparsable_text_is_reported_not_dropped() -> None:
    result = scan_text(text="x = 1;\ny = (2 + ;\nz = 3;\n", name="t.m", index=INDEX)
    errors = [f for f in result.findings if f.tag == "parse:error"]
    assert errors and errors[0].loc.line == 2
    assert errors[0].unsupported


# --- scanning -----------------------------------------------------------------


def test_index_versus_call_is_resolved_by_assignment() -> None:
    program = lower_source(text="x = [1 2 3];\na = x(2);\nb = numel(x);\n")
    first, second = program.body[1], program.body[2]
    assert isinstance(first, ir.Assign) and isinstance(first.value, ir.Index)
    assert isinstance(second, ir.Assign) and isinstance(second.value, ir.Call)


def test_unknown_builtin_is_named_as_unsupported() -> None:
    result = scan_text(text="w = strsplit('a b', ' ');\n", name="t.m", index=INDEX)
    gaps = result.unsupported()
    assert [gap.tag for gap in gaps] == ["call:strsplit"]
    assert gaps[0].rule is None and gaps[0].loc.line == 1
    assert "call:strsplit" in result.report()


def test_struct_and_cell_access_are_named() -> None:
    tags = _tags("s.a = 1;\nc = {1, 2};\nv = c{1};\n")
    assert {"expr:field", "literal:cell", "expr:cell-index"} <= tags


def test_semantic_tags_use_inferred_facts() -> None:
    tags = _tags("x = [1 2 3];\nm = x > 1;\ny = x(m);\nx(end+1) = 4;\nz = [];\nz(3) = 1;\n")
    assert {"index:logical", "assign:grow"} <= tags
    # An in-range element assignment does not grow the array.
    assert "assign:grow" not in _tags("x = zeros(1, 3);\nx(2) = 5;\n")
    # A loop's later iterations must not make an undefined array look defined.
    assert "assign:grow" in _tags("for i = 1:3\n  A(i) = i;\nend\n")


def test_matlab_dialect_rejects_octave_only_syntax() -> None:
    text = "x = 1;  # hash comment\nx += 1;\n"
    result = scan_text(
        text=text, name="t.m", index=RuleIndex(rules=RULES, dialect=parse_dialect("matlab"))
    )
    tags = [f.tag for f in result.unsupported()]
    assert OCTAVE_ONLY_TAG in tags
    # The Octave-only rule for `+=` does not apply to the matlab dialect.
    assert "op:augassign" in tags
    assert "operators/augmented-assignment" in _scan(text)


def test_dialect_versions_filter_rules() -> None:
    data = {"dialects": ["matlab"], "since": {"matlab": "R2016b"}, "until": None}
    assert parse_dialect("matlab@R2023a").allows(data)
    assert not parse_dialect("matlab@R2016a").allows(data)
    assert not parse_dialect("octave").allows(data)
    with pytest.raises(DialectError):
        parse_dialect("fortran")
    with pytest.raises(DialectError):
        parse_dialect("matlab@2023")


# --- context --------------------------------------------------------------------


def test_context_contains_exactly_the_needed_rules(tmp_path: Path) -> None:
    source = tmp_path / "t.m"
    source.write_text("x = linspace(0, 1, 5);\ny = strsplit('a b', ' ');\n", encoding="utf-8")
    result = scan_file(path=source, rules=RULES, dialect=Dialect())
    text = build_context(result=result)
    needed = {rule.id for rule in result.needed_rules()}
    assert "builtins/linspace" in needed
    for rule in RULES:
        assert (f"## {rule.id} (" in text) == (rule.id in needed), rule.id
    assert "id: builtins/linspace" in text  # the rule.yaml itself
    assert "x = linspace(0, 1, 5);" in text  # a case, .m side
    assert "np.linspace(0, 1, 5)" in text  # and its Python twin
    assert "call:strsplit at 2:5" in text


def test_scan_and_context_commands(capsys: pytest.CaptureFixture[str]) -> None:
    case = RULES_DIR / "builtins" / "find" / "cases" / "basic.m"
    assert main(["scan", str(case)]) == EXIT_OK
    assert "builtins/find" in capsys.readouterr().out
    assert main(["scan", "--json", str(case)]) == EXIT_OK
    assert '"id": "builtins/find"' in capsys.readouterr().out
    assert main(["context", str(case)]) == EXIT_OK
    assert "## builtins/find" in capsys.readouterr().out
