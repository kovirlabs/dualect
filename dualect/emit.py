# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Emitting Python/NumPy from the IR (docs/design.md section 5, steps 3-5).

The emitter walks the IR statement by statement. For each statement it first
*plans*: every construct tag the statement uses must map to a ``mechanical``
rule. If one does not, the statement becomes a **hole**: a ``TODO(dualect)``
comment carrying the original source and a ``raise NotImplementedError``, so
the output can never run past a part that was not translated. Otherwise the
statement is translated, with each construct handed to its rule's
``translate.py`` (loaded by :mod:`dualect.translators`). A translator that
meets a form it cannot translate correctly raises :class:`Decline`, which also
turns the statement into a hole, reported as ``guided``.

What lives here, rather than in a rule, is what every rule shares: the
dispatch, Python's statement layout, copy-on-write semantics (a NumPy
assignment ``y = x`` aliases, an ``.m`` assignment copies), and the analysis of
which variables only ever hold positions and can therefore be stored 0-based.

Python code is built as :mod:`ast` nodes and printed with :func:`ast.unparse`,
which inserts the parentheses precedence needs; :mod:`dualect.pyformat` then
formats the file with ruff.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import NoReturn

from dualect import ir
from dualect.constructs import primary_tag
from dualect.infer import Analysis, Info, Kind
from dualect.rulebook import Rule, Status
from dualect.scan import Finding, ScanResult

#: Indentation unit of the emitted code (ruff reformats anyway).
INDENT = "    "

#: A float literal that Python spells the same way.
_PY_FLOAT = re.compile(r"(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


class Decline(Exception):
    """A translator cannot translate this form correctly; the statement is guided.

    :param reason: what is missing, for the report.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class _Unsupported(Exception):
    """A construct with no mechanical rule was reached during emission."""

    def __init__(self, tag: str, rule: Rule | None) -> None:
        self.tag = tag
        self.rule = rule
        super().__init__(tag)


@dataclass(frozen=True)
class Line:
    """One emitted line of Python.

    :param indent: nesting depth.
    :param text: the code or comment, without indentation.
    """

    indent: int
    text: str

    def render(self) -> str:
        """The line with its indentation."""
        return f"{INDENT * self.indent}{self.text}" if self.text else ""


@dataclass
class StmtReport:
    """How one statement was handled.

    :param loc: where the statement is.
    :param kind: the statement kind (``Assign``, ``For``, ...).
    :param rules: ids of the rules its constructs needed.
    :param status: ``mechanical`` (translated by rules), ``guided`` (left for an
        LLM: a guided rule, or a translator declined) or ``unsupported`` (a
        construct with no rule, or an ``unsupported`` rule).
    :param reason: why it was not mechanical.
    """

    loc: ir.Loc
    kind: str
    rules: list[str]
    status: Status
    reason: str = ""


@dataclass
class ScopeFacts:
    """Per-function (or script) facts the emitter needs.

    :param index_vars: variables stored 0-based because they only hold positions.
    :param mutated: variables modified in place (indexed assignment).
    :param variables: every variable of the scope.
    :param outputs: the function's output names (empty for the script).
    :param is_script: ``True`` for the script's top level, whose variables are
        the compared outputs.
    """

    index_vars: set[str] = field(default_factory=set)
    mutated: set[str] = field(default_factory=set)
    variables: set[str] = field(default_factory=set)
    outputs: list[str] = field(default_factory=list)
    is_script: bool = False


class Context:
    """What a rule translator works with while translating one statement.

    Translators call back into the context to translate sub-expressions
    (:meth:`expr`), read inferred facts (:meth:`info`), refer to NumPy
    (:meth:`np`), record assumptions (:meth:`assume`) and decline
    (:meth:`decline`).
    """

    def __init__(
        self,
        *,
        scan: ScanResult,
        translator_for: Callable[[str], ModuleType | None],
        rule_for: Callable[[str], Rule | None],
        rule_by_id: Callable[[str], Rule | None],
    ) -> None:
        self.scan = scan
        self.analysis: Analysis = scan.analysis
        self._translator_for = translator_for
        self._rule_for = rule_for
        self._rule_by_id = rule_by_id
        self.imports: set[str] = set()
        self.assumptions: list[str] = []
        self.scope = ScopeFacts()
        self.functions: dict[str, ir.FunctionDef] = {
            s.name: s for s in scan.program.body if isinstance(s, ir.FunctionDef)
        }
        self.loop_stack: list[ir.Stmt] = []
        self.indent = 0
        self.reports: list[StmtReport] = []
        self.directives: set[str] = set()
        self.source_lines = scan.program_text.splitlines() if scan.program_text else []

    # --- facts -------------------------------------------------------------

    def info(self, node: ir.Expr) -> Info:
        """Inferred facts about an expression."""
        return self.analysis.info(node)

    def index_directive(self, name: str) -> list[Line]:
        """A ``# dualect: index`` line the first time a script variable is stored 0-based.

        Script variables are the outputs verification compares; the directive
        tells it to compare this one as a 0-based position.
        """
        if not self.scope.is_script or name in self.directives:
            return []
        self.directives.add(name)
        return [self.line(f"# dualect: index {name}")]

    def is_index_var(self, name: str) -> bool:
        """Whether ``name`` is stored 0-based in the output (see :func:`index_variables`)."""
        return name in self.scope.index_vars

    # --- building Python -----------------------------------------------------

    def np(self, attr: str) -> ast.expr:
        """``np.<attr>``, recording the import."""
        self.imports.add("import numpy as np")
        node: ast.expr = ast.Name(id="np")
        for part in attr.split("."):
            node = ast.Attribute(value=node, attr=part)
        return node

    @staticmethod
    def call(func: ast.expr, *args: ast.expr, **keywords: ast.expr) -> ast.expr:
        """A call node."""
        return ast.Call(
            func=func,
            args=list(args),
            keywords=[ast.keyword(arg=key, value=value) for key, value in keywords.items()],
        )

    @staticmethod
    def method(obj: ast.expr, name: str, *args: ast.expr, **keywords: ast.expr) -> ast.expr:
        """``obj.name(args)``."""
        return Context.call(ast.Attribute(value=obj, attr=name), *args, **keywords)

    @staticmethod
    def const(value: int | float | complex | str | bool | None) -> ast.expr:
        """A literal; negative numbers become a unary minus, as Python parses them."""
        if isinstance(value, int | float) and not isinstance(value, bool) and value < 0:
            return ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=-value))
        return ast.Constant(value=value)

    def assume(self, text: str) -> None:
        """Record an assumption (docs/design.md section 5.2) for the current statement."""
        if text not in self.assumptions:
            self.assumptions.append(text)

    @staticmethod
    def decline(reason: str) -> NoReturn:
        """Give up on this statement; it becomes a guided hole."""
        raise Decline(reason)

    def has_mechanical(self, rule_id: str) -> bool:
        """Whether a rule applies to this dialect, is ``mechanical`` and has a translator."""
        rule = self._rule_by_id(rule_id)
        return (
            rule is not None
            and rule.status is Status.MECHANICAL
            and self._translator_for(rule_id) is not None
        )

    def module(self, rule_id: str) -> ModuleType:
        """Another rule's translator, for its helpers; only a ``mechanical`` rule's.

        :raises _Unsupported: when that rule is not mechanical, so the statement
            becomes a hole reported against that rule.
        """
        rule = self._rule_by_id(rule_id)
        module = self._translator_for(rule_id)
        if rule is None or rule.status is not Status.MECHANICAL or module is None:
            raise _Unsupported(rule.match[0] if rule is not None and rule.match else rule_id, rule)
        return module

    def position_outputs(self, call: ir.Call, nargout: int) -> set[int]:
        """Outputs of a built-in call that hold 1-based positions (from its rule)."""
        rule = self._rule_for(f"call:{call.func}")
        if rule is None or rule.status is not Status.MECHANICAL:
            return set()
        module = self._translator_for(rule.id)
        hook = getattr(module, "position_outputs", None)
        return set(hook(call, nargout)) if hook is not None else set()

    def enter_script(self) -> None:
        """Compute the script-level scope facts."""
        body = [s for s in self.scan.program.body if not isinstance(s, ir.FunctionDef)]
        self.scope = scope_facts(
            body=body,
            params=[],
            outputs=[],
            position_outputs=self.position_outputs,
            analysis=self.analysis,
        )
        self.scope.is_script = True

    @contextmanager
    def function_scope(self, fdef: ir.FunctionDef) -> Iterator[ScopeFacts]:
        """Translate inside a function: its own variables, copies and index variables."""
        saved = self.scope
        self.scope = scope_facts(
            body=fdef.body,
            params=fdef.params,
            outputs=fdef.outputs,
            position_outputs=self.position_outputs,
            analysis=self.analysis,
        )
        try:
            yield self.scope
        finally:
            self.scope = saved

    # --- dispatch ------------------------------------------------------------

    def expr(self, node: ir.Expr) -> ast.expr:
        """Translate an expression (in value context)."""
        match node:
            case ir.Num(value=value, text=text):
                if isinstance(value, float) and _PY_FLOAT.fullmatch(text):
                    # Keep the source spelling (200e9, not 200000000000.0); the
                    # printer writes a Name's id verbatim.
                    return ast.Name(id=text)
                return self.const(value)
            case ir.Name(id=name):
                if self.is_index_var(name):
                    # Stored 0-based; its source value is one more.
                    return ast.BinOp(left=ast.Name(id=name), op=ast.Add(), right=ast.Constant(1))
                return ast.Name(id=name)
            case ir.Colon():
                return ast.Slice()
        tag = primary_tag(node, self.analysis)
        if tag is None:
            raise Decline(f"no translation for {type(node).__name__}")
        translator = self._mechanical(tag)
        result = translator.translate(node, self)
        if not isinstance(result, ast.expr):
            raise Decline(f"{tag} did not give an expression")
        return result

    def arg(self, node: ir.Expr) -> ast.expr:
        """An argument to a NumPy function: a vector literal stays a plain list."""
        if (
            isinstance(node, ir.Matrix)
            and len(node.rows) == 1
            and len(node.rows[0]) > 1
            and all(self.info(item).is_scalar for item in node.rows[0])
        ):
            return ast.List(elts=[self.expr(item) for item in node.rows[0]])
        return self.expr(node)

    def _mechanical(self, tag: str) -> ModuleType:
        rule = self._rule_for(tag)
        if rule is None or rule.status is not Status.MECHANICAL:
            raise _Unsupported(tag, rule)
        module = self._translator_for(rule.id)
        if module is None:
            raise _Unsupported(tag, rule)
        return module

    def stmt_translator(self, tag: str) -> ModuleType:
        """The translator for a statement-level construct."""
        return self._mechanical(tag)

    # --- statements ----------------------------------------------------------

    def block(self, body: list[ir.Stmt]) -> list[Line]:
        """Translate a nested body one level deeper; never empty."""
        self.indent += 1
        try:
            lines = self.statements(body)
        finally:
            self.indent -= 1
        if not any(line.text and not line.text.startswith("#") for line in lines):
            lines.append(Line(self.indent + 1, "pass"))
        return lines

    @contextmanager
    def loop(self, stmt: ir.Stmt) -> Iterator[None]:
        """Track the enclosing loop (``continue`` in ``do ... until`` needs to know)."""
        self.loop_stack.append(stmt)
        try:
            yield
        finally:
            self.loop_stack.pop()

    def statements(self, body: list[ir.Stmt]) -> list[Line]:
        """Translate statements at the current indentation."""
        lines: list[Line] = []
        previous_end: int | None = None
        for stmt in body:
            if previous_end is not None and stmt.loc.line > previous_end + 1 and lines:
                lines.append(Line(self.indent, ""))  # keep the source's paragraphs
            lines.extend(self.statement(stmt))
            previous_end = stmt.loc.end_line
        return lines

    def statement(self, stmt: ir.Stmt) -> list[Line]:
        """Translate one statement, or emit a hole for it."""
        findings = self.scan.uses_by_stmt.get(id(stmt), [])
        rules = sorted({f.rule.id for f in findings if f.rule is not None})
        report = StmtReport(
            loc=stmt.loc, kind=type(stmt).__name__, rules=rules, status=Status.MECHANICAL
        )
        self.reports.append(report)
        blocking = [f for f in findings if f.rule is None or f.rule.status is not Status.MECHANICAL]
        if blocking:
            worst = _worst(blocking)
            report.status = worst.status
            report.reason = _describe(blocking)
            return self.hole(stmt, report)
        saved_imports = set(self.imports)
        saved_directives = set(self.directives)
        self.assumptions = []
        mark = len(self.reports)
        try:
            lines = self.translate_statement(stmt)
        except Decline as exc:
            self.imports = saved_imports
            self.directives = saved_directives
            del self.reports[mark:]
            report.status = Status.GUIDED
            report.reason = f"translator declined: {exc.reason}"
            return self.hole(stmt, report)
        except _Unsupported as exc:
            self.imports = saved_imports
            self.directives = saved_directives
            del self.reports[mark:]
            report.status = Status.UNSUPPORTED if exc.rule is None else exc.rule.status
            label = exc.rule.id if exc.rule is not None else "no rule"
            report.reason = f"{exc.tag} ({label})"
            return self.hole(stmt, report)
        notes = [*self.assumptions]
        if stmt.comment:
            notes.insert(0, stmt.comment)
        if notes and lines:
            # Compound statements carry the note on their header line.
            compound = isinstance(stmt, ir.If | ir.For | ir.While | ir.Switch | ir.DoUntil)
            code = [k for k, line in enumerate(lines) if not line.text.startswith("#")]
            if code:
                index = code[0] if compound or isinstance(stmt, ir.FunctionDef) else code[-1]
                parts = [stmt.comment] if stmt.comment else []
                if self.assumptions:
                    parts.append(f"dualect: assumed {'; '.join(self.assumptions)}")
                line = lines[index]
                lines[index] = Line(line.indent, f"{line.text}  # {'; '.join(parts)}")
        return lines

    def translate_statement(self, stmt: ir.Stmt) -> list[Line]:
        """Translate a statement whose constructs are all mechanical."""
        match stmt:
            case ir.Assign():
                return self.assignment(stmt)
            case ir.ExprStmt(value=value):
                if "syntax:script-marker" in stmt.syntax:
                    return []
                return [Line(self.indent, ast.unparse(self.expr(value)))]
        primary = primary_tag(stmt, self.analysis)
        if primary is None:
            raise Decline(f"no translation for {type(stmt).__name__}")
        return self._lines(self.stmt_translator(primary).translate(stmt, self))

    def _lines(self, result: object) -> list[Line]:
        if not isinstance(result, list):
            raise Decline("translator did not give statements")
        return result

    def line(self, text: str, *, extra: int = 0) -> Line:
        """A line at the current indentation (plus ``extra`` levels)."""
        return Line(self.indent + extra, text)

    def assignment(self, stmt: ir.Assign) -> list[Line]:
        """Plain and indexed assignment; multi-output calls go to their rule."""
        targets = stmt.targets
        value = stmt.value
        if len(targets) > 1 or isinstance(targets[0], ir.Ignore):
            if isinstance(value, ir.Call) and not value.local and not value.handle:
                module = self.stmt_translator(f"call:{value.func}")
                multi = getattr(module, "translate_multi", None)
                if multi is None:
                    raise Decline(f"{value.func} with several outputs")
                return self._lines(multi(stmt, self))
            return self._lines(self.stmt_translator("assign:multi").translate(stmt, self))
        target = targets[0]
        if isinstance(target, ir.Index):
            module = self.stmt_translator("index:paren")
            return self._lines(module.translate_store(stmt, self))
        if not isinstance(target, ir.Name):
            raise _Unsupported(primary_tag(target, self.analysis) or "parse:error", None)
        name = target.id
        lines: list[Line] = []
        if self.is_index_var(name):
            lines.extend(self.index_directive(name))
            rhs = self.position_value(value)
            lines.append(self.line(f"{name} = {ast.unparse(rhs)}"))
            return lines
        rhs = self.expr(value)
        augmented = self._augmented(name, value)
        if augmented is not None:
            return [self.line(augmented)]
        if self._needs_copy(name, value, rhs):
            rhs = self.method(rhs, "copy")
        lines.append(self.line(f"{name} = {ast.unparse(rhs)}"))
        return lines

    def position_value(self, value: ir.Expr) -> ast.expr:
        """The 0-based form of a value assigned to an index variable."""
        if isinstance(value, ir.Num) and isinstance(value.value, int):
            return self.const(value.value - 1)
        if isinstance(value, ir.Matrix):
            items = [item for row in value.rows for item in row]
            if all(isinstance(i, ir.Num) and isinstance(i.value, int) for i in items):
                shifted = [ast.Constant(i.value - 1) for i in items if isinstance(i, ir.Num)]
                return self.call(self.np("array"), ast.List(elts=list(shifted)))
        if isinstance(value, ir.Call):
            module = self.stmt_translator(f"call:{value.func}")
            positions = getattr(module, "translate_positions", None)
            if positions is not None:
                result = positions(value, self)
                if isinstance(result, ast.expr):
                    return result
        raise Decline("cannot store this value 0-based")

    def _augmented(self, name: str, value: ir.Expr) -> str | None:
        """``x = x + e`` on a known scalar reads best as ``x += e``."""
        if not (
            isinstance(value, ir.BinOp)
            and value.op in ("plus", "minus")
            and isinstance(value.left, ir.Name)
            and value.left.id == name
        ):
            return None
        before = self.info(value.left)
        if not (before.is_scalar and before.elem.value in ("real", "logical")):
            return None
        # Only simple steps read better augmented (`count += 1`, `s += k`).
        if not isinstance(value.right, ir.Num | ir.Name):
            return None
        op = "+=" if value.op == "plus" else "-="
        return f"{name} {op} {ast.unparse(self.expr(value.right))}"

    def _needs_copy(self, name: str, value: ir.Expr, rhs: ast.expr) -> bool:
        """Whether ``name = value`` must copy to keep ``.m`` copy semantics.

        NumPy assignment binds the same array, and slices and transposes are
        views; the ``.m`` side always has a private copy. A copy is needed when
        either side is later modified in place and the right-hand side may share
        memory (a variable, an index or slice, a transpose).
        """
        info = self.info(value)
        if not (info.is_array or info.kind is Kind.UNKNOWN):
            return False
        shares = isinstance(rhs, ast.Name | ast.Subscript | ast.Attribute)
        if not shares:
            return False
        source = value.id if isinstance(value, ir.Name) else None
        return name in self.scope.mutated or (source is not None and source in self.scope.mutated)

    # --- holes ---------------------------------------------------------------

    def hole(self, stmt: ir.Stmt, report: StmtReport) -> list[Line]:
        """Placeholder for a statement that was not translated."""
        lines = [
            self.line(f"# TODO(dualect): {report.status.value}, {report.reason}"),
            self.line(f"# line {stmt.loc.line} of the source:"),
        ]
        for text in self.source_text(stmt.loc):
            lines.append(self.line(f"#     {text}"))
        message = f"dualect: line {stmt.loc.line} was not translated ({report.status.value})"
        lines.append(self.line(f"raise NotImplementedError({message!r})"))
        # Nested statements of an untranslated compound are not reported again.
        return lines

    def source_text(self, loc: ir.Loc) -> list[str]:
        """The original source lines of a span."""
        if not self.source_lines:
            return []
        first = max(loc.line, 1)
        last = min(max(loc.end_line, first), len(self.source_lines))
        return [self.source_lines[n - 1].rstrip() for n in range(first, last + 1)]


def _worst(findings: list[Finding]) -> Finding:
    for finding in findings:
        if finding.status is Status.UNSUPPORTED:
            return finding
    return findings[0]


def _describe(findings: list[Finding]) -> str:
    parts = []
    for finding in findings:
        label = finding.rule.id if finding.rule is not None else "no rule"
        text = f"{finding.tag} ({label})"
        if text not in parts:
            parts.append(text)
    return "needs " + ", ".join(parts)


# --- scope analyses --------------------------------------------------------------


def scope_facts(
    *,
    body: list[ir.Stmt],
    params: list[str],
    outputs: list[str],
    position_outputs: Callable[[ir.Call, int], set[int]],
    analysis: Analysis,
) -> ScopeFacts:
    """Compute :class:`ScopeFacts` for one function body (or the script).

    :param body: the statements of the scope (nested functions excluded).
    :param params: parameter names (empty for the script).
    :param outputs: output names (empty for the script).
    :param position_outputs: for a built-in call and an output count, the
        output positions that hold 1-based positions (``find``, ``max``'s second).
    :param analysis: inference results.
    """
    facts = ScopeFacts(outputs=list(outputs), variables={*params, *outputs})
    stmts = _scope_statements(body)
    for stmt in stmts:
        if isinstance(stmt, ir.Assign):
            # Deleting elements (`x(m) = []`) rebinds the name; it does not
            # modify the array in place.
            deleting = isinstance(stmt.value, ir.Matrix) and not stmt.value.rows
            for target in stmt.targets:
                if isinstance(target, ir.Name):
                    facts.variables.add(target.id)
                elif isinstance(target, ir.Index) and isinstance(target.target, ir.Name):
                    facts.variables.add(target.target.id)
                    if not deleting:
                        facts.mutated.add(target.target.id)
        elif isinstance(stmt, ir.AugAssign):
            if isinstance(stmt.target, ir.Index) and isinstance(stmt.target.target, ir.Name):
                facts.mutated.add(stmt.target.target.id)
        elif isinstance(stmt, ir.For):
            facts.variables.add(stmt.var.id)
    facts.index_vars = index_variables(
        stmts=stmts,
        excluded={*params, *outputs},
        position_outputs=position_outputs,
        analysis=analysis,
    )
    return facts


def _scope_statements(body: list[ir.Stmt]) -> list[ir.Stmt]:
    """Every statement of a scope, without descending into nested functions."""
    found: list[ir.Stmt] = []
    for stmt in body:
        if isinstance(stmt, ir.FunctionDef):
            continue
        found.append(stmt)
        for inner in ir.stmt_bodies(stmt):
            found.extend(_scope_statements(inner))
    return found


def index_variables(
    *,
    stmts: list[ir.Stmt],
    excluded: set[str],
    position_outputs: Callable[[ir.Call, int], set[int]],
    analysis: Analysis,
) -> set[str]:
    """Variables that can be stored 0-based: they only ever hold positions.

    A variable qualifies when every assignment to it produces positions (an
    integer literal, the position output of a built-in such as ``find``, or a
    ``for`` loop over an integer range) and every read of it is inside a
    subscript, where a 1-based position would be shifted anyway. Storing it
    0-based then removes every ``- 1`` at its uses: ``i = 2; x(i + 1)`` becomes
    ``i = 1; x[i + 1]``. A variable from ``find`` qualifies even when unused,
    since NumPy users expect positions to be 0-based. Function parameters and
    outputs never qualify: their values cross the function boundary.

    :param stmts: every statement of the scope (flattened).
    :param excluded: names that never qualify.
    :param position_outputs: see :func:`scope_facts`.
    :param analysis: inference results.
    """
    sources: dict[str, set[str]] = {}
    for stmt in stmts:
        if isinstance(stmt, ir.Assign):
            names = [t.id if isinstance(t, ir.Name) else None for t in stmt.targets]
            value = stmt.value
            positions: set[int] = set()
            if isinstance(value, ir.Call) and not value.local and not value.handle:
                positions = position_outputs(value, len(stmt.targets))
            for k, name in enumerate(names):
                if name is None:
                    continue
                if k in positions:
                    kind = "position"
                elif len(names) == 1 and _is_position_literal(value):
                    kind = "literal"
                else:
                    kind = "other"
                sources.setdefault(name, set()).add(kind)
        elif isinstance(stmt, ir.AugAssign) and isinstance(stmt.target, ir.Name):
            sources.setdefault(stmt.target.id, set()).add("other")
        elif isinstance(stmt, ir.For):
            kind = "loop" if _is_position_range(stmt.iterable, analysis) else "other"
            sources.setdefault(stmt.var.id, set()).add(kind)

    reads: dict[str, list[bool]] = {}
    for stmt in stmts:
        for root in _read_roots(stmt):
            _collect_reads(root, reads, in_subscript=False)

    chosen: set[str] = set()
    for name, kinds in sources.items():
        if name in excluded or "other" in kinds:
            continue
        uses = reads.get(name, [])
        if not all(uses):
            continue
        if kinds == {"position"} or uses:
            chosen.add(name)
    return chosen


def _is_position_literal(value: ir.Expr) -> bool:
    """An integer literal >= 1, or a vector literal of them."""
    if isinstance(value, ir.Num):
        return isinstance(value.value, int) and value.value >= 1
    if isinstance(value, ir.Matrix) and value.rows:
        items = [item for row in value.rows for item in row]
        is_vector = len(value.rows) == 1 or all(len(row) == 1 for row in value.rows)
        return is_vector and all(_is_position_literal(item) for item in items)
    return False


def _is_position_range(iterable: ir.Expr, analysis: Analysis) -> bool:
    """A ``for`` range of integer positions with step 1: ``1:n``, ``k:m``."""
    if not isinstance(iterable, ir.Range) or iterable.step is not None:
        return False
    start = analysis.info(iterable.start).value
    return isinstance(start, int | float) and float(start).is_integer() and start >= 1


def _read_roots(stmt: ir.Stmt) -> list[ir.Expr]:
    """Expressions a statement reads (assignment targets' subscripts included)."""
    roots: list[ir.Expr] = []
    for expr in ir.stmt_exprs(stmt):
        if isinstance(stmt, ir.Assign | ir.AugAssign) and isinstance(expr, ir.Name):
            if isinstance(stmt, ir.AugAssign) and expr is stmt.target:
                roots.append(expr)
            elif isinstance(stmt, ir.Assign) and expr in stmt.targets:
                continue
            else:
                roots.append(expr)
            continue
        if isinstance(stmt, ir.For) and expr is stmt.var:
            continue
        roots.append(expr)
    return roots


def _collect_reads(node: ir.Expr, reads: dict[str, list[bool]], *, in_subscript: bool) -> None:
    """Record, for each name read, whether the read is position-only."""
    match node:
        case ir.Name(id=name):
            reads.setdefault(name, []).append(in_subscript)
        case ir.Index(target=target, args=args):
            if isinstance(target, ir.Name):
                # Indexing a position array keeps its elements positions only when
                # the result is itself used as a subscript.
                reads.setdefault(target.id, []).append(in_subscript)
            else:
                _collect_reads(target, reads, in_subscript=False)
            for arg in args:
                _collect_subscript(arg, reads)
        case ir.Lambda(body=body):
            _collect_reads(body, reads, in_subscript=False)
        case _:
            for child in ir.child_exprs(node):
                _collect_reads(child, reads, in_subscript=False)


def _collect_subscript(node: ir.Expr, reads: dict[str, list[bool]]) -> None:
    """Reads inside a subscript: linear uses (``i``, ``i + 1``, ``a:i``) are positions."""
    match node:
        case ir.Name() | ir.Index():
            _collect_reads(node, reads, in_subscript=True)
        case ir.BinOp(op="plus", left=left, right=right):
            _collect_subscript(left, reads)
            _collect_subscript(right, reads)
        case ir.BinOp(op="minus", left=left, right=right):
            _collect_subscript(left, reads)
            _collect_reads(right, reads, in_subscript=False)
        case ir.Range(start=start, stop=stop, step=step):
            _collect_subscript(start, reads)
            _collect_subscript(stop, reads)
            if step is not None:
                _collect_reads(step, reads, in_subscript=False)
        case _:
            _collect_reads(node, reads, in_subscript=False)


# --- helpers for translators ---------------------------------------------------


def int_constant(node: ast.expr) -> int | None:
    """The value of an integer literal (including ``-k``), else ``None``."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
    ):
        return -node.operand.value
    return None


def offset(node: ast.expr, delta: int) -> ast.expr:
    """``node + delta`` with integer constants folded: ``(i + 1) - 1`` is ``i``.

    :param node: a Python expression.
    :param delta: the integer to add.
    """
    if delta == 0:
        return node
    value = int_constant(node)
    if value is not None:
        return Context.const(value + delta)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub):
        right = int_constant(node.right)
        if right is not None:
            carried = right if isinstance(node.op, ast.Add) else -right
            return offset(node.left, carried + delta)
    if delta > 0:
        return ast.BinOp(left=node, op=ast.Add(), right=ast.Constant(delta))
    return ast.BinOp(left=node, op=ast.Sub(), right=ast.Constant(-delta))


def render(lines: list[Line]) -> str:
    """Join emitted lines into source text."""
    return "\n".join(line.render() for line in lines) + "\n"
