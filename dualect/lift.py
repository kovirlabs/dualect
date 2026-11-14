# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Lifting Python source into the IR (the Python -> ``.m`` direction).

This is the mirror of :mod:`dualect.lower` plus :mod:`dualect.emit`: it parses
Python with the stdlib :mod:`ast`, infers facts (:mod:`dualect.pyinfer`), tags
every statement's constructs (:mod:`dualect.pyconstructs`), plans each
statement and lifts it into the shared IR with ``.m`` semantics, which
:mod:`dualect.memit` prints.

The rule knowledge is the same rule folders as the other direction. A rule
covers Python constructs through ``from_python: {status, match}`` in its
``rule.yaml``, and its ``translate.py`` defines ``lift(node, ctx)`` (and
optionally ``lift_store``, ``lift_multi``) that turns a Python node into IR
through the :class:`LiftContext`. Planning is per statement, as in the other
direction: every construct must map to a rule that is ``mechanical`` *from
Python*, else the statement becomes a hole (a ``TODO(dualect)`` comment and an
``error(...)`` call, so the ``.m`` output can never run past it); a lift that
meets a form it cannot translate correctly raises :class:`~dualect.emit.Decline`.

What lives here rather than in a rule is what every rule shares: names,
numbers and plain assignment, comments, and NumPy's *aliasing*: ``y = x``,
a slice and ``.T`` share memory in NumPy, so writing into one changes the
other, while the ``.m`` side always copies. A statement whose meaning depends
on that sharing is declined, never translated as if it did not exist.
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import NoReturn

from dualect import ir
from dualect.emit import Decline, StmtReport
from dualect.errors import DualectError, RuleError
from dualect.memit import HOLE
from dualect.pyconstructs import PyUse, Tagger, loc_of
from dualect.pyinfer import PyAnalysis, PyInfo, PyKind, analyze_python
from dualect.rulebook import FROM_PYTHON, Rule, Status
from dualect.scan import Finding

#: Comment directives the verify harness reads; they are never copied over,
#: because their meaning (a 0-based position) belongs to the source pairing.
_DIRECTIVE_PREFIX = "dualect:"


class PyRuleIndex:
    """Maps ``py:`` construct tags to the rule covering them (``from_python.match``)."""

    def __init__(self, *, rules: list[Rule]) -> None:
        """Index the rules' Python tags.

        :param rules: all loaded rules.
        :raises RuleError: when two rules claim the same Python tag.
        """
        self.by_tag: dict[str, Rule] = {}
        self.by_id: dict[str, Rule] = {rule.id: rule for rule in rules}
        for rule in rules:
            for tag in rule.python_match:
                other = self.by_tag.get(tag)
                if other is not None:
                    raise RuleError(f"{rule.id} and {other.id} both match {tag!r} from Python")
                self.by_tag[tag] = rule

    def rule_for(self, tag: str) -> Rule | None:
        """The rule covering ``tag``, or ``None``."""
        return self.by_tag.get(tag)


@dataclass
class ScopeInfo:
    """Names of one Python scope (the module, or a ``def``).

    :param variables: names assigned in the scope (parameters included).
    :param mutated: names written into in place (``x[i] = v``, ``x += v`` on
        an array, in-place methods).
    :param params: the parameters of a ``def``.
    """

    variables: set[str] = field(default_factory=set)
    mutated: set[str] = field(default_factory=set)
    params: list[str] = field(default_factory=list)


class LiftContext:
    """What a rule's ``lift`` works with while lifting one statement."""

    def __init__(
        self,
        *,
        module: ast.Module,
        text: str,
        analysis: PyAnalysis,
        index: PyRuleIndex,
        translator_for: Callable[[str], ModuleType | None],
    ) -> None:
        self.module = module
        self.analysis = analysis
        self.index = index
        self._translator_for = translator_for
        self.functions: dict[str, ast.FunctionDef] = {
            s.name: s for s in module.body if isinstance(s, ast.FunctionDef)
        }
        self.tagger = Tagger(analysis=analysis, functions=set(self.functions))
        self.source_lines = text.splitlines()
        self.comments, self.standalone = _comments(text)
        self.reports: list[StmtReport] = []
        self.findings: list[Finding] = []
        self.uses_by_stmt: dict[int, list[Finding]] = {}
        self.assumptions: list[str] = []
        self.scope = scope_info(module.body, params=[])
        self.module_scope = self.scope
        self.in_condition = False
        self.function: ast.FunctionDef | None = None
        self.outputs: list[str] = []
        self.loop_depth = 0
        #: File-level verification directives the ``.m`` output needs
        #: (``stdout`` when it prints, ``shape r`` for random values).
        self.file_directives: list[str] = []
        self._used_comments: set[int] = set()
        self._plan(module.body)

    # --- planning -----------------------------------------------------------

    def _plan(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            uses = self.tagger.statement_uses(stmt)
            if stmt.lineno in self.comments and not _is_directive(self.comments[stmt.lineno]):
                uses.append(PyUse("py:comment", loc_of(stmt)))
            found = [
                Finding(
                    tag=use.tag,
                    loc=use.loc,
                    rule=self.index.rule_for(use.tag),
                    text=self.excerpt(use.loc) if self.index.rule_for(use.tag) is None else "",
                    direction=FROM_PYTHON,
                )
                for use in _dedupe(uses)
            ]
            self.uses_by_stmt[id(stmt)] = found
            self.findings.extend(found)
            for inner in _bodies(stmt):
                self._plan(inner)

    def excerpt(self, loc: ir.Loc) -> str:
        """The first source line of a span, trimmed, for reports."""
        if not 1 <= loc.line <= len(self.source_lines):
            return ""
        line = self.source_lines[loc.line - 1]
        end = loc.end_col - 1 if loc.end_line == loc.line else len(line)
        snippet = line[max(loc.col - 1, 0) : end].strip() or line.strip()
        return snippet if len(snippet) <= 48 else snippet[:45] + "..."

    # --- facts and helpers for rules ----------------------------------------

    def info(self, node: ast.AST) -> PyInfo:
        """Inferred facts about a Python expression."""
        return self.analysis.info(node)

    @staticmethod
    def loc(node: ast.AST) -> ir.Loc:
        """The IR location of a Python node."""
        return loc_of(node)

    def file_directive(self, text: str) -> None:
        """Ask for a ``% dualect: <text>`` line at the top of the ``.m`` output."""
        if text not in self.file_directives:
            self.file_directives.append(text)

    def assume(self, text: str) -> None:
        """Record an assumption; it is printed as ``% dualect: assumed ...``."""
        if text not in self.assumptions:
            self.assumptions.append(text)

    @staticmethod
    def decline(reason: str) -> NoReturn:
        """Give up on this statement; it becomes a guided hole."""
        raise Decline(reason)

    def module_for(self, rule_id: str) -> ModuleType:
        """Another rule's translator, for its helpers; only when it is mechanical from Python."""
        rule = self.index.by_id.get(rule_id)
        module = self._translator_for(rule_id)
        if rule is None or rule.python_status is not Status.MECHANICAL or module is None:
            tag = rule.python_match[0] if rule is not None and rule.python_match else rule_id
            raise _Unsupported(tag, rule)
        return module

    def tag(self, node: ast.AST) -> str | None:
        """The primary construct tag of a Python expression."""
        return self.tagger.primary(node)

    @property
    def numpy(self) -> str | None:
        """The name numpy is imported as."""
        return self.analysis.numpy_alias

    # --- IR constructors -------------------------------------------------------

    @staticmethod
    def num(value: int | float, *, loc: ir.Loc = ir.NO_LOC) -> ir.Expr:
        """A numeric literal (a negative one as a unary minus)."""
        if value < 0:
            return ir.UnOp(loc=loc, op="uminus", operand=LiftContext.num(-value, loc=loc))
        text = str(value) if isinstance(value, int) else repr(value)
        return ir.Num(loc=loc, text=text, value=value)

    def call(self, func: str, *args: ir.Expr, loc: ir.Loc = ir.NO_LOC) -> ir.Call:
        """A call of a built-in.

        A variable of the same name would shadow the function on the ``.m``
        side (``max = 3; max(x)`` indexes the variable), so that is declined.
        """
        if func in self.scope.variables or func in self.module_scope.variables:
            self.decline(f"a variable named {func} would hide the function {func}()")
        return ir.Call(loc=loc, func=func, args=list(args))

    def condition(self, node: ast.expr) -> ir.Expr:
        """Lift an ``if``/``while`` test: only its truth matters there.

        ``a and b`` is a value in Python (one of the operands); in a condition
        only its truth is used, which is what ``&&`` gives.
        """
        saved = self.in_condition
        self.in_condition = True
        try:
            return self.expr(node)
        finally:
            self.in_condition = saved

    @staticmethod
    def binop(op: str, left: ir.Expr, right: ir.Expr, *, loc: ir.Loc = ir.NO_LOC) -> ir.Expr:
        """A binary operation."""
        return ir.BinOp(loc=loc, op=op, left=left, right=right)

    @staticmethod
    def transpose(node: ir.Expr) -> ir.Expr:
        """``x.'``; a transpose of a transpose cancels."""
        if isinstance(node, ir.Transpose) and not node.conjugate:
            return node.operand
        return ir.Transpose(loc=node.loc, operand=node, conjugate=False)

    @staticmethod
    def shift(node: ir.Expr, delta: int) -> ir.Expr:
        """``node + delta`` with integer literals folded (``(k - 1) + 1`` is ``k``)."""
        if delta == 0:
            return node
        if isinstance(node, ir.Num) and isinstance(node.value, int):
            return LiftContext.num(node.value + delta, loc=node.loc)
        if (
            isinstance(node, ir.UnOp)
            and node.op == "uminus"
            and isinstance(node.operand, ir.Num)
            and isinstance(node.operand.value, int)
        ):
            return LiftContext.num(-node.operand.value + delta, loc=node.loc)
        if (
            isinstance(node, ir.BinOp)
            and node.op in ("plus", "minus")
            and isinstance(node.right, ir.Num)
            and isinstance(node.right.value, int)
        ):
            carried = node.right.value if node.op == "plus" else -node.right.value
            return LiftContext.shift(node.left, carried + delta)
        op = "plus" if delta > 0 else "minus"
        return ir.BinOp(loc=node.loc, op=op, left=node, right=LiftContext.num(abs(delta)))

    # --- dispatch ----------------------------------------------------------------

    def expr(self, node: ast.expr) -> ir.Expr:
        """Lift a Python expression into the IR."""
        saved = self.in_condition
        # Only `and`/`or`/`not` pass the condition context on to their operands.
        boolean = isinstance(node, ast.BoolOp) or (
            isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        )
        self.in_condition = saved and boolean
        try:
            return self._expr(node)
        finally:
            self.in_condition = saved

    def _expr(self, node: ast.expr) -> ir.Expr:
        loc = loc_of(node)
        match node:
            case ast.Constant(value=value) if isinstance(value, bool):
                return ir.Call(loc=loc, func="true" if value else "false", args=[])
            case ast.Constant(value=value) if isinstance(value, int | float):
                return self.num(value, loc=loc)
            case ast.Name(id=name):
                if name in self.functions and name not in self.scope.variables:
                    return ir.FuncRef(loc=loc, name=name)
                return ir.Name(loc=loc, id=name)
            case ast.Call(func=ast.Attribute(attr="copy", value=inner)) if not node.args:
                # Every .m value is a private copy already.
                return self.expr(inner)
        tag = self.tag(node)
        if tag is None:
            self.decline(f"no translation for {type(node).__name__}")
        result = self._lifter(tag).lift(node, self)
        if not isinstance(result, ir.Expr):
            self.decline(f"{tag} did not give an expression")
        return result

    def _lifter(self, tag: str) -> ModuleType:
        rule = self.index.rule_for(tag)
        if rule is None or rule.python_status is not Status.MECHANICAL:
            raise _Unsupported(tag, rule)
        module = self._translator_for(rule.id)
        if module is None or not callable(getattr(module, "lift", None)):
            raise _Unsupported(tag, rule)
        return module

    # --- statements ----------------------------------------------------------------

    def block(self, body: list[ast.stmt]) -> list[ir.Stmt]:
        """Lift a statement list, with the comments that sit between its statements."""
        lifted: list[ir.Stmt] = []
        for stmt in body:
            lifted.extend(self.comments_before(stmt.lineno))
            lifted.extend(self.statement(stmt))
        return lifted

    def comments_before(self, line: int) -> list[ir.Stmt]:
        """Own-line comments above ``line`` that were not placed yet."""
        found: list[ir.Stmt] = []
        for number in sorted(self.standalone):
            if number >= line:
                break
            if number in self._used_comments:
                continue
            self._used_comments.add(number)
            text = self.standalone[number]
            if _is_directive(text):
                continue
            comment = ir.Comment(loc=ir.Loc(number, 1, number, 1), text=text)
            comment.syntax.append("syntax:comment")
            found.append(comment)
        return found

    def statement(self, stmt: ast.stmt) -> list[ir.Stmt]:
        """Lift one statement, or a hole for it."""
        findings = self.uses_by_stmt.get(id(stmt), [])
        rules = sorted({f.rule.id for f in findings if f.rule is not None})
        report = StmtReport(
            loc=loc_of(stmt), kind=type(stmt).__name__, rules=rules, status=Status.MECHANICAL
        )
        self.reports.append(report)
        blocking = [f for f in findings if f.status is not Status.MECHANICAL]
        if blocking:
            worst = next((f for f in blocking if f.unsupported), blocking[0])
            report.status = worst.status
            report.reason = "needs " + ", ".join(
                dict.fromkeys(
                    f"{f.tag} ({f.rule.id if f.rule is not None else 'no rule'})" for f in blocking
                )
            )
            return self.hole(stmt, report)
        outer = self.assumptions
        self.assumptions = []
        mark = len(self.reports)
        used = set(self._used_comments)
        file_directives = list(self.file_directives)
        try:
            lifted = self.translate_statement(stmt)
        except Decline as exc:
            del self.reports[mark:]
            self._used_comments = used
            self.file_directives = file_directives
            report.status = Status.GUIDED
            report.reason = f"translator declined: {exc.reason}"
            return self.hole(stmt, report)
        except _Unsupported as exc:
            del self.reports[mark:]
            self._used_comments = used
            self.file_directives = file_directives
            status = Status.UNSUPPORTED
            if exc.rule is not None and exc.rule.python_status is not None:
                status = exc.rule.python_status
            report.status = status
            label = exc.rule.id if exc.rule is not None else "no rule"
            report.reason = f"{exc.tag} ({label})"
            return self.hole(stmt, report)
        finally:
            mine = self.assumptions
            self.assumptions = outer
        notes = []
        comment = self.comments.get(getattr(stmt, "end_lineno", None) or stmt.lineno)
        if comment is None:
            comment = self.comments.get(stmt.lineno)
        if comment is not None and not _is_directive(comment):
            notes.append(comment)
        if mine:
            notes.append(f"dualect: assumed {'; '.join(mine)}")
        if notes:
            target = next((s for s in lifted if not isinstance(s, ir.Comment)), None)
            if target is not None:
                if isinstance(stmt, ast.FunctionDef | ast.For | ast.While | ast.If | ast.Match):
                    target.comment = "; ".join(notes)
                else:
                    last = [s for s in lifted if not isinstance(s, ir.Comment)][-1]
                    last.comment = "; ".join(notes)
        return lifted

    def translate_statement(self, stmt: ast.stmt) -> list[ir.Stmt]:
        """Lift a statement whose constructs are all mechanical from Python."""
        loc = loc_of(stmt)
        match stmt:
            case ast.Assign(targets=[target], value=value):
                return self.assignment(stmt, target, value)
            case ast.Expr(value=ast.Constant(value=str() as text)):
                return [ir.Comment(loc=loc, text=line.strip()) for line in text.strip().split("\n")]
            case ast.Expr(value=value):
                return [ir.ExprStmt(loc=loc, value=self.expr(value))]
            case ast.Import() | ast.ImportFrom() | ast.Pass():
                # Planning already required a mechanical rule for every module
                # imported other than numpy (`py:import:scipy.signal`).
                return []
        tag = _statement_tag(stmt)
        if tag is None:
            self.decline(f"no translation for {type(stmt).__name__}")
        result = self._lifter(tag).lift(stmt, self)
        if not isinstance(result, list):
            self.decline(f"{tag} did not give statements")
        return result

    def assignment(self, stmt: ast.Assign, target: ast.expr, value: ast.expr) -> list[ir.Stmt]:
        """Plain assignment; stores and unpacking go to their rules."""
        loc = loc_of(stmt)
        if isinstance(target, ast.Subscript):
            return self._lifter("py:assign-store").lift_store(stmt, self)
        if isinstance(target, ast.Tuple | ast.List):
            return self._lifter("py:assign-multi").lift_multi(stmt, self)
        if not isinstance(target, ast.Name):
            self.decline("assignment to this target")
        self.check_alias(target.id, value)
        if self.info(value).kind is PyKind.LIST:
            self.decline("a Python list (lists grow and nest; .m arrays do not)")
        if isinstance(value, ast.Call):
            tag = self.tag(value)
            rule = self.index.rule_for(tag) if tag is not None else None
            module = self._translator_for(rule.id) if rule is not None else None
            mechanical = rule is not None and rule.python_status is Status.MECHANICAL
            hook = getattr(module, "lift_assign", None) if mechanical else None
            if hook is not None:
                lifted = hook(stmt, self)
                if lifted is not None:
                    return list(lifted)
        rhs = self.expr(value)
        if isinstance(rhs, ir.Name) and rhs.id == target.id:
            return []  # `x = x.copy()`: the .m side copies anyway
        return [ir.Assign(loc=loc, targets=[ir.Name(loc=loc_of(target), id=target.id)], value=rhs)]

    def check_alias(self, name: str, value: ast.expr) -> None:
        """Decline ``name = value`` when NumPy would share memory that is later written.

        ``y = x``, ``y = x[1:3]`` and ``y = x.T`` bind views of ``x``: writing
        into ``y`` changes ``x`` and vice versa, which the ``.m`` copy cannot do.
        """
        info = self.info(value)
        # A bare name of an array binds the same array; a slice or .T is a view.
        if not (
            info.view
            or (
                isinstance(value, ast.Name)
                and info.kind is not PyKind.SCALAR
                and info.kind is not PyKind.STRING
                and info.kind is not PyKind.FUNC
            )
        ):
            return
        root = view_root(value)
        written = self.scope.mutated
        if name in written or (root is not None and root in written):
            self.decline(f"{name} shares memory with {root or 'another array'}, which is written")

    # --- holes ------------------------------------------------------------------------

    def hole(self, stmt: ast.stmt, report: StmtReport) -> list[ir.Stmt]:
        """A placeholder that names the gap and stops the ``.m`` script there."""
        loc = loc_of(stmt)
        lines = [
            f"% TODO(dualect): {report.status.value}, {report.reason}",
            f"% line {loc.line} of the source:",
        ]
        last = min(max(loc.end_line, loc.line), len(self.source_lines))
        for number in range(loc.line, last + 1):
            lines.append(f"%     {self.source_lines[number - 1].rstrip()}")
            self._used_comments.add(number)
        message = f"dualect: line {loc.line} was not translated ({report.status.value})"
        lines.append(f"error({message!r});".replace('"', "'"))
        return [ir.OpaqueStmt(loc=loc, construct=HOLE, text="\n".join(lines))]

    # --- scopes -------------------------------------------------------------------------

    @contextmanager
    def function_scope(self, fdef: ast.FunctionDef, outputs: list[str]) -> Iterator[ScopeInfo]:
        """Lift inside a ``def``: its own names and output variables."""
        saved = (self.scope, self.function, self.outputs)
        self.scope = scope_info(fdef.body, params=[a.arg for a in fdef.args.args])
        self.function = fdef
        self.outputs = outputs
        try:
            yield self.scope
        finally:
            self.scope, self.function, self.outputs = saved

    @contextmanager
    def loop(self) -> Iterator[None]:
        """Track loop nesting."""
        self.loop_depth += 1
        try:
            yield
        finally:
            self.loop_depth -= 1


class _Unsupported(Exception):
    """A construct with no mechanical (from Python) rule was reached while lifting."""

    def __init__(self, tag: str, rule: Rule | None) -> None:
        self.tag = tag
        self.rule = rule
        super().__init__(tag)


_STATEMENT_TAGS: dict[type, str] = {
    ast.For: "py:for",
    ast.While: "py:while",
    ast.If: "py:if",
    ast.Match: "py:match",
    ast.Break: "py:break",
    ast.Continue: "py:continue",
    ast.FunctionDef: "py:def",
    ast.Return: "py:return",
    ast.AugAssign: "py:augassign",
}


def _statement_tag(stmt: ast.stmt) -> str | None:
    return _STATEMENT_TAGS.get(type(stmt))


def _bodies(stmt: ast.stmt) -> list[list[ast.stmt]]:
    match stmt:
        case ast.For(body=body, orelse=orelse) | ast.While(body=body, orelse=orelse):
            return [body, orelse]
        case ast.If(body=body, orelse=orelse):
            return [body, orelse]
        case ast.FunctionDef(body=body):
            return [body]
        case ast.Match(cases=cases):
            return [case.body for case in cases]
        case ast.With(body=body):
            return [body]
        case ast.Try(body=body, orelse=orelse, finalbody=final, handlers=handlers):
            return [body, orelse, final, *(h.body for h in handlers)]
    return []


def _dedupe(uses: list[PyUse]) -> list[PyUse]:
    return list(dict.fromkeys(uses))


def _is_directive(text: str) -> bool:
    return text.strip().startswith(_DIRECTIVE_PREFIX)


def _comments(text: str) -> tuple[dict[int, str], dict[int, str]]:
    """Trailing comments by line, and own-line comments by line."""
    trailing: dict[int, str] = {}
    standalone: dict[int, str] = {}
    lines = text.splitlines()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return trailing, standalone
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        line, col = token.start
        body = token.string[1:]
        body = body[1:] if body.startswith(" ") else body
        if lines[line - 1][:col].strip():
            trailing[line] = body
        elif not (line == 1 and body.startswith("!")):
            standalone[line] = body
    return trailing, standalone


def view_root(node: ast.expr) -> str | None:
    """The variable a view expression shares memory with (``x`` of ``x[1:3].T``)."""
    while True:
        match node:
            case ast.Name(id=name):
                return name
            case ast.Subscript(value=inner) | ast.Attribute(value=inner):
                node = inner
            case ast.Call(func=ast.Attribute(attr="ravel" | "reshape" | "conj", value=inner)):
                node = inner
            case _:
                return None


_IN_PLACE_METHODS = frozenset({"sort", "fill", "resize", "itemset", "put", "append", "extend"})


def scope_info(body: list[ast.stmt], *, params: list[str]) -> ScopeInfo:
    """Names assigned and written in place in one scope (nested ``def`` bodies excluded)."""
    info = ScopeInfo(variables=set(params), params=list(params))
    for stmt in _scope_statements(body):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                for name_node in ast.walk(target):
                    if isinstance(name_node, ast.Name) and isinstance(target, ast.Name | ast.Tuple):
                        info.variables.add(name_node.id)
                if isinstance(target, ast.Subscript):
                    root = view_root(target.value)
                    if root is not None:
                        info.mutated.add(root)
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Subscript):
                root = view_root(stmt.target.value)
                if root is not None:
                    info.mutated.add(root)
            elif isinstance(stmt.target, ast.Name):
                info.variables.add(stmt.target.id)
                info.mutated.add(stmt.target.id)  # in place when it is an array
        elif isinstance(stmt, ast.For):
            for name_node in ast.walk(stmt.target):
                if isinstance(name_node, ast.Name):
                    info.variables.add(name_node.id)
        for node in ast.walk(stmt) if not isinstance(stmt, ast.FunctionDef) else []:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _IN_PLACE_METHODS
            ):
                root = view_root(node.func.value)
                if root is not None:
                    info.mutated.add(root)
    return info


def _scope_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    found: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, ast.FunctionDef):
            continue
        found.append(stmt)
        for inner in _bodies(stmt):
            found.extend(_scope_statements(inner))
    return found


@dataclass
class Lifted:
    """The result of lifting one Python file.

    :param body: the IR statements (holes included).
    :param reports: one report per statement.
    :param findings: every construct use.
    :param context: the context the lift ran in.
    """

    body: list[ir.Stmt]
    reports: list[StmtReport]
    findings: list[Finding]
    context: LiftContext


def lift_python(
    *, text: str, name: str, rules: list[Rule], translator_for: Callable[[str], ModuleType | None]
) -> Lifted:
    """Parse, analyse, plan and lift Python source.

    :param text: the source.
    :param name: its file name, for reports.
    :param rules: all loaded rules.
    :param translator_for: rule id -> its translator module.
    :raises PythonSyntaxError: when the text is not valid Python.
    """
    module = parse_python(text=text, name=name)
    analysis = analyze_python(module)
    ctx = LiftContext(
        module=module,
        text=text,
        analysis=analysis,
        index=PyRuleIndex(rules=rules),
        translator_for=translator_for,
    )
    body = ctx.block(module.body)
    body.extend(ctx.comments_before(len(ctx.source_lines) + 1))
    return Lifted(body=body, reports=ctx.reports, findings=ctx.findings, context=ctx)


def plan_python(*, text: str, name: str, rules: list[Rule]) -> list[Finding]:
    """The construct uses of Python source, resolved to rules (no lifting).

    :param text: the source.
    :param name: file name for reports.
    :param rules: all loaded rules.
    :raises DualectError: when the text is not valid Python.
    """
    module = parse_python(text=text, name=name)
    ctx = LiftContext(
        module=module,
        text=text,
        analysis=analyze_python(module),
        index=PyRuleIndex(rules=rules),
        translator_for=lambda _rule_id: None,
    )
    return ctx.findings


def parse_python(*, text: str, name: str) -> ast.Module:
    """Parse Python source, reporting a syntax error as a :class:`PythonSyntaxError`."""
    try:
        return ast.parse(text, filename=name)
    except SyntaxError as exc:
        raise PythonSyntaxError(f"{name}:{exc.lineno}: {exc.msg}") from exc


class PythonSyntaxError(DualectError):
    """The Python source could not be parsed."""
