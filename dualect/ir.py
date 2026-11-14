# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""The shared intermediate representation (docs/design.md section 5.1).

The IR is deliberately small and **direction-neutral**: node kinds name what a
construct *means*, not how one language spells it, and every place where the two
languages disagree is an explicit field rather than an implicit convention:

* :class:`Index` carries its ``base`` (1 for ``.m`` code, 0 for Python) and its
  linear ``order`` (``"F"``, column-major, for ``.m`` code; ``"C"`` for NumPy).
* :class:`Range` carries ``inclusive`` (the ``.m`` colon includes its stop;
  ``range``/``np.arange`` do not).
* Operators use the semantic names of the ``.m`` operator functions
  (``mtimes`` is the matrix product, ``times`` is element-wise), so a Python
  lowering maps ``@`` to ``mtimes`` and ``*`` to ``times``.
* :class:`End` is "the last index of the dimension being indexed", which a
  Python lowering produces from ``-1`` and an ``.m`` emitter prints as ``end``.

Milestone 4 lowers ``.m`` source into this IR (:mod:`dualect.lower`) and emits
Python from it (:mod:`dualect.emit`); milestone 5 adds a Python ``ast`` lowering
and an ``.m`` emitter over the same node kinds.

Anything the IR cannot express is kept as an :class:`Opaque` expression or
:class:`OpaqueStmt` carrying a construct tag (see :mod:`dualect.constructs`),
so the scanner can name it and the translator refuses it instead of guessing.

Nodes compare by identity (``eq=False``): analyses key their results on
``id(node)``, and two structurally equal expressions at different places are
different nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Loc:
    """A source span: 1-based lines and columns, end exclusive.

    :param line: first line.
    :param col: first column.
    :param end_line: last line.
    :param end_col: column just past the end.
    """

    line: int
    col: int
    end_line: int
    end_col: int

    def __str__(self) -> str:
        """Render as ``line:col``."""
        return f"{self.line}:{self.col}"


NO_LOC = Loc(0, 0, 0, 0)


@dataclass(eq=False)
class Node:
    """Base of every IR node.

    :param loc: where the node came from in the source.
    """

    loc: Loc


# --- expressions -----------------------------------------------------------


@dataclass(eq=False)
class Expr(Node):
    """Base of expression nodes."""


@dataclass(eq=False)
class Num(Expr):
    """A numeric literal.

    :param text: the literal as written (``"1e-3"``, ``"2i"``).
    :param value: its value; ``int`` when written without a fraction or
        exponent, ``complex`` for an imaginary literal.
    """

    text: str
    value: int | float | complex


@dataclass(eq=False)
class Str(Expr):
    """A string literal (escapes already resolved).

    :param value: the characters.
    :param quote: ``"'"`` (char array) or ``'"'`` (double-quoted).
    """

    value: str
    quote: str = "'"


@dataclass(eq=False)
class Name(Expr):
    """A bare identifier: a variable, or a function called without arguments.

    :param id: the identifier.
    """

    id: str


@dataclass(eq=False)
class Colon(Expr):
    """A bare ``:`` subscript: every index along that dimension."""


@dataclass(eq=False)
class End(Expr):
    """``end`` inside a subscript: the last index of the dimension being indexed.

    :param target: the expression being indexed (``end`` binds to the innermost).
    :param dim: 0-based position of the subscript this ``end`` appears in.
    :param ndims: how many subscripts that index expression has (one means
        linear indexing, where ``end`` is the element count).
    """

    target: Expr
    dim: int
    ndims: int


@dataclass(eq=False)
class Range(Expr):
    """``start:stop`` or ``start:step:stop``.

    :param start: first value.
    :param stop: the bound.
    :param step: increment, or ``None`` for 1.
    :param inclusive: whether ``stop`` itself is included when reached
        (``True`` for ``.m`` ranges).
    """

    start: Expr
    stop: Expr
    step: Expr | None = None
    inclusive: bool = True


@dataclass(eq=False)
class Matrix(Expr):
    """A bracket literal ``[a b; c d]``: rows of elements, concatenated.

    :param rows: elements of each row; an empty list is ``[]``.
    """

    rows: list[list[Expr]]


@dataclass(eq=False)
class CellLit(Expr):
    """A brace literal ``{a, b}`` (only lowered where it lists ``case`` values).

    :param items: the elements, row-major.
    """

    items: list[Expr]


@dataclass(eq=False)
class BinOp(Expr):
    """A binary operator, by semantic name.

    :param op: one of :data:`BINARY_OPS`.
    :param left: left operand.
    :param right: right operand.
    """

    op: str
    left: Expr
    right: Expr


@dataclass(eq=False)
class UnOp(Expr):
    """A prefix operator: ``uminus``, ``uplus`` or ``not``.

    :param op: the operator name.
    :param operand: the operand.
    """

    op: str
    operand: Expr


@dataclass(eq=False)
class Transpose(Expr):
    """A transpose.

    :param operand: the transposed expression.
    :param conjugate: ``True`` for the conjugate transpose (``'``).
    """

    operand: Expr
    conjugate: bool


@dataclass(eq=False)
class Index(Expr):
    """Indexing a variable with parentheses: ``x(i)``, ``A(i, j)``.

    :param target: the indexed expression (a :class:`Name` for ``.m`` code).
    :param args: one subscript per dimension, or one for linear indexing.
    :param base: the index base of the subscripts' values.
    :param order: memory order of linear indexing (``"F"``: column-major).
    """

    target: Expr
    args: list[Expr]
    base: int = 1
    order: str = "F"


@dataclass(eq=False)
class Call(Expr):
    """A function call.

    :param func: the function name as written in the source language.
    :param args: the arguments.
    :param local: ``True`` when the function is defined in the same file.
    :param handle: ``True`` when ``func`` is a variable holding a function handle.
    """

    func: str
    args: list[Expr]
    local: bool = False
    handle: bool = False


@dataclass(eq=False)
class Lambda(Expr):
    """An anonymous function ``@(params) body``.

    :param params: parameter names.
    :param body: the single expression it returns.
    """

    params: list[str]
    body: Expr


@dataclass(eq=False)
class FuncRef(Expr):
    """A handle to a named function, ``@sin``.

    :param name: the function name.
    """

    name: str


@dataclass(eq=False)
class Ignore(Expr):
    """The ``~`` placeholder for a discarded output in ``[~, i] = f(x)``."""


@dataclass(eq=False)
class Opaque(Expr):
    """An expression the IR does not model (a cell index, a struct field, ...).

    :param construct: construct tag naming it, e.g. ``"expr:field"``.
    :param text: the source text, for reports and LLM context.
    """

    construct: str
    text: str


# --- statements ------------------------------------------------------------


@dataclass(eq=False)
class Stmt(Node):
    """Base of statement nodes.

    :param comment: a comment on the same line as the statement (without the
        comment marker), or ``None``.
    :param syntax: construct tags for surface syntax the statement used that
        leaves no other trace in the IR (``"syntax:continuation"``, ...).
    """

    comment: str | None = field(default=None, kw_only=True)
    syntax: list[str] = field(default_factory=list, kw_only=True)


@dataclass(eq=False)
class Assign(Stmt):
    """``target = value`` or ``[t1, t2] = value``.

    :param targets: :class:`Name`, :class:`Index` or :class:`Ignore` nodes.
    :param value: the right-hand side.
    :param display: ``True`` when the statement is not terminated by ``;``,
        so the ``.m`` side prints the result.
    """

    targets: list[Expr]
    value: Expr
    display: bool = False


@dataclass(eq=False)
class AugAssign(Stmt):
    """``x += e`` and friends (``x++`` is ``x += 1``); ``.m`` code: Octave only.

    :param target: the updated variable or element.
    :param op: binary operator name (``plus``, ``minus``, ``mtimes``, ...).
    :param value: the right-hand side.
    """

    target: Expr
    op: str
    value: Expr


@dataclass(eq=False)
class ExprStmt(Stmt):
    """An expression evaluated for its effect or its display.

    :param value: the expression.
    :param display: ``True`` when not terminated by ``;``.
    """

    value: Expr
    display: bool = False


@dataclass(eq=False)
class If(Stmt):
    """``if`` / ``elseif`` / ``else``.

    :param branches: ``(condition, body)`` pairs in order.
    :param orelse: the ``else`` body, or ``None``.
    """

    branches: list[tuple[Expr, list[Stmt]]]
    orelse: list[Stmt] | None = None


@dataclass(eq=False)
class For(Stmt):
    """``for var = iterable``: iterates over the *columns* of ``iterable``.

    :param var: the loop variable.
    :param iterable: what it loops over.
    :param body: the loop body.
    """

    var: Name
    iterable: Expr
    body: list[Stmt]


@dataclass(eq=False)
class While(Stmt):
    """``while cond ... end``.

    :param cond: the condition, tested before each iteration.
    :param body: the loop body.
    """

    cond: Expr
    body: list[Stmt]


@dataclass(eq=False)
class DoUntil(Stmt):
    """``do ... until cond`` (Octave only): the body runs at least once.

    :param body: the loop body.
    :param cond: the exit condition, tested after each iteration.
    """

    body: list[Stmt]
    cond: Expr


@dataclass(eq=False)
class Switch(Stmt):
    """``switch`` / ``case`` / ``otherwise``.

    :param subject: the value being matched.
    :param cases: ``(values, body)``: a case matches when any value matches.
    :param otherwise: the ``otherwise`` body, or ``None``.
    """

    subject: Expr
    cases: list[tuple[list[Expr], list[Stmt]]]
    otherwise: list[Stmt] | None = None


@dataclass(eq=False)
class Break(Stmt):
    """``break``."""


@dataclass(eq=False)
class Continue(Stmt):
    """``continue``."""


@dataclass(eq=False)
class Return(Stmt):
    """``return`` (leave the function; its outputs are returned as they stand)."""


@dataclass(eq=False)
class Command(Stmt):
    """Command syntax, ``name word1 word2``: every word is a string argument.

    :param name: the command.
    :param words: its arguments, as text.
    """

    name: str
    words: list[str]


@dataclass(eq=False)
class Comment(Stmt):
    """A comment on its own line(s).

    :param text: the comment text without markers; several lines for a block.
    :param block: ``True`` for a ``%{ ... %}`` block.
    """

    text: str
    block: bool = False


@dataclass(eq=False)
class FunctionDef(Stmt):
    """``function [outs] = name(params) ... end``.

    :param name: the function name.
    :param params: parameter names (``varargin`` included as written).
    :param outputs: output variable names (``varargout`` included as written).
    :param body: the statements.
    """

    name: str
    params: list[str]
    outputs: list[str]
    body: list[Stmt]


@dataclass(eq=False)
class OpaqueStmt(Stmt):
    """A statement the IR does not model, or a region the parser could not read.

    :param construct: construct tag naming it (``"parse:error"`` for unparsed text).
    :param text: the source text.
    """

    construct: str
    text: str


@dataclass(eq=False)
class Program(Node):
    """A whole source file.

    :param body: top-level statements, including function definitions.
    :param source_name: file name, for reports.
    """

    body: list[Stmt]
    source_name: str = "<source>"


#: Binary operator names (the ``.m`` operator-function names, which are also
#: the construct tags ``op:<name>``).
BINARY_OPS = frozenset(
    {
        "plus", "minus", "mtimes", "times", "mrdivide", "rdivide", "mldivide", "ldivide",
        "mpower", "power", "eq", "ne", "lt", "le", "gt", "ge", "and", "or", "andand", "oror",
    }
)  # fmt: skip

COMPARISONS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


def child_exprs(node: Expr) -> list[Expr]:
    """Return the direct sub-expressions of an expression, in source order.

    :param node: any expression.
    """
    match node:
        case End():
            # Its target is a back-reference for analysis, not a child: walking
            # into it would visit the indexed expression twice.
            return []
        case Range(start=start, stop=stop, step=step):
            return [start, step, stop] if step is not None else [start, stop]
        case Matrix(rows=rows):
            return [item for row in rows for item in row]
        case CellLit(items=items):
            return list(items)
        case BinOp(left=left, right=right):
            return [left, right]
        case UnOp(operand=operand) | Transpose(operand=operand):
            return [operand]
        case Index(target=target, args=args):
            return [target, *args]
        case Call(args=args):
            return list(args)
        case Lambda(body=body):
            return [body]
    return []


def walk_expr(node: Expr) -> list[Expr]:
    """Return ``node`` and every expression below it, pre-order.

    :param node: the root expression.
    """
    found = [node]
    for child in child_exprs(node):
        found.extend(walk_expr(child))
    return found


def stmt_exprs(stmt: Stmt) -> list[Expr]:
    """Return the expressions that belong to a statement itself (not its body).

    For compound statements this is the header: the ``if``/``while``
    conditions, the ``for`` iterable, the ``switch`` subject and case values.

    :param stmt: any statement.
    """
    match stmt:
        case Assign(targets=targets, value=value):
            return [*targets, value]
        case AugAssign(target=target, value=value):
            return [target, value]
        case ExprStmt(value=value):
            return [value]
        case If(branches=branches):
            return [cond for cond, _body in branches]
        case For(var=var, iterable=iterable):
            return [var, iterable]
        case While(cond=cond) | DoUntil(cond=cond):
            return [cond]
        case Switch(subject=subject, cases=cases):
            return [subject, *(value for values, _body in cases for value in values)]
    return []


def stmt_bodies(stmt: Stmt) -> list[list[Stmt]]:
    """Return the nested statement lists of a compound statement.

    :param stmt: any statement.
    """
    match stmt:
        case If(branches=branches, orelse=orelse):
            bodies = [body for _cond, body in branches]
            return bodies + ([orelse] if orelse is not None else [])
        case For(body=body) | While(body=body) | DoUntil(body=body) | FunctionDef(body=body):
            return [body]
        case Switch(cases=cases, otherwise=otherwise):
            bodies = [body for _values, body in cases]
            return bodies + ([otherwise] if otherwise is not None else [])
    return []


def walk_stmts(body: list[Stmt]) -> list[Stmt]:
    """Return every statement in ``body`` and below, pre-order.

    :param body: a statement list.
    """
    found: list[Stmt] = []
    for stmt in body:
        found.append(stmt)
        for inner in stmt_bodies(stmt):
            found.extend(walk_stmts(inner))
    return found
