# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Shape and type inference over the IR (docs/design.md section 5.2, step 1).

Several translations depend on facts the source does not state: is ``a * b`` a
scalar product or a matrix product, is ``x`` a row or a column, can ``max(x)``
meet a NaN? This module infers what it can from literals, known built-in
return shapes and data flow, and records an :class:`Info` for every expression.
When a fact is unknown it stays unknown: translators then emit the safe form
with a ``# dualect: assumed ...`` comment, or decline, and verification checks
the result.

The analysis is flow-sensitive within a body, joins branches, iterates loops to
a fixed point, and specializes each local function once, for the join of the
argument facts at all of its call sites in the file.

What a built-in returns is known here for the core ones; a rule can teach it
about its own built-ins with ``infer_call(node, args, nargout)`` (the facts
of the result, or ``None``) and ``infer_outputs(node, args, nargout)`` (the
facts of each output of a multi-output call) in its ``translate.py``, so a
new built-in needs no engine change.

All facts describe the *source* (``.m``) semantics: shapes are ``rows x cols``.
How a fact maps to NumPy (a ``1xN`` row is a 1-D array) is the emitter's
business.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import ModuleType
from typing import TypedDict

from dualect import ir


class Kind(StrEnum):
    """Coarse shape class of a value."""

    SCALAR = "scalar"
    ROW = "row"  # 1xN, N != 1
    COL = "col"  # Nx1, N != 1
    VECTOR = "vector"  # a row or a column, orientation unknown
    MATRIX = "matrix"  # MxN, neither dimension known to be 1
    EMPTY = "empty"  # 0x0, written []
    ARRAY = "array"  # some numeric array, shape unknown
    STRING = "string"  # a char row vector (or double-quoted string)
    FUNC = "func"  # a function handle
    CELL = "cell"
    UNKNOWN = "unknown"


class Elem(StrEnum):
    """Element type."""

    REAL = "real"
    COMPLEX = "complex"
    LOGICAL = "logical"
    CHAR = "char"
    UNKNOWN = "unknown"


_ARRAY_KINDS = frozenset({Kind.ROW, Kind.COL, Kind.VECTOR, Kind.MATRIX, Kind.EMPTY, Kind.ARRAY})
_VECTOR_KINDS = frozenset({Kind.ROW, Kind.COL, Kind.VECTOR})


@dataclass(frozen=True)
class Info:
    """What is known about one value.

    :param kind: shape class.
    :param rows: number of rows, if known.
    :param cols: number of columns, if known.
    :param elem: element type.
    :param finite: known to contain no NaN or Inf.
    :param nonneg: known to contain no negative value.
    :param integer: known to hold integer values only.
    :param value: the exact value of a scalar, when known (constant folding).
    :param int_dtype: the *NumPy translation* may hold an integer dtype. Source
        numbers are doubles, but idiomatic NumPy writes ``np.array([1, 2])``;
        storing a fraction into such an array would truncate silently, so
        translators check this before an element assignment.
    :param maybe_undefined: on some path the variable was never assigned.
    :param func: for a function handle, the lambda it holds (if known).
    :param env: for a lambda, the variables it captured.
    """

    kind: Kind = Kind.UNKNOWN
    rows: int | None = None
    cols: int | None = None
    elem: Elem = Elem.UNKNOWN
    finite: bool = False
    nonneg: bool = False
    integer: bool = False
    value: float | complex | None = None
    int_dtype: bool = False
    maybe_undefined: bool = False
    func: ir.Lambda | None = field(default=None, compare=False)
    env: dict[str, Info] | None = field(default=None, compare=False, repr=False)

    @property
    def is_scalar(self) -> bool:
        """A single number (or logical)."""
        return self.kind is Kind.SCALAR

    @property
    def is_array(self) -> bool:
        """A numeric or logical array that is not known to be a scalar."""
        return self.kind in _ARRAY_KINDS

    @property
    def is_vector(self) -> bool:
        """A row or column (1-D in NumPy), possibly of unknown orientation."""
        return self.kind in _VECTOR_KINDS or self.kind is Kind.EMPTY

    @property
    def is_matrix(self) -> bool:
        """Known to be two-dimensional with neither dimension 1 (2-D in NumPy)."""
        return self.kind is Kind.MATRIX

    @property
    def is_logical(self) -> bool:
        """Known logical elements."""
        return self.elem is Elem.LOGICAL

    @property
    def is_real(self) -> bool:
        """Known real (or logical) elements; complex values need conjugation."""
        return self.elem in (Elem.REAL, Elem.LOGICAL)

    @property
    def numel(self) -> int | None:
        """Element count, if known."""
        if self.kind is Kind.SCALAR:
            return 1
        if self.rows is not None and self.cols is not None:
            return self.rows * self.cols
        return None


UNKNOWN = Info()


class _Facts(TypedDict, total=False):
    """Element facts shared by a value and the values derived from it."""

    elem: Elem
    finite: bool
    nonneg: bool
    integer: bool


def scalar(
    *,
    elem: Elem = Elem.REAL,
    finite: bool = True,
    nonneg: bool = False,
    integer: bool = False,
    value: float | complex | None = None,
) -> Info:
    """Info for a scalar."""
    return Info(
        kind=Kind.SCALAR,
        rows=1,
        cols=1,
        elem=elem,
        finite=finite,
        nonneg=nonneg,
        integer=integer,
        value=value,
    )


def array(
    rows: int | None,
    cols: int | None,
    *,
    elem: Elem = Elem.REAL,
    finite: bool = False,
    nonneg: bool = False,
    integer: bool = False,
    kind: Kind | None = None,
) -> Info:
    """Info for an array of the given dimensions (``None`` = unknown)."""
    if kind is None:
        kind = _kind_of(rows, cols)
    return Info(
        kind=kind,
        rows=rows,
        cols=cols,
        elem=elem,
        finite=finite,
        nonneg=nonneg,
        integer=integer,
    )


def _kind_of(rows: int | None, cols: int | None) -> Kind:
    if rows == 1 and cols == 1:
        return Kind.SCALAR
    if rows == 0 and cols == 0:
        return Kind.EMPTY
    if rows == 1:
        return Kind.ROW
    if cols == 1:
        return Kind.COL
    if rows is not None and cols is not None:
        return Kind.MATRIX
    return Kind.ARRAY


def join(a: Info, b: Info) -> Info:
    """The facts that hold for a value that may be either ``a`` or ``b``."""
    if a == b and a.func is b.func:
        return a
    kind = a.kind if a.kind is b.kind else Kind.UNKNOWN
    if kind is Kind.UNKNOWN:
        pair = {a.kind, b.kind}
        if pair <= {Kind.ROW, Kind.EMPTY}:
            kind = Kind.ROW
        elif pair <= {Kind.COL, Kind.EMPTY}:
            kind = Kind.COL
        elif pair <= _VECTOR_KINDS | {Kind.EMPTY}:
            kind = Kind.VECTOR
        elif pair <= _ARRAY_KINDS:
            kind = Kind.ARRAY
    rows = a.rows if a.rows == b.rows else None
    cols = a.cols if a.cols == b.cols else None
    if kind is Kind.ROW:
        rows = 1
    if kind is Kind.COL:
        cols = 1
    return Info(
        kind=kind,
        rows=rows,
        cols=cols,
        elem=a.elem if a.elem is b.elem else Elem.UNKNOWN,
        finite=a.finite and b.finite,
        nonneg=a.nonneg and b.nonneg,
        integer=a.integer and b.integer,
        value=a.value if a.value == b.value else None,
        int_dtype=a.int_dtype or b.int_dtype,
        maybe_undefined=a.maybe_undefined or b.maybe_undefined,
        func=a.func if a.func is b.func else None,
        env=a.env if a.func is b.func else None,
    )


Env = dict[str, Info]


def join_envs(a: Env, b: Env) -> Env:
    """Join two variable environments; a name missing on one side may be undefined."""
    missing = Info(maybe_undefined=True)
    return {name: join(a.get(name, missing), b.get(name, missing)) for name in a.keys() | b.keys()}


# --- built-in return facts -------------------------------------------------

#: Element-wise functions: same shape as the (first) argument.
ELEMENTWISE = frozenset(
    {"abs", "sqrt", "exp", "log", "log2", "log10", "sin", "cos", "tan", "asin", "acos",
     "atan", "sinh", "cosh", "tanh", "floor", "ceil", "round", "fix", "sign", "real",
     "imag", "conj", "isnan", "isinf", "isfinite", "mod", "rem", "atan2", "hypot"}
)  # fmt: skip
_LOGICAL_RESULT = frozenset({"isnan", "isinf", "isfinite"})
_INTEGER_RESULT = frozenset({"floor", "ceil", "round", "fix", "sign"})
_NONNEG_RESULT = frozenset({"abs", "sqrt", "exp", "hypot"})
_REDUCTIONS = frozenset({"sum", "mean", "max", "min", "prod"})
CONSTANTS: dict[str, Info] = {
    "pi": scalar(nonneg=True, value=math.pi),
    "e": scalar(nonneg=True, value=math.e),
    "eps": scalar(nonneg=True, value=2.220446049250313e-16),
    "Inf": scalar(finite=False, nonneg=True, value=math.inf),
    "inf": scalar(finite=False, nonneg=True, value=math.inf),
    "NaN": scalar(finite=False),
    "nan": scalar(finite=False),
    "true": scalar(elem=Elem.LOGICAL, nonneg=True, integer=True, value=1),
    "false": scalar(elem=Elem.LOGICAL, nonneg=True, integer=True, value=0),
    "i": scalar(elem=Elem.COMPLEX, value=1j),
    "j": scalar(elem=Elem.COMPLEX, value=1j),
}


def broadcast(a: Info, b: Info) -> Info:
    """Shape of an element-wise operation on ``a`` and ``b`` (implicit expansion)."""
    if a.is_scalar:
        return replace(b, value=None, func=None) if b.kind is not Kind.UNKNOWN else UNKNOWN
    if b.is_scalar:
        return replace(a, value=None, func=None) if a.kind is not Kind.UNKNOWN else UNKNOWN
    if a.kind is Kind.UNKNOWN or b.kind is Kind.UNKNOWN:
        return UNKNOWN
    if {a.kind, b.kind} == {Kind.ROW, Kind.COL}:
        row, col = (a, b) if a.kind is Kind.ROW else (b, a)
        return array(col.rows, row.cols, kind=Kind.MATRIX)
    if a.kind is b.kind:
        return array(
            a.rows if a.rows == b.rows else a.rows or b.rows,
            a.cols if a.cols == b.cols else a.cols or b.cols,
            kind=a.kind,
        )
    if Kind.MATRIX in (a.kind, b.kind) and {a.kind, b.kind} <= {Kind.MATRIX, Kind.ROW, Kind.COL}:
        matrix = a if a.kind is Kind.MATRIX else b
        return array(matrix.rows, matrix.cols, kind=Kind.MATRIX)
    return array(None, None, kind=Kind.ARRAY)


def _numeric_elem(a: Info, b: Info) -> Elem:
    if Elem.COMPLEX in (a.elem, b.elem):
        return Elem.COMPLEX
    if a.elem in (Elem.REAL, Elem.LOGICAL, Elem.CHAR) and b.elem in (
        Elem.REAL,
        Elem.LOGICAL,
        Elem.CHAR,
    ):
        return Elem.REAL
    return Elem.UNKNOWN


def _fold(op: str, a: Info, b: Info) -> float | complex | None:
    """Constant-fold a scalar binary operation when both values are known."""
    x, y = a.value, b.value
    if x is None or y is None:
        return None
    try:
        if op == "plus":
            return x + y
        if op == "minus":
            return x - y
        if op in ("mtimes", "times"):
            return x * y
        if op in ("mrdivide", "rdivide"):
            return x / y
    except (ZeroDivisionError, OverflowError):
        return None
    return None


@dataclass
class FunctionFacts:
    """What the analysis learned about one local function.

    :param params: joined argument facts over all call sites.
    :param outputs: facts about each output at the end of the function.
    :param call_counts: numbers of arguments the call sites passed.
    :param nargouts: numbers of outputs the call sites requested.
    """

    params: list[Info]
    outputs: list[Info]
    call_counts: set[int] = field(default_factory=set)
    nargouts: set[int] = field(default_factory=set)


@dataclass
class Analysis:
    """Inference results for one program.

    :param infos: ``id(expr)`` -> facts about that expression's value.
    :param functions: local function name -> facts.
    :param env_after: ``id(stmt)`` -> environment after the statement ran.
    :param env_before: ``id(stmt)`` -> environment before the statement ran.
    """

    infos: dict[int, Info] = field(default_factory=dict)
    functions: dict[str, FunctionFacts] = field(default_factory=dict)
    env_after: dict[int, Env] = field(default_factory=dict)
    env_before: dict[int, Env] = field(default_factory=dict)

    def info(self, expr: ir.Expr) -> Info:
        """Facts about an expression (unknown if it was never analysed)."""
        return self.infos.get(id(expr), UNKNOWN)


#: Built-in name -> the translator module of the rule that covers it, if any.
CallHook = Callable[[str], ModuleType | None]


class _Inferrer:
    def __init__(self, program: ir.Program, hook: CallHook | None = None) -> None:
        self.program = program
        self.hook = hook
        # id(call) of the calls whose facts came from a rule (int_dtype included).
        self.hooked: set[int] = set()
        self.analysis = Analysis()
        self.defs: dict[str, ir.FunctionDef] = {
            s.name: s for s in program.body if isinstance(s, ir.FunctionDef)
        }
        self.call_sites: dict[str, list[list[Info]]] = {}
        # id(lambda) -> its parameters' facts joined over every call so far in
        # this round (see :meth:`call`).
        self.lambda_params: dict[int, list[Info]] = {}
        self.current_function: ir.FunctionDef | None = None

    # --- driver --------------------------------------------------------------

    def run(self) -> Analysis:
        for name, fdef in self.defs.items():
            self.analysis.functions[name] = FunctionFacts(
                params=[UNKNOWN] * len(fdef.params), outputs=[UNKNOWN] * len(fdef.outputs)
            )
        # Script first (records call sites), then each function with the joined
        # facts of its callers; repeat so facts flow through call chains.
        for _round in range(3):
            self.call_sites = {}
            self.lambda_params = {}
            self.body(self.program.body, {})
            for name, fdef in self.defs.items():
                self.function(name, fdef)
        return self.analysis

    def function(self, name: str, fdef: ir.FunctionDef) -> None:
        facts = self.analysis.functions[name]
        sites = self.call_sites.get(name, [])
        params: list[Info] = []
        for index, _param in enumerate(fdef.params):
            passed = [site[index] for site in sites if index < len(site)]
            if not passed:
                params.append(UNKNOWN)
                continue
            joined = passed[0]
            for info in passed[1:]:
                joined = join(joined, info)
            params.append(joined)
        facts.params = params
        env: Env = {
            param: info for param, info in zip(fdef.params, params, strict=True) if param != "~"
        }
        outer = self.current_function
        self.current_function = fdef
        try:
            env = self.body(fdef.body, env)
        finally:
            self.current_function = outer
        facts.outputs = [env.get(out, UNKNOWN) for out in fdef.outputs]

    # --- statements ------------------------------------------------------------

    def body(self, stmts: list[ir.Stmt], env: Env) -> Env:
        for stmt in stmts:
            self.analysis.env_before[id(stmt)] = dict(env)
            env = self.stmt(stmt, env)
            self.analysis.env_after[id(stmt)] = dict(env)
        return env

    def stmt(self, stmt: ir.Stmt, env: Env) -> Env:
        match stmt:
            case ir.Assign():
                return self.assign(stmt, env)
            case ir.AugAssign(target=target, op=op, value=value):
                current = self.expr(target, env)
                rhs = self.expr(value, env)
                result = self.binop(op, current, rhs)
                if isinstance(target, ir.Name):
                    env = {**env, target.id: result}
                return env
            case ir.ExprStmt(value=value):
                self.expr(value, env)
                return env
            case ir.If(branches=branches, orelse=orelse):
                results = []
                for cond, body in branches:
                    self.expr(cond, env)
                    results.append(self.body(body, dict(env)))
                results.append(self.body(orelse, dict(env)) if orelse is not None else env)
                merged = results[0]
                for other in results[1:]:
                    merged = join_envs(merged, other)
                return merged
            case ir.Switch(subject=subject, cases=cases, otherwise=otherwise):
                self.expr(subject, env)
                results = []
                for values, body in cases:
                    for value in values:
                        self.expr(value, env)
                    results.append(self.body(body, dict(env)))
                results.append(self.body(otherwise, dict(env)) if otherwise is not None else env)
                merged = results[0]
                for other in results[1:]:
                    merged = join_envs(merged, other)
                return merged
            case ir.For(var=var, iterable=iterable, body=body):
                iter_info = self.expr(iterable, env)
                element = self.loop_element(iterable, iter_info)
                return self.loop(body, env, bind={var.id: element}, cond=None)
            case ir.While(cond=cond, body=body):
                return self.loop(body, env, bind={}, cond=cond)
            case ir.DoUntil(body=body, cond=cond):
                return self.loop(body, env, bind={}, cond=cond, cond_after=True)
            case ir.FunctionDef():
                return env
        return env

    def loop(
        self,
        body: list[ir.Stmt],
        env: Env,
        *,
        bind: Env,
        cond: ir.Expr | None,
        cond_after: bool = False,
    ) -> Env:
        """Iterate a loop body to a fixed point of the environment."""
        state = dict(env)
        for _ in range(6):
            entry = {**state, **bind}
            if cond is not None and not cond_after:
                self.expr(cond, entry)
            out = self.body(body, dict(entry))
            if cond is not None and cond_after:
                self.expr(cond, out)
            merged = join_envs(state, out)
            if merged == state:
                break
            state = merged
        else:
            # No fixed point (a value keeps growing): forget what changed.
            state = {k: v if env.get(k) == v else UNKNOWN for k, v in state.items()}
        # One last pass so the annotations describe the loop-invariant facts.
        entry = {**state, **bind}
        if cond is not None and not cond_after:
            self.expr(cond, entry)
        out = self.body(body, dict(entry))
        if cond is not None and cond_after:
            self.expr(cond, out)
        return join_envs(state, out)

    @staticmethod
    def loop_element(iterable: ir.Expr, info: Info) -> Info:
        """Facts about the loop variable: one *column* of the iterable per pass."""
        base: _Facts = {
            "elem": info.elem,
            "finite": info.finite,
            "nonneg": info.nonneg,
            "integer": info.integer,
        }
        if info.kind in (Kind.ROW, Kind.SCALAR):
            return replace(scalar(**base), int_dtype=info.int_dtype)
        if info.kind in (Kind.COL, Kind.MATRIX):
            return replace(array(info.rows, 1, **base), int_dtype=info.int_dtype)
        return UNKNOWN

    def assign(self, stmt: ir.Assign, env: Env) -> Env:
        value = stmt.value
        targets = stmt.targets
        if len(targets) > 1:
            outputs = self.multi_outputs(value, env, len(targets))
            env = dict(env)
            for target, info in zip(targets, outputs, strict=True):
                if isinstance(target, ir.Name):
                    env[target.id] = info
                else:
                    self.expr(target, env)
            return env
        rhs = self.expr(value, env)
        target = targets[0]
        if isinstance(target, ir.Name):
            return {**env, target.id: rhs}
        if isinstance(target, ir.Index) and isinstance(target.target, ir.Name):
            name = target.target.id
            before = env.get(name)
            for arg in target.args:
                self.expr(arg, env)
            self.analysis.infos[id(target.target)] = before or UNKNOWN
            self.analysis.infos[id(target)] = UNKNOWN
            if (
                before is None
                or before.kind in (Kind.UNKNOWN, Kind.EMPTY)
                or index_may_grow(target, before, self.analysis)
            ):
                grown = array(None, None, elem=_numeric_elem(before or UNKNOWN, rhs))
                # np.append and np.pad keep an integer dtype (np.zeros and
                # [] are floats), so a later fractional store must be refused.
                keeps_ints = (
                    before is not None
                    and before.kind not in (Kind.UNKNOWN, Kind.EMPTY)
                    and before.int_dtype
                    and rhs.int_dtype
                )
                if (
                    before is not None
                    and before.kind in (Kind.ROW, Kind.EMPTY)
                    and len(target.args) == 1
                ):
                    grown = array(1, None, elem=grown.elem)
                elif before is not None and before.kind is Kind.COL and len(target.args) == 1:
                    grown = array(None, 1, elem=grown.elem)
                return {**env, name: replace(grown, int_dtype=keeps_ints)}
            elem = before.elem
            if rhs.elem is Elem.COMPLEX:
                elem = Elem.COMPLEX
            elif before.elem is Elem.LOGICAL and rhs.elem is not Elem.LOGICAL:
                elem = Elem.UNKNOWN
            updated = replace(
                before,
                elem=elem,
                finite=before.finite and rhs.finite,
                nonneg=before.nonneg and rhs.nonneg,
                integer=before.integer and rhs.integer,
                value=None,
            )
            return {**env, name: updated}
        self.expr(target, env)
        return env

    def multi_outputs(self, value: ir.Expr, env: Env, count: int) -> list[Info]:
        first = self.expr(value, env, nargout=count)
        if isinstance(value, ir.Call):
            arg_infos = [self.analysis.info(arg) for arg in value.args]
            hooked = self._hook(value, "infer_outputs", arg_infos, count)
            if isinstance(hooked, list):
                return [*hooked[:count], *[UNKNOWN] * (count - len(hooked))]
            if value.local and value.func in self.analysis.functions:
                outputs = self.analysis.functions[value.func].outputs
                return [outputs[k] if k < len(outputs) else UNKNOWN for k in range(count)]
            if value.func in ("max", "min") and arg_infos:
                position = replace(first, elem=Elem.REAL, finite=True, nonneg=True, integer=True)
                return [first, position] + [UNKNOWN] * (count - 2)
            if value.func == "size":
                dim = scalar(nonneg=True, integer=True)
                return [dim] * count
            if value.func == "find" and arg_infos:
                return [first] * count
        return [first] + [UNKNOWN] * (count - 1)

    # --- expressions -------------------------------------------------------------

    def expr(self, node: ir.Expr, env: Env, *, nargout: int = 1) -> Info:
        info = self._expr(node, env, nargout=nargout)
        self.analysis.infos[id(node)] = info
        return info

    def _expr(self, node: ir.Expr, env: Env, *, nargout: int) -> Info:
        match node:
            case ir.Num(value=value):
                if isinstance(value, complex):
                    return scalar(elem=Elem.COMPLEX, value=value)
                literal = scalar(nonneg=value >= 0, integer=float(value).is_integer(), value=value)
                return replace(literal, int_dtype=isinstance(value, int))
            case ir.Str(value=value):
                return Info(kind=Kind.STRING, rows=1, cols=len(value), elem=Elem.CHAR)
            case ir.Name(id=name):
                return env.get(name, UNKNOWN)
            case ir.Colon() | ir.End() | ir.Ignore():
                return scalar(nonneg=True, integer=True)
            case ir.Range():
                return self.range(node, env)
            case ir.Matrix(rows=rows):
                return self.matrix(rows, env)
            case ir.CellLit(items=items):
                for item in items:
                    self.expr(item, env)
                return Info(kind=Kind.CELL)
            case ir.BinOp(op=op, left=left, right=right):
                return self.binop(op, self.expr(left, env), self.expr(right, env))
            case ir.UnOp(op=op, operand=operand):
                inner = self.expr(operand, env)
                if op == "not":
                    return replace(
                        inner, elem=Elem.LOGICAL, finite=True, nonneg=True, integer=True, value=None
                    )
                if op == "uminus":
                    negated = -inner.value if inner.value is not None else None
                    return replace(inner, nonneg=negated == 0, value=negated, func=None)
                return inner
            case ir.Transpose(operand=operand):
                inner = self.expr(operand, env)
                kind = {Kind.ROW: Kind.COL, Kind.COL: Kind.ROW}.get(inner.kind, inner.kind)
                return replace(inner, kind=kind, rows=inner.cols, cols=inner.rows)
            case ir.Index():
                result = self.index(node, env)
                target = self.analysis.info(node.target)
                return replace(result, int_dtype=target.int_dtype)
            case ir.Call():
                result = self.call(node, env, nargout=nargout)
                return replace(result, int_dtype=self.int_result(node, result))
            case ir.Lambda():
                return Info(kind=Kind.FUNC, func=node, env=dict(env))
            case ir.FuncRef():
                return Info(kind=Kind.FUNC)
        return UNKNOWN

    def range(self, node: ir.Range, env: Env) -> Info:
        start = self.expr(node.start, env)
        stop = self.expr(node.stop, env)
        step = self.expr(node.step, env) if node.step is not None else scalar(integer=True, value=1)
        count = range_count(start.value, step.value, stop.value)
        finite = start.finite and stop.finite and step.finite
        info = array(
            1,
            count,
            finite=finite,
            nonneg=start.nonneg and step.nonneg,
            integer=start.integer and step.integer,
            kind=Kind.ROW if count != 1 else Kind.SCALAR,
        )
        # An integer range translates to np.arange / range(), which hold ints.
        return replace(info, int_dtype=start.integer and step.integer)

    def matrix(self, rows: list[list[ir.Expr]], env: Env) -> Info:
        if not rows:
            return array(0, 0, finite=True, nonneg=True, integer=True)
        infos = [[self.expr(item, env) for item in row] for row in rows]
        flat = [info for row in infos for info in row]
        if all(info.kind is Kind.STRING for info in flat) and len(rows) == 1:
            cols = sum(info.cols or 0 for info in flat) if all(i.cols for i in flat) else None
            return Info(kind=Kind.STRING, rows=1, cols=cols, elem=Elem.CHAR)
        elem = Elem.LOGICAL if all(i.elem is Elem.LOGICAL for i in flat) else Elem.REAL
        if any(i.elem is Elem.COMPLEX for i in flat):
            elem = Elem.COMPLEX
        if any(i.elem in (Elem.UNKNOWN, Elem.CHAR) for i in flat):
            elem = Elem.UNKNOWN
        facts: _Facts = {
            "elem": elem,
            "finite": all(i.finite for i in flat),
            "nonneg": all(i.nonneg for i in flat),
            "integer": all(i.integer for i in flat),
        }
        # Each row: horizontal concatenation; then the rows stack vertically.
        row_shapes: list[tuple[int | None, int | None]] = []
        for row in infos:
            heights = {i.rows for i in row}
            if any(i.kind in (Kind.UNKNOWN, Kind.ARRAY, Kind.VECTOR, Kind.CELL) for i in row):
                return array(None, None, **facts)
            nonempty = [i for i in row if i.kind is not Kind.EMPTY]
            if not nonempty:
                row_shapes.append((0, 0))
                continue
            heights = {i.rows for i in nonempty}
            height = heights.pop() if len(heights) == 1 else None
            widths = [i.cols for i in nonempty]
            width = sum(w for w in widths if w is not None) if None not in widths else None
            row_shapes.append((height, width))
        shapes = [s for s in row_shapes if s != (0, 0)]
        if not shapes:
            return array(0, 0, **facts)
        widths_set = {w for _h, w in shapes}
        width = widths_set.pop() if len(widths_set) == 1 else None
        heights_list = [h for h, _w in shapes]
        height = sum(h for h in heights_list if h is not None) if None not in heights_list else None
        info = array(height, width, **facts)
        if info.kind is Kind.SCALAR and len(flat) == 1:
            return replace(flat[0], func=None)
        return replace(info, int_dtype=all(i.int_dtype for i in flat))

    def binop(self, op: str, a: Info, b: Info) -> Info:
        if op in ir.COMPARISONS or op in ("and", "or"):
            shape = broadcast(a, b)
            return replace(
                shape, elem=Elem.LOGICAL, finite=True, nonneg=True, integer=True, value=None
            )
        if op in ("andand", "oror"):
            return scalar(elem=Elem.LOGICAL, nonneg=True, integer=True)
        elem = _numeric_elem(a, b)
        if op == "mtimes" and not (a.is_scalar or b.is_scalar):
            shape = matrix_product(a, b)
        elif op == "mrdivide" and not b.is_scalar:
            shape = array(a.rows, b.rows) if a.is_array and b.is_array else UNKNOWN
        elif op == "mldivide" and not a.is_scalar:
            shape = array(a.cols, b.cols) if a.is_array and b.is_array else UNKNOWN
        elif op == "mpower" and not (a.is_scalar and b.is_scalar):
            shape = a if a.is_matrix else UNKNOWN
        else:
            shape = broadcast(a, b)
        if shape.kind is Kind.UNKNOWN:
            return Info(elem=elem)
        finite = a.finite and b.finite and op in ("plus", "minus", "times", "mtimes")
        nonneg = False
        integer = False
        if op in ("plus", "times", "mtimes"):
            nonneg = a.nonneg and b.nonneg
            integer = a.integer and b.integer
        elif op == "minus":
            integer = a.integer and b.integer
        elif op in ("rdivide", "mrdivide"):
            nonneg = a.nonneg and b.nonneg
        elif op in ("power", "mpower"):
            even = b.value is not None and isinstance(b.value, int | float) and b.value % 2 == 0
            nonneg = a.nonneg or (even and a.is_real)
            integer = a.integer and b.integer and b.nonneg
            finite = a.finite and b.finite and b.nonneg and b.integer
        value = _fold(op, a, b) if a.is_scalar and b.is_scalar else None
        if isinstance(value, int | float) and math.isfinite(value):
            finite = True  # a constant-folded value is known exactly
        keeps_ints = op in ("plus", "minus", "times", "mtimes") or (
            op in ("power", "mpower") and b.integer and b.nonneg
        )
        return replace(
            shape,
            elem=elem,
            finite=finite,
            nonneg=nonneg,
            integer=integer,
            value=value,
            int_dtype=keeps_ints and a.int_dtype and b.int_dtype,
            func=None,
        )

    def index(self, node: ir.Index, env: Env) -> Info:
        target = self.expr(node.target, env)
        subs = [self.expr(arg, env) for arg in node.args]
        facts: _Facts = {
            "elem": target.elem,
            "finite": target.finite,
            "nonneg": target.nonneg,
            "integer": target.integer,
        }
        if target.kind is Kind.STRING:
            if len(subs) == 1 and subs[0].is_scalar and not isinstance(node.args[0], ir.Colon):
                return Info(kind=Kind.STRING, rows=1, cols=1, elem=Elem.CHAR)
            return Info(kind=Kind.STRING, rows=1, elem=Elem.CHAR)
        if len(node.args) == 1:
            arg, sub = node.args[0], subs[0]
            if isinstance(arg, ir.Colon):
                return array(target.numel, 1, **facts)
            if sub.is_scalar and not sub.is_logical:
                return scalar(**facts)
            if sub.is_logical:
                if target.kind is Kind.ROW:
                    return array(1, None, kind=Kind.ROW, **facts)
                if target.is_vector or target.is_matrix:
                    return array(None, 1, kind=Kind.COL, **facts)
                return array(None, None, **facts)
            count = sub.numel
            if target.kind is Kind.COL:
                return array(count, 1, kind=Kind.COL if count != 1 else None, **facts)
            if target.kind is Kind.ROW:
                return array(1, count, kind=Kind.ROW if count != 1 else None, **facts)
            if target.is_matrix and sub.kind in (Kind.ROW, Kind.COL):
                return array(sub.rows, sub.cols, **facts)
            return array(None, None, **facts)
        if len(node.args) == 2:
            dims: list[int | None] = []
            for axis, (arg, sub) in enumerate(zip(node.args, subs, strict=True)):
                extent = target.rows if axis == 0 else target.cols
                if isinstance(arg, ir.Colon):
                    dims.append(extent)
                elif sub.is_scalar and not sub.is_logical:
                    dims.append(1)
                else:
                    dims.append(sub.numel if not sub.is_logical else None)
            if target.kind is Kind.UNKNOWN:
                if dims == [1, 1]:
                    return scalar(**facts)
                return array(None, None, **facts)
            kind = None
            if dims[0] == 1 and dims[1] != 1:
                kind = Kind.ROW
            elif dims[1] == 1 and dims[0] != 1:
                kind = Kind.COL
            return array(dims[0], dims[1], kind=kind, **facts)
        return array(None, None, **facts)

    def call(self, node: ir.Call, env: Env, *, nargout: int) -> Info:
        args = [self.expr(arg, env) for arg in node.args]
        name = node.func
        if node.local:
            self.call_sites.setdefault(name, []).append(args)
            known = self.analysis.functions.get(name)
            if known is not None:
                known.call_counts.add(len(args))
                known.nargouts.add(nargout)
                return known.outputs[0] if known.outputs else UNKNOWN
            return UNKNOWN
        if node.handle:
            handle = env.get(name, UNKNOWN)
            lam = handle.func
            if lam is not None and len(lam.params) == len(args):
                captured = handle.env or {}
                # This call's result follows this call's arguments...
                result = self.expr(
                    lam.body, {**captured, **dict(zip(lam.params, args, strict=True))}
                )
                # ...but the lambda is translated once, so its body is annotated
                # with the facts that hold for *every* call: a lambda called
                # with a column and with a row has a vector of unknown
                # orientation, and its translators must decline or assume out
                # loud rather than follow whichever call came last.
                seen = self.lambda_params.get(id(lam))
                joined = (
                    args if seen is None else [join(a, b) for a, b in zip(seen, args, strict=True)]
                )
                self.lambda_params[id(lam)] = joined
                self.expr(lam.body, {**captured, **dict(zip(lam.params, joined, strict=True))})
                return result
            return UNKNOWN
        hooked = self._hook(node, "infer_call", args, nargout)
        if isinstance(hooked, Info):
            self.hooked.add(id(node))
            return hooked
        if not node.args and name in CONSTANTS:
            return CONSTANTS[name]
        if name in ("nargin", "nargout"):
            return scalar(nonneg=True, integer=True)
        first = args[0] if args else UNKNOWN
        if name in ("zeros", "ones", "eye", "rand", "randn"):
            return self.constructor(name, node.args, args)
        if name == "linspace":
            points: int | None = 100
            if len(args) >= 3:
                points = int(args[2].value) if isinstance(args[2].value, int | float) else None
            finite = all(a.finite for a in args[:2])
            if points == 1:
                return scalar(finite=finite)
            return array(1, points, finite=finite, kind=Kind.ROW)
        if name in ("numel", "length", "ndims", "columns", "rows"):
            count: int | None = None
            if name == "numel":
                count = first.numel
            elif name == "length" and first.rows is not None and first.cols is not None:
                count = max(first.rows, first.cols) if first.numel else 0
            return scalar(nonneg=True, integer=True, value=count)
        if name == "size":
            if len(args) >= 2:
                axis = args[1].value
                extent = first.rows if axis == 1 else first.cols if axis == 2 else None
                return scalar(nonneg=True, integer=True, value=extent)
            return array(1, 2, finite=True, nonneg=True, integer=True)
        if name in ("isempty", "strcmp", "strcmpi", "isscalar", "isvector", "any", "all"):
            if name in ("any", "all") and first.is_matrix:
                return array(1, first.cols, elem=Elem.LOGICAL, finite=True, nonneg=True)
            return scalar(elem=Elem.LOGICAL, nonneg=True, integer=True)
        if name in ELEMENTWISE:
            shape = broadcast(first, args[1]) if len(args) == 2 else first
            if shape.kind is Kind.UNKNOWN:
                return UNKNOWN
            elem = Elem.LOGICAL if name in _LOGICAL_RESULT else Elem.REAL
            if name in ("sqrt", "log", "log2", "log10", "asin", "acos") and not first.nonneg:
                elem = Elem.UNKNOWN  # may be complex
            if first.elem is Elem.COMPLEX and name not in ("abs", "real", "imag", "isnan"):
                elem = Elem.COMPLEX
            value = None
            if name == "abs" and isinstance(first.value, int | float):
                value = abs(first.value)
            return replace(
                shape,
                elem=elem,
                finite=first.finite
                and name not in ("exp", "log", "log2", "log10")
                or name in _LOGICAL_RESULT,
                nonneg=name in _NONNEG_RESULT or name in _LOGICAL_RESULT or first.nonneg,
                integer=name in _INTEGER_RESULT or name in _LOGICAL_RESULT or first.integer,
                value=value,
                func=None,
            )
        if name in _REDUCTIONS:
            return self.reduction(name, node.args, args)
        if name == "find":
            limit = len(args) >= 2
            positions: _Facts = {"finite": True, "nonneg": True, "integer": True}
            if first.kind is Kind.ROW:
                return array(1, None, kind=Kind.ROW, **positions)
            if first.is_array:
                return array(None, 1, kind=Kind.COL if not limit else None, **positions)
            return array(None, None, **positions)
        if name == "xor":
            return replace(
                broadcast(first, args[1] if len(args) > 1 else UNKNOWN), elem=Elem.LOGICAL
            )
        if name in ("num2str", "sprintf", "strcat", "upper", "lower", "strtrim"):
            return Info(kind=Kind.STRING, rows=1, elem=Elem.CHAR)
        if name in ("double", "logical") and len(args) == 1:
            elem = Elem.REAL if name == "double" else Elem.LOGICAL
            return replace(first, elem=elem, value=None, func=None, int_dtype=False)
        if name == "filter" and len(args) >= 3:
            # The output has the shape of the data (the third argument).
            return replace(args[2], elem=Elem.REAL, finite=False, value=None, func=None)
        if name in ("cumsum", "cumprod", "sort", "flip", "fliplr", "flipud", "diff"):
            return replace(first, value=None, func=None)
        return UNKNOWN

    def _hook(self, node: ir.Call, name: str, args: list[Info], nargout: int) -> object:
        """Ask the rule covering a built-in for its facts (``None``: it does not know)."""
        if self.hook is None or node.local or node.handle:
            return None
        module = self.hook(node.func)
        function = getattr(module, name, None) if module is not None else None
        return function(node, args, nargout) if callable(function) else None

    def int_result(self, node: ir.Call, result: Info) -> bool:
        """Whether the NumPy translation of a call may hold integers."""
        if node.local or node.handle or id(node) in self.hooked:
            return result.int_dtype
        if node.func in ("find", "size", "numel", "length", "ndims", "nargin", "nargout"):
            return True
        if node.func in ("sum", "max", "min", "prod", "abs", "sign", "cumsum", "sort"):
            return any(self.analysis.info(arg).int_dtype for arg in node.args[:2])
        return False

    def constructor(self, name: str, nodes: list[ir.Expr], args: list[Info]) -> Info:
        dims: list[int | None] = []
        if len(args) == 1 and args[0].kind is Kind.ROW and args[0].cols == 2:
            # zeros(size(A)): the dimensions of A, when the argument is size(A).
            inner = nodes[0]
            if isinstance(inner, ir.Call) and inner.func == "size" and len(inner.args) == 1:
                source = self.analysis.info(inner.args[0])
                dims = [source.rows, source.cols]
            else:
                dims = [None, None]
        else:
            for arg in args:
                if arg.kind is Kind.STRING:
                    continue
                dims.append(int(arg.value) if isinstance(arg.value, int | float) else None)
            if len(dims) == 1:
                dims = [dims[0], dims[0]]
            if not dims:
                dims = [1, 1]
        facts: _Facts = {"finite": True, "nonneg": name != "randn", "integer": name != "rand"}
        if name in ("rand", "randn"):
            facts["integer"] = False
        rows, cols = dims[0], dims[1] if len(dims) > 1 else None
        if rows != 1 and cols != 1 and not (len(args) == 1 and dims == [None, None]):
            # zeros(m, n) with neither size known to be 1 is written as a 2-D
            # array, so it is a matrix from here on.
            return array(rows, cols, kind=Kind.MATRIX, **facts)
        return array(rows, cols, **facts)

    def reduction(self, name: str, nodes: list[ir.Expr], args: list[Info]) -> Info:
        if not args:
            return UNKNOWN
        first = args[0]
        if name in ("max", "min") and len(args) == 2:
            shape = broadcast(first, args[1])
            return replace(
                shape,
                elem=_numeric_elem(first, args[1]),
                finite=first.finite and args[1].finite,
                nonneg=first.nonneg or args[1].nonneg,
                integer=first.integer and args[1].integer,
                value=None,
            )
        dim: int | None = None
        dim_arg = 2 if name in ("max", "min") else 1
        if len(args) > dim_arg:
            value = args[dim_arg].value
            dim = int(value) if isinstance(value, int | float) else None
            if dim is None:
                return UNKNOWN
        facts: _Facts = {
            "elem": Elem.REAL if first.elem in (Elem.LOGICAL, Elem.CHAR) else first.elem,
            "finite": first.finite and name != "prod",
            "nonneg": first.nonneg,
            "integer": first.integer and name != "mean",
        }
        if first.is_scalar:
            return scalar(**facts)
        if dim is None:
            if first.is_vector:
                return scalar(**facts)
            if first.is_matrix:
                return array(1, first.cols, kind=Kind.ROW, **facts)
            return UNKNOWN
        if first.is_matrix:
            if dim == 1:
                return array(1, first.cols, kind=Kind.ROW, **facts)
            if dim == 2:
                return array(first.rows, 1, kind=Kind.COL, **facts)
        if first.kind is Kind.ROW:
            return scalar(**facts) if dim == 2 else replace(first, value=None)
        if first.kind is Kind.COL:
            return scalar(**facts) if dim == 1 else replace(first, value=None)
        return UNKNOWN


def matrix_product(a: Info, b: Info) -> Info:
    """Shape of ``a * b`` for two non-scalars."""
    if not (a.is_array and b.is_array):
        return UNKNOWN
    if a.kind is Kind.ROW and b.kind is Kind.COL:
        return scalar(finite=False)
    rows = a.rows if a.kind is not Kind.VECTOR else None
    cols = b.cols if b.kind is not Kind.VECTOR else None
    if a.kind is Kind.COL:
        rows = a.rows
    if b.kind is Kind.ROW:
        cols = b.cols
    kind = _kind_of(rows, cols)
    if kind is Kind.ARRAY:
        if a.kind is Kind.COL and b.kind is Kind.ROW:
            kind = Kind.MATRIX
        elif a.is_matrix and b.kind is Kind.COL:
            kind = Kind.COL
        elif a.kind is Kind.ROW and b.is_matrix:
            kind = Kind.ROW
        elif a.is_matrix and b.is_matrix:
            kind = Kind.MATRIX
    return array(rows, cols, kind=kind)


def range_count(
    start: float | complex | None, step: float | complex | None, stop: float | complex | None
) -> int | None:
    """Number of elements of ``start:step:stop`` when all three are known reals.

    Uses a relative tolerance like the interpreter does, so ``0:0.1:1`` has 11
    elements even though ``(1 - 0) / 0.1`` is not exactly 10 in floating point.
    """
    if not all(isinstance(v, int | float) for v in (start, step, stop)):
        return None
    assert isinstance(start, int | float) and isinstance(step, int | float)
    assert isinstance(stop, int | float)
    if step == 0 or (stop - start) / step < 0:
        return 0
    span = (stop - start) / step
    nearest = round(span)
    if abs(span - nearest) <= 3 * 2.220446049250313e-16 * max(abs(span), 1):
        return int(nearest) + 1
    return math.floor(span) + 1


def index_may_grow(node: ir.Index, before: Info, analysis: Analysis) -> bool:
    """Whether assigning through ``node`` may grow the array (needs a guided rule).

    True for a subscript that is ``end + k`` (k > 0), or a known number beyond
    the known size of that dimension. A subscript that *might* be out of range
    is not flagged: NumPy raises ``IndexError`` there, so a wrong guess fails
    loudly instead of translating silently wrong.

    :param node: the indexed assignment target.
    :param before: facts about the array before the assignment.
    :param analysis: inference results (for constant subscripts).
    """
    for dim, arg in enumerate(node.args):
        if _is_end_plus(arg):
            return True
        value = analysis.info(arg).value
        if not isinstance(value, int | float):
            continue
        if len(node.args) == 1:
            limit = before.numel
        else:
            limit = before.rows if dim == 0 else before.cols if dim == 1 else 1
        if limit is not None and value > limit:
            return True
    return False


def _is_end_plus(arg: ir.Expr) -> bool:
    """``end + k`` with positive ``k`` (either operand order)."""
    if isinstance(arg, ir.BinOp) and arg.op == "plus":
        pair = (arg.left, arg.right)
        return any(isinstance(side, ir.End) for side in pair) and any(
            isinstance(side, ir.Num) and isinstance(side.value, int | float) and side.value > 0
            for side in pair
        )
    return False


def analyze(program: ir.Program, *, hook: CallHook | None = None) -> Analysis:
    """Infer facts about every expression of ``program``.

    :param program: lowered source.
    :param hook: built-in name -> the module of its rule, whose
        ``infer_call``/``infer_outputs`` describe what the built-in returns.
    """
    return _Inferrer(program, hook).run()
