# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Shape and type inference over Python source (the Python -> ``.m`` direction).

The ``.m`` -> Python direction infers facts over the IR (:mod:`dualect.infer`);
this module does the same job for a Python :mod:`ast` tree, *before* it is
lifted into the IR, because the lifting itself depends on the facts: ``A @ v``
is ``(A * v.').'`` when ``v`` is a 1-D array and ``A * B`` when both are 2-D,
``x[i]`` is an element of a vector but a whole row of a matrix, and
``x.max()`` may only become ``max(x)`` when ``x`` holds no NaN (the ``.m``
``max`` skips NaN, NumPy's propagates it).

Facts describe *NumPy* semantics: a :attr:`PyKind.VECTOR` is a 1-D array
(it becomes a ``1xN`` row in ``.m`` code), a :attr:`PyKind.MATRIX` a 2-D
array. The analysis is flow-sensitive within a body, joins branches, runs loop
bodies to a fixed point and analyses each function (and lambda) once for the
join of the arguments at its call sites, like :mod:`dualect.infer`.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum


class PyKind(StrEnum):
    """Coarse shape class of a Python value."""

    SCALAR = "scalar"  # a Python or NumPy number (or bool)
    VECTOR = "vector"  # a 1-D array
    MATRIX = "matrix"  # a 2-D array
    STRING = "string"
    FUNC = "func"
    LIST = "list"  # a list literal (an argument to np functions)
    TUPLE = "tuple"
    RANGE = "range"  # range(...)
    NONE = "none"
    MODULE = "module"  # the numpy module (or np.linalg)
    UNKNOWN = "unknown"


class DType(StrEnum):
    """Element type."""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


_ARRAYS = frozenset({PyKind.VECTOR, PyKind.MATRIX})


@dataclass(frozen=True)
class PyInfo:
    """What is known about one Python value.

    :param kind: shape class.
    :param dtype: element type.
    :param finite: known to hold no NaN or Inf.
    :param nonneg: known to hold no negative value.
    :param value: the exact value of a scalar constant, when known.
    :param items: element facts of a tuple (``A.shape``, a multiple return).
    :param length: number of elements of a vector, when known.
    :param func: for a lambda value, its node.
    :param module: for a module value, its dotted name (``"np"``, ``"np.linalg"``).
    :param view: the value may share memory with another array (a slice, a
        transpose, a plain alias): writing into it would change that array.
    """

    kind: PyKind = PyKind.UNKNOWN
    dtype: DType = DType.UNKNOWN
    finite: bool = False
    nonneg: bool = False
    value: int | float | complex | str | None = None
    items: tuple[PyInfo, ...] | None = field(default=None, compare=False)
    length: int | None = None
    func: ast.Lambda | None = field(default=None, compare=False)
    module: str | None = None
    view: bool = False

    @property
    def is_scalar(self) -> bool:
        """A single number."""
        return self.kind is PyKind.SCALAR

    @property
    def is_array(self) -> bool:
        """A NumPy array (1-D or 2-D)."""
        return self.kind in _ARRAYS

    @property
    def is_vector(self) -> bool:
        """A 1-D array."""
        return self.kind is PyKind.VECTOR

    @property
    def is_matrix(self) -> bool:
        """A 2-D array."""
        return self.kind is PyKind.MATRIX

    @property
    def is_bool(self) -> bool:
        """Boolean elements."""
        return self.dtype is DType.BOOL

    @property
    def is_int(self) -> bool:
        """Integer (or bool) elements: a NumPy store of a fraction would truncate."""
        return self.dtype in (DType.INT, DType.BOOL)


UNKNOWN = PyInfo()


def scalar(
    dtype: DType = DType.FLOAT,
    *,
    finite: bool = True,
    nonneg: bool = False,
    value: int | float | complex | None = None,
) -> PyInfo:
    """Facts for a scalar."""
    return PyInfo(kind=PyKind.SCALAR, dtype=dtype, finite=finite, nonneg=nonneg, value=value)


def array(
    kind: PyKind,
    dtype: DType = DType.FLOAT,
    *,
    finite: bool = False,
    nonneg: bool = False,
    length: int | None = None,
) -> PyInfo:
    """Facts for an array."""
    return PyInfo(kind=kind, dtype=dtype, finite=finite, nonneg=nonneg, length=length)


def join(a: PyInfo, b: PyInfo) -> PyInfo:
    """Facts that hold for a value that may be either ``a`` or ``b``."""
    if a == b and a.func is b.func:
        return a
    return PyInfo(
        kind=a.kind if a.kind is b.kind else PyKind.UNKNOWN,
        dtype=a.dtype if a.dtype is b.dtype else _promote(a.dtype, b.dtype, join=True),
        finite=a.finite and b.finite,
        nonneg=a.nonneg and b.nonneg,
        value=a.value if a.value == b.value else None,
        items=a.items if a.items == b.items else None,
        length=a.length if a.length == b.length else None,
        func=a.func if a.func is b.func else None,
        module=a.module if a.module == b.module else None,
        view=a.view or b.view,
    )


_ORDER = [DType.BOOL, DType.INT, DType.FLOAT, DType.COMPLEX]


def _promote(a: DType, b: DType, *, join: bool = False) -> DType:
    """NumPy's result type for two element types (for a join: the wider one)."""
    if DType.UNKNOWN in (a, b):
        return DType.UNKNOWN
    if join and a is not b:
        return DType.UNKNOWN if DType.BOOL in (a, b) else max(a, b, key=_ORDER.index)
    return max(a, b, key=_ORDER.index)


Env = dict[str, PyInfo]


def join_envs(a: Env, b: Env) -> Env:
    """Join two environments; a name missing on one side is unknown."""
    return {name: join(a.get(name, UNKNOWN), b.get(name, UNKNOWN)) for name in a.keys() | b.keys()}


#: NumPy element-wise functions that keep the argument's shape.
ELEMENTWISE = frozenset(
    {"abs", "sqrt", "exp", "log", "log2", "log10", "sin", "cos", "tan", "arcsin", "arccos",
     "arctan", "sinh", "cosh", "tanh", "floor", "ceil", "trunc", "sign", "real", "imag",
     "conj", "isnan", "isinf", "isfinite", "mod", "fmod", "arctan2", "hypot", "round"}
)  # fmt: skip
_BOOL_RESULT = frozenset({"isnan", "isinf", "isfinite"})
_REDUCTIONS = frozenset({"sum", "mean", "max", "min", "prod", "nanmax", "nanmin", "all", "any"})
_NAN_FREE_RESULT = frozenset({"floor", "ceil", "trunc", "sign", "abs", "round", "conj"})


@dataclass
class FunctionFacts:
    """What the analysis learned about one ``def``.

    :param params: joined argument facts over all call sites.
    :param result: facts about the returned value (a tuple for several).
    """

    params: list[PyInfo]
    result: PyInfo = UNKNOWN


@dataclass
class PyAnalysis:
    """Inference results for one Python module.

    :param infos: ``id(node)`` -> facts about that expression.
    :param functions: ``def`` name -> facts.
    :param numpy_alias: the name numpy is imported as (``"np"``), or ``None``.
    :param env_before: ``id(stmt)`` -> environment before the statement.
    """

    infos: dict[int, PyInfo] = field(default_factory=dict)
    functions: dict[str, FunctionFacts] = field(default_factory=dict)
    numpy_alias: str | None = None
    env_before: dict[int, Env] = field(default_factory=dict)

    def info(self, node: ast.AST) -> PyInfo:
        """Facts about an expression (unknown when it was never analysed)."""
        return self.infos.get(id(node), UNKNOWN)


def dotted(node: ast.expr, alias: str | None) -> str | None:
    """``np.linalg.solve`` for an attribute chain rooted at the numpy alias, else ``None``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and alias is not None and node.id == alias:
        return ".".join(["np", *reversed(parts)])
    return None


class _Inferrer:
    def __init__(self, module: ast.Module) -> None:
        self.module = module
        self.analysis = PyAnalysis()
        self.defs: dict[str, ast.FunctionDef] = {}
        for stmt in module.body:
            if isinstance(stmt, ast.FunctionDef):
                self.defs[stmt.name] = stmt
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name == "numpy":
                        self.analysis.numpy_alias = alias.asname or "numpy"
        self.call_sites: dict[str, list[list[PyInfo]]] = {}
        self.lambda_params: dict[int, list[PyInfo]] = {}
        self.returns: list[PyInfo] = []

    def run(self) -> PyAnalysis:
        for name, fdef in self.defs.items():
            self.analysis.functions[name] = FunctionFacts(params=[UNKNOWN] * len(fdef.args.args))
        for _round in range(3):
            self.call_sites = {}
            self.lambda_params = {}
            self.body(self.module.body, {})
            for name, fdef in self.defs.items():
                self.function(name, fdef)
        return self.analysis

    def function(self, name: str, fdef: ast.FunctionDef) -> None:
        facts = self.analysis.functions[name]
        sites = self.call_sites.get(name, [])
        params: list[PyInfo] = []
        defaults = fdef.args.defaults
        first_default = len(fdef.args.args) - len(defaults)
        for k, _arg in enumerate(fdef.args.args):
            passed = [site[k] for site in sites if k < len(site)]
            if k >= first_default and any(len(site) <= k for site in sites):
                passed.append(self.expr(defaults[k - first_default], {}))
            joined = passed[0] if passed else UNKNOWN
            for info in passed[1:]:
                joined = join(joined, info)
            params.append(joined)
        facts.params = params
        env = {arg.arg: info for arg, info in zip(fdef.args.args, params, strict=True)}
        outer = self.returns
        self.returns = []
        try:
            self.body(fdef.body, env)
            result = self.returns[0] if self.returns else PyInfo(kind=PyKind.NONE)
            for info in self.returns[1:]:
                result = join(result, info)
            facts.result = result
        finally:
            self.returns = outer

    # --- statements --------------------------------------------------------

    def body(self, stmts: list[ast.stmt], env: Env) -> Env:
        for stmt in stmts:
            self.analysis.env_before[id(stmt)] = dict(env)
            env = self.stmt(stmt, env)
        return env

    def stmt(self, stmt: ast.stmt, env: Env) -> Env:
        match stmt:
            case ast.Assign(targets=targets, value=value):
                info = self.expr(value, env)
                for target in targets:
                    env = self.bind(target, info, env, value)
                return env
            case ast.AugAssign(target=target, op=op, value=value):
                current = self.expr(target, env)
                result = self.binop(op, current, self.expr(value, env))
                if isinstance(target, ast.Name):
                    # In-place on an array: the dtype stays that of the target.
                    if current.is_array:
                        result = replace(result, dtype=current.dtype, view=current.view)
                    return {**env, target.id: result}
                return env
            case ast.Expr(value=value):
                self.expr(value, env)
                return env
            case ast.If(test=test, body=body, orelse=orelse):
                self.expr(test, env)
                return join_envs(self.body(body, dict(env)), self.body(orelse, dict(env)))
            case ast.For(target=target, iter=iterable, body=body, orelse=orelse):
                element = self.element(self.expr(iterable, env), iterable)
                state = dict(env)
                for _ in range(4):
                    entry = self.bind(target, element, dict(state), None)
                    out = self.body(body, entry)
                    merged = join_envs(state, out)
                    if merged == state:
                        break
                    state = merged
                entry = self.bind(target, element, dict(state), None)
                out = join_envs(state, self.body(body, entry))
                return self.body(orelse, out) if orelse else out
            case ast.While(test=test, body=body, orelse=orelse):
                state = dict(env)
                for _ in range(4):
                    self.expr(test, state)
                    merged = join_envs(state, self.body(body, dict(state)))
                    if merged == state:
                        break
                    state = merged
                self.expr(test, state)
                out = join_envs(state, self.body(body, dict(state)))
                return self.body(orelse, out) if orelse else out
            case ast.Match(subject=subject, cases=cases):
                self.expr(subject, env)
                joined = dict(env)
                for case in cases:
                    for pattern in ast.walk(case.pattern):
                        if isinstance(pattern, ast.MatchValue):
                            self.expr(pattern.value, env)
                    joined = join_envs(joined, self.body(case.body, dict(env)))
                return joined
            case ast.Return(value=value):
                self.returns.append(
                    self.expr(value, env) if value is not None else PyInfo(kind=PyKind.NONE)
                )
                return env
            case ast.FunctionDef():
                return env
        return env

    def bind(self, target: ast.expr, info: PyInfo, env: Env, value: ast.expr | None) -> Env:
        """Bind an assignment target; tuple targets take the tuple's items."""
        if isinstance(target, ast.Name):
            self.analysis.infos[id(target)] = info
            # `y = x` binds the same array: y is a view of x from here on.
            aliased = isinstance(value, ast.Name) and info.is_array
            return {**env, target.id: replace(info, view=info.view or aliased)}
        if isinstance(target, ast.Tuple):
            items = info.items or ()
            for k, element in enumerate(target.elts):
                item = items[k] if k < len(items) else UNKNOWN
                env = self.bind(element, item, env, None)
            return env
        if isinstance(target, ast.Subscript):
            self.expr(target, env)
            base = target.value
            if isinstance(base, ast.Name) and base.id in env:
                before = env[base.id]
                stored = self.expr(target, env)
                del stored
                updated = replace(
                    before,
                    finite=before.finite and info.finite,
                    nonneg=before.nonneg and info.nonneg,
                    value=None,
                )
                return {**env, base.id: updated}
        return env

    @staticmethod
    def element(info: PyInfo, iterable: ast.expr) -> PyInfo:
        """Facts about one item of a ``for`` iterable."""
        if info.kind is PyKind.RANGE:
            return scalar(DType.INT, nonneg=info.nonneg)
        if info.kind is PyKind.VECTOR:
            return scalar(info.dtype, finite=info.finite, nonneg=info.nonneg)
        if info.kind is PyKind.MATRIX:
            return array(PyKind.VECTOR, info.dtype, finite=info.finite, nonneg=info.nonneg)
        if info.kind is PyKind.LIST and info.items:
            item = info.items[0]
            for other in info.items[1:]:
                item = join(item, other)
            return item
        return UNKNOWN

    # --- expressions -------------------------------------------------------

    def expr(self, node: ast.expr, env: Env) -> PyInfo:
        info = self._expr(node, env)
        self.analysis.infos[id(node)] = info
        return info

    def _expr(self, node: ast.expr, env: Env) -> PyInfo:
        alias = self.analysis.numpy_alias
        match node:
            case ast.Constant(value=value):
                return _constant(value)
            case ast.Name(id=name):
                if name == alias:
                    return PyInfo(kind=PyKind.MODULE, module="np")
                return env.get(name, UNKNOWN)
            case ast.List(elts=elts) | ast.Tuple(elts=elts):
                items = tuple(self.expr(e, env) for e in elts)
                kind = PyKind.LIST if isinstance(node, ast.List) else PyKind.TUPLE
                return PyInfo(kind=kind, items=items, length=len(items))
            case ast.BinOp(left=left, op=op, right=right):
                return self.binop(op, self.expr(left, env), self.expr(right, env))
            case ast.UnaryOp(op=op, operand=operand):
                inner = self.expr(operand, env)
                if isinstance(op, ast.Not):
                    return scalar(DType.BOOL, nonneg=True)
                if isinstance(op, ast.USub):
                    value = -inner.value if isinstance(inner.value, int | float) else None
                    return replace(inner, nonneg=value == 0, value=value, view=False)
                return replace(inner, view=False)
            case ast.Compare(left=left, comparators=comparators):
                shape = self.expr(left, env)
                for other in comparators:
                    shape = _broadcast(shape, self.expr(other, env))
                return replace(shape, dtype=DType.BOOL, finite=True, nonneg=True, value=None)
            case ast.BoolOp(values=values):
                infos = [self.expr(v, env) for v in values]
                result = infos[0]
                for info in infos[1:]:
                    result = join(result, info)
                return result
            case ast.IfExp(test=test, body=body, orelse=orelse):
                self.expr(test, env)
                return join(self.expr(body, env), self.expr(orelse, env))
            case ast.Subscript():
                return self.subscript(node, env)
            case ast.Attribute():
                return self.attribute(node, env)
            case ast.Call():
                return self.call(node, env)
            case ast.Lambda():
                return PyInfo(kind=PyKind.FUNC, func=node)
            case ast.JoinedStr(values=values):
                for piece in values:
                    if isinstance(piece, ast.FormattedValue):
                        self.expr(piece.value, env)
                return PyInfo(kind=PyKind.STRING)
            case ast.ListComp(elt=elt, generators=[generator]) if not generator.ifs:
                # One `for` clause: the element is analysed with its target bound.
                item = self.element(self.expr(generator.iter, env), generator.iter)
                bound = self.bind(generator.target, item, dict(env), None)
                self.expr(elt, bound)
                return PyInfo(kind=PyKind.LIST)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.expr(child, env)
        return UNKNOWN

    def binop(self, op: ast.operator, a: PyInfo, b: PyInfo) -> PyInfo:
        if a.kind is PyKind.STRING and b.kind is PyKind.STRING and isinstance(op, ast.Add):
            return PyInfo(kind=PyKind.STRING)
        shape = _matmul(a, b) if isinstance(op, ast.MatMult) else _broadcast(a, b)
        if shape.kind is PyKind.UNKNOWN:
            return UNKNOWN
        dtype = _promote(a.dtype, b.dtype)
        if isinstance(op, ast.Div):
            dtype = DType.COMPLEX if dtype is DType.COMPLEX else DType.FLOAT
            if dtype is DType.UNKNOWN:
                dtype = DType.UNKNOWN
        if isinstance(op, ast.FloorDiv | ast.Mod) and dtype is DType.BOOL:
            dtype = DType.INT
        if isinstance(op, ast.Pow) and dtype is DType.BOOL:
            dtype = DType.INT
        finite = a.finite and b.finite and isinstance(op, ast.Add | ast.Sub | ast.Mult)
        if isinstance(op, ast.Pow):
            exponent = b.value
            finite = a.finite and b.finite and isinstance(exponent, int) and exponent >= 0
        nonneg = False
        if isinstance(op, ast.Add | ast.Mult | ast.Div | ast.MatMult):
            nonneg = a.nonneg and b.nonneg
        elif isinstance(op, ast.Pow):
            even = isinstance(b.value, int) and b.value % 2 == 0
            nonneg = a.nonneg or even
        elif isinstance(op, ast.Mod):
            nonneg = b.nonneg
        value = _fold(op, a.value, b.value) if a.is_scalar and b.is_scalar else None
        return replace(shape, dtype=dtype, finite=finite, nonneg=nonneg, value=value, view=False)

    def subscript(self, node: ast.Subscript, env: Env) -> PyInfo:
        target = self.expr(node.value, env)
        index = node.slice
        parts = index.elts if isinstance(index, ast.Tuple) else [index]
        subs = [
            self.expr(p, env) if not isinstance(p, ast.Slice) else self.slice(p, env) for p in parts
        ]
        facts = {
            "dtype": target.dtype,
            "finite": target.finite,
            "nonneg": target.nonneg,
        }
        if target.kind is PyKind.STRING:
            return PyInfo(kind=PyKind.STRING)
        if target.kind in (PyKind.TUPLE, PyKind.LIST) and target.items is not None:
            value = subs[0].value if len(subs) == 1 else None
            if isinstance(value, int) and -len(target.items) <= value < len(target.items):
                return target.items[value]
            return UNKNOWN
        if not target.is_array:
            return UNKNOWN
        dims = []
        for part, sub in zip(parts, subs, strict=True):
            if _is_newaxis(part, self.analysis.numpy_alias):
                dims.append("new")
            elif isinstance(part, ast.Slice):
                dims.append("slice")
            elif sub.is_scalar and not sub.is_bool:
                dims.append("int")
            elif sub.is_bool:
                dims.append("mask")
            else:
                dims.append("array")
        view = "slice" in dims or "new" in dims
        if target.is_vector:
            if dims == ["int"]:
                return scalar(**facts)  # type: ignore[arg-type]
            if dims in (["slice", "new"], ["new", "slice"]):
                return replace(array(PyKind.MATRIX, **facts), view=True)  # type: ignore[arg-type]
            if len(dims) == 1:
                return replace(array(PyKind.VECTOR, **facts), view=view)  # type: ignore[arg-type]
            return UNKNOWN
        # a matrix
        if dims == ["mask"]:
            return array(PyKind.VECTOR, **facts)  # type: ignore[arg-type]
        if len(dims) == 1:
            dims.append("slice")
        if len(dims) != 2 or "new" in dims:
            return UNKNOWN
        ints = dims.count("int")
        if ints == 2:
            return scalar(**facts)  # type: ignore[arg-type]
        if ints == 1:
            return replace(array(PyKind.VECTOR, **facts), view=view)  # type: ignore[arg-type]
        return replace(array(PyKind.MATRIX, **facts), view=view)  # type: ignore[arg-type]

    def slice(self, node: ast.Slice, env: Env) -> PyInfo:
        for part in (node.lower, node.upper, node.step):
            if part is not None:
                self.expr(part, env)
        return PyInfo(kind=PyKind.UNKNOWN)

    def attribute(self, node: ast.Attribute, env: Env) -> PyInfo:
        name = dotted(node, self.analysis.numpy_alias)
        if name is not None:
            if name == "np.linalg":
                return PyInfo(kind=PyKind.MODULE, module=name)
            constants = {
                "np.pi": scalar(nonneg=True, value=math.pi),
                "np.e": scalar(nonneg=True, value=math.e),
                "np.inf": scalar(finite=False, nonneg=True, value=math.inf),
                "np.nan": scalar(finite=False),
            }
            return constants.get(name, UNKNOWN)
        base = self.expr(node.value, env)
        if node.attr == "T":
            return replace(base, view=base.is_array or base.view)
        if node.attr == "size":
            return scalar(DType.INT, nonneg=True, value=base.length if base.is_vector else None)
        if node.attr == "shape":
            dim = scalar(DType.INT, nonneg=True)
            if base.is_vector:
                return PyInfo(kind=PyKind.TUPLE, items=(replace(dim, value=base.length),), length=1)
            if base.is_matrix:
                return PyInfo(kind=PyKind.TUPLE, items=(dim, dim), length=2)
            return PyInfo(kind=PyKind.TUPLE)
        if node.attr in ("real", "imag"):
            return replace(base, dtype=DType.FLOAT, view=base.is_array)
        if node.attr == "ndim":
            return scalar(DType.INT, nonneg=True)
        return UNKNOWN

    def call(self, node: ast.Call, env: Env) -> PyInfo:
        alias = self.analysis.numpy_alias
        args = [self.expr(a, env) for a in node.args]
        keywords = {k.arg: self.expr(k.value, env) for k in node.keywords if k.arg is not None}
        func = node.func
        name = dotted(func, alias)
        if name is not None:
            return self.numpy_call(name[3:], node, args, keywords)
        if isinstance(func, ast.Attribute):
            receiver = self.expr(func.value, env)
            result = self.method(func.attr, receiver, args, keywords)
            target = node.args[0] if node.args else None
            if func.attr == "astype" and isinstance(target, ast.Name):
                # x.astype(float): the element type is the one named.
                dtypes = {"float": DType.FLOAT, "bool": DType.BOOL, "int": DType.INT}
                result = replace(result, dtype=dtypes.get(target.id, DType.UNKNOWN))
                if target.id == "bool":
                    result = replace(result, finite=True, nonneg=True)
            return result
        if isinstance(func, ast.Name):
            fname = func.id
            self.analysis.infos[id(func)] = env.get(fname, UNKNOWN)
            if fname in self.defs and fname not in env:
                self.call_sites.setdefault(fname, []).append(args)
                return self.analysis.functions[fname].result
            if fname in env:
                handle = env[fname]
                lam = handle.func
                if lam is not None and len(lam.args.args) >= len(args):
                    return self.lambda_call(lam, args, env)
                return UNKNOWN
            return _builtin(fname, args)
        return UNKNOWN

    def lambda_call(self, lam: ast.Lambda, args: list[PyInfo], env: Env) -> PyInfo:
        """A lambda's result for this call; its body is annotated for every call."""
        params = [a.arg for a in lam.args.args]
        defaults = lam.args.defaults
        first_default = len(params) - len(defaults)
        full = list(args)
        for k in range(len(args), len(params)):
            full.append(self.expr(defaults[k - first_default], env))
        result = self.expr(lam.body, {**env, **dict(zip(params, full, strict=True))})
        seen = self.lambda_params.get(id(lam))
        joined = full if seen is None else [join(a, b) for a, b in zip(seen, full, strict=True)]
        self.lambda_params[id(lam)] = joined
        self.expr(lam.body, {**env, **dict(zip(params, joined, strict=True))})
        return result

    def numpy_call(
        self, name: str, node: ast.Call, args: list[PyInfo], keywords: dict[str, PyInfo]
    ) -> PyInfo:
        first = args[0] if args else UNKNOWN
        if name == "array":
            return _array_literal(first)
        if name in ("zeros", "ones"):
            if first.kind is PyKind.TUPLE:
                dims = first.length
                if dims == 1:
                    return array(PyKind.VECTOR, finite=True, nonneg=True)
                if dims == 2:
                    return array(PyKind.MATRIX, finite=True, nonneg=True)
                return UNKNOWN
            if first.is_scalar:
                length = first.value if isinstance(first.value, int) else None
                return array(PyKind.VECTOR, finite=True, nonneg=True, length=length)
            return UNKNOWN
        if name == "eye":
            return array(PyKind.MATRIX, finite=True, nonneg=True)
        if name == "linspace":
            count = args[2].value if len(args) > 2 else keywords.get("num", scalar(value=50)).value
            finite = all(a.finite for a in args[:2])
            points = count if isinstance(count, int) else None
            return array(PyKind.VECTOR, finite=finite, length=points)
        if name == "arange":
            ints = all(a.dtype is DType.INT for a in args)
            nonneg = (
                (len(args) == 1 or args[0].nonneg)
                and (len(args) < 3 or args[2].nonneg)
                and all(a.is_scalar for a in args)
            )
            return array(
                PyKind.VECTOR, DType.INT if ints else DType.FLOAT, finite=True, nonneg=nonneg
            )
        if name in ELEMENTWISE:
            shape = _broadcast(first, args[1]) if len(args) == 2 else _as_array(first)
            if shape.kind is PyKind.UNKNOWN:
                return UNKNOWN
            dtype = DType.BOOL if name in _BOOL_RESULT else DType.FLOAT
            if name in ("abs", "sign", "mod", "fmod") and first.dtype is DType.INT:
                dtype = DType.INT
            if first.dtype is DType.COMPLEX and name not in ("abs", "real", "imag", *_BOOL_RESULT):
                dtype = DType.COMPLEX
            return replace(
                shape,
                dtype=dtype,
                finite=(first.finite and name in _NAN_FREE_RESULT) or name in _BOOL_RESULT,
                nonneg=name in ("abs", "sqrt", "exp", "hypot")
                or name in _BOOL_RESULT
                or (first.nonneg and name in _NAN_FREE_RESULT),
                value=None,
                view=False,
            )
        if name in _REDUCTIONS:
            return self.method(name, first, args[1:], keywords)
        if name in ("argmax", "argmin", "nanargmax", "nanargmin", "count_nonzero"):
            return scalar(DType.INT, nonneg=True)
        if name in ("maximum", "minimum", "fmax", "fmin"):
            second = args[1] if len(args) > 1 else UNKNOWN
            shape = _broadcast(_as_array(first), _as_array(second))
            return replace(
                shape,
                dtype=_promote(first.dtype, second.dtype),
                finite=all(a.finite for a in args),
                nonneg=any(a.nonneg for a in args),
                view=False,
            )
        if name == "clip":
            return replace(_as_array(first), finite=all(a.finite for a in args), view=False)
        if name in ("logical_and", "logical_or", "logical_xor", "logical_not"):
            shape = _as_array(first)
            for other in args[1:]:
                shape = _broadcast(shape, _as_array(other))
            return replace(
                shape, dtype=DType.BOOL, finite=True, nonneg=True, value=None, view=False
            )
        if name == "outer":
            return array(
                PyKind.MATRIX,
                _promote(first.dtype, args[1].dtype) if len(args) > 1 else DType.UNKNOWN,
            )
        if name == "linalg.solve":
            b = args[1] if len(args) > 1 else UNKNOWN
            return replace(b, dtype=DType.FLOAT, finite=False, nonneg=False, view=False)
        if name == "linalg.matrix_power":
            return array(PyKind.MATRIX, first.dtype, finite=False)
        if name in ("flatnonzero",):
            return array(PyKind.VECTOR, DType.INT, finite=True, nonneg=True)
        if name in ("concatenate", "append"):
            parts = first.items if name == "concatenate" and first.items else tuple(args)
            dtype = DType.BOOL
            for part in parts:
                dtype = _promote(
                    dtype, part.dtype if part.kind is not PyKind.LIST else _list_dtype(part)
                )
            return array(PyKind.VECTOR, dtype, finite=all(_finite(p) for p in parts))
        if name in ("vstack", "column_stack"):
            parts = first.items or ()
            dtype = DType.BOOL
            for part in parts:
                dtype = _promote(dtype, part.dtype)
            return array(PyKind.MATRIX, dtype, finite=all(p.finite for p in parts))
        if name in ("size",):
            return scalar(DType.INT, nonneg=True)
        if name in ("conj", "transpose"):
            return replace(first, view=False)
        if name == "finfo":
            return PyInfo(kind=PyKind.UNKNOWN)
        return UNKNOWN

    def method(
        self, name: str, receiver: PyInfo, args: list[PyInfo], keywords: dict[str, PyInfo]
    ) -> PyInfo:
        if receiver.kind is PyKind.UNKNOWN and receiver.dtype is DType.UNKNOWN:
            return UNKNOWN
        if name in ("copy",):
            return replace(receiver, view=False)
        if name in ("conj",):
            return replace(receiver, view=False)
        if name in ("ravel", "flatten"):
            return replace(receiver, kind=PyKind.VECTOR, view=name == "ravel", length=None)
        if name == "reshape":
            shape = args[0] if args else UNKNOWN
            count = shape.length if shape.kind is PyKind.TUPLE else len(args)
            kind = PyKind.MATRIX if count == 2 else PyKind.VECTOR if count == 1 else PyKind.UNKNOWN
            return replace(receiver, kind=kind, view=True, length=None)
        if name == "astype":
            return replace(receiver, dtype=DType.UNKNOWN, view=False)
        if name in _REDUCTIONS:
            axis = keywords.get("axis", args[0] if args else None)
            dtype = receiver.dtype
            if name in ("sum", "prod") and dtype is DType.BOOL:
                dtype = DType.INT
            if name == "mean":
                dtype = DType.FLOAT if dtype is not DType.COMPLEX else dtype
            if name in ("all", "any"):
                dtype = DType.BOOL
            facts = {
                "dtype": dtype,
                "finite": receiver.finite and name != "prod",
                "nonneg": receiver.nonneg or name in ("all", "any"),
            }
            if axis is not None and receiver.is_matrix:
                return array(PyKind.VECTOR, **facts)  # type: ignore[arg-type]
            if receiver.is_array or receiver.kind is PyKind.LIST:
                return scalar(**facts)  # type: ignore[arg-type]
            return UNKNOWN
        if name in ("argmax", "argmin"):
            return scalar(DType.INT, nonneg=True)
        if name == "all" or name == "any":
            return scalar(DType.BOOL, nonneg=True)
        if name == "append":
            return PyInfo(kind=PyKind.NONE)
        return UNKNOWN


def _constant(value: object) -> PyInfo:
    if isinstance(value, bool):
        return scalar(DType.BOOL, nonneg=True, value=int(value))
    if isinstance(value, int):
        return scalar(DType.INT, nonneg=value >= 0, value=value)
    if isinstance(value, float):
        return scalar(DType.FLOAT, finite=math.isfinite(value), nonneg=value >= 0, value=value)
    if isinstance(value, complex):
        return scalar(DType.COMPLEX, value=value)
    if isinstance(value, str):
        return PyInfo(kind=PyKind.STRING, value=value)
    if value is None:
        return PyInfo(kind=PyKind.NONE)
    return UNKNOWN


def _fold(op: ast.operator, a: object, b: object) -> int | float | None:
    if not (isinstance(a, int | float) and isinstance(b, int | float)):
        return None
    try:
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            return a / b
    except (ZeroDivisionError, OverflowError):
        return None
    return None


def _list_dtype(info: PyInfo) -> DType:
    dtype = DType.BOOL
    for item in info.items or ():
        if item.kind is PyKind.LIST:
            dtype = _promote(dtype, _list_dtype(item))
        elif item.is_scalar or item.is_array:
            dtype = _promote(dtype, item.dtype)
        else:
            return DType.UNKNOWN
    return dtype


def _finite(info: PyInfo) -> bool:
    if info.kind is PyKind.LIST:
        return all(_finite(item) for item in info.items or ())
    return info.finite


def _array_literal(first: PyInfo) -> PyInfo:
    """``np.array(<list>)``: a vector of scalars or a matrix of equal rows."""
    if first.is_array:
        return replace(first, view=False)
    if first.kind is not PyKind.LIST or first.items is None:
        return UNKNOWN
    items = first.items
    dtype = _list_dtype(first)
    finite = _finite(first)
    nonneg = all(i.nonneg for i in items) if items else True
    if all(i.is_scalar for i in items):
        return array(PyKind.VECTOR, dtype, finite=finite, nonneg=nonneg, length=len(items))
    if all(
        i.kind is PyKind.LIST and i.items is not None and all(x.is_scalar for x in i.items)
        for i in items
    ):
        rows = {i.length for i in items}
        if len(rows) == 1:
            nonneg = all(x.nonneg for i in items for x in (i.items or ()))
            return array(PyKind.MATRIX, dtype, finite=finite, nonneg=nonneg)
    return UNKNOWN


def _as_array(info: PyInfo) -> PyInfo:
    """A list argument is a 1-D array (or a 2-D one for a list of lists)."""
    if info.kind is PyKind.LIST:
        return _array_literal(info)
    return info


def _broadcast(a: PyInfo, b: PyInfo) -> PyInfo:
    a, b = _as_array(a), _as_array(b)
    if a.is_scalar and b.is_scalar:
        return replace(a, value=None, view=False)
    if a.is_scalar and b.is_array:
        return replace(b, value=None, view=False)
    if b.is_scalar and a.is_array:
        return replace(a, value=None, view=False)
    if a.is_array and b.is_array:
        kind = PyKind.MATRIX if PyKind.MATRIX in (a.kind, b.kind) else PyKind.VECTOR
        return array(kind, length=a.length if a.length == b.length else None)
    return UNKNOWN


def _matmul(a: PyInfo, b: PyInfo) -> PyInfo:
    a, b = _as_array(a), _as_array(b)
    if not (a.is_array and b.is_array):
        return UNKNOWN
    if a.is_vector and b.is_vector:
        return scalar(finite=False)
    if a.is_matrix and b.is_matrix:
        return array(PyKind.MATRIX)
    return array(PyKind.VECTOR)


def _builtin(name: str, args: list[PyInfo]) -> PyInfo:
    first = args[0] if args else UNKNOWN
    if name == "range":
        nonneg = (len(args) == 1 or first.nonneg) and (len(args) < 3 or args[2].nonneg)
        return PyInfo(kind=PyKind.RANGE, nonneg=nonneg)
    if name == "len":
        return scalar(DType.INT, nonneg=True)
    if name == "abs":
        if first.is_scalar:
            value = abs(first.value) if isinstance(first.value, int | float) else None
            return replace(
                first,
                nonneg=True,
                value=value,
                dtype=DType.FLOAT if first.dtype is DType.COMPLEX else first.dtype,
            )
        return replace(first, nonneg=True, view=False)
    if name in ("max", "min"):
        if len(args) == 1 and first.kind is PyKind.TUPLE:
            return scalar(DType.INT, nonneg=True)
        if all(a.is_scalar for a in args) and args:
            result = args[0]
            for other in args[1:]:
                result = join(result, other)
            return replace(result, value=None)
        return UNKNOWN
    if name == "float":
        return scalar(DType.FLOAT, finite=first.finite, nonneg=first.nonneg)
    if name == "int":
        return scalar(DType.INT, nonneg=first.nonneg)
    if name == "bool":
        return scalar(DType.BOOL, nonneg=True)
    if name == "print":
        return PyInfo(kind=PyKind.NONE)
    if name == "lfilter" and len(args) >= 3:
        # scipy.signal.lfilter: the shape of the data, in floating point.
        return replace(_as_array(args[2]), dtype=DType.FLOAT, finite=False, value=None, view=False)
    return UNKNOWN


def _is_newaxis(node: ast.expr, alias: str | None) -> bool:
    return (isinstance(node, ast.Constant) and node.value is None) or dotted(
        node, alias
    ) == "np.newaxis"


def analyze_python(module: ast.Module) -> PyAnalysis:
    """Infer facts about every expression of a Python module.

    :param module: the parsed source.
    """
    return _Inferrer(module).run()
