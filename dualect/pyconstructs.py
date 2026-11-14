# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Construct tags for Python source (the Python -> ``.m`` direction).

The ``.m`` scanner tags what a statement uses (:mod:`dualect.constructs`); this
module does the same for Python, with tags in the ``py:`` namespace:
``py:op:matmult``, ``py:call:np.zeros``, ``py:method:sum``, ``py:attr:T``,
``py:subscript``, ``py:for``, ... Each rule lists the Python tags it covers
under ``from_python: match:`` in its ``rule.yaml``, next to the ``.m`` tags it
covers under ``match:``, so one rule folder serves both directions. A tag that
no rule matches is reported as unsupported by that tag (``py:call:np.fft.fft``,
``py:stmt:try``), which is the named error of the design.

Core constructs that mean the same in both languages carry no tag: names,
numbers, ``True``/``False``, plain assignment to a name, ``import numpy as
np``, ``pass`` and ``x.copy()`` (the ``.m`` side always copies).
"""

import ast
from dataclasses import dataclass

from dualect import ir
from dualect.pyinfer import PyAnalysis, PyKind, dotted

#: Tag families (besides ``py:call:<name>``, ``py:method:<name>``,
#: ``py:attr:<name>``, ``py:const:<name>``, ``py:import:<module>``,
#: ``py:stmt:<Type>`` and ``py:expr:<Type>``, whose names are open-ended).
PY_VOCABULARY: dict[str, str] = {
    "py:comment": "a # comment",
    "py:subscript": "indexing x[i], A[i, j]",
    "py:index:negative": "a negative index x[-1] (counts from the end)",
    "py:index:slice": "a slice x[a:b:c]",
    "py:index:mask": "indexing with a boolean mask",
    "py:index:newaxis": "a new axis c[:, np.newaxis]",
    "py:list": "a list literal",
    "py:tuple": "a tuple literal",
    "py:str": "a string literal",
    "py:fstring": "an f-string",
    "py:complex": "an imaginary literal 2j",
    "py:none": "None",
    "py:ifexp": "a conditional expression a if c else b",
    "py:lambda": "a lambda",
    "py:call-local": "a call to a function defined in the same file",
    "py:call-handle": "a call through a variable holding a lambda",
    "py:assign-multi": "unpacking several values: a, b = f(x)",
    "py:assign-store": "assigning into an element or slice",
    "py:augassign": "x += e and friends",
    "py:for": "a for loop",
    "py:while": "a while loop",
    "py:if": "if / elif / else",
    "py:match": "match / case",
    "py:break": "break",
    "py:continue": "continue",
    "py:def": "a function definition",
    "py:def-default": "a parameter with a default value",
    "py:return": "return",
    "py:return-multi": "returning several values",
    "py:cmp:chain": "a chained comparison a < b < c",
}
for _op in ("add", "sub", "mult", "matmult", "div", "floordiv", "mod", "pow", "lshift",
            "rshift", "bitor", "bitxor", "bitand", "usub", "uadd", "not", "invert"):  # fmt: skip
    PY_VOCABULARY[f"py:op:{_op}"] = f"the Python {_op} operator"
for _cmp in ("eq", "noteq", "lt", "lte", "gt", "gte", "is", "isnot", "in", "notin"):
    PY_VOCABULARY[f"py:cmp:{_cmp}"] = f"the {_cmp} comparison"
for _bool in ("and", "or"):
    PY_VOCABULARY[f"py:boolop:{_bool}"] = f"the {_bool} operator"

_OPEN_FAMILIES = ("py:call:", "py:method:", "py:attr:", "py:const:", "py:import:", "py:stmt:",
                  "py:expr:")  # fmt: skip


def is_known_py_tag(tag: str) -> bool:
    """Whether a ``py:`` tag is in the vocabulary (open families always are)."""
    if tag in PY_VOCABULARY:
        return True
    return any(tag.startswith(family) and len(tag) > len(family) for family in _OPEN_FAMILIES)


@dataclass(frozen=True)
class PyUse:
    """One use of a Python construct.

    :param tag: the construct tag.
    :param loc: where it is.
    """

    tag: str
    loc: ir.Loc


def loc_of(node: ast.AST) -> ir.Loc:
    """The span of a Python node, in the IR's 1-based convention."""
    line = getattr(node, "lineno", 0)
    col = getattr(node, "col_offset", 0) + 1
    end_line = getattr(node, "end_lineno", None) or line
    end_col = (getattr(node, "end_col_offset", None) or 0) + 1
    return ir.Loc(line, col, end_line, end_col)


_OPS = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mult", ast.MatMult: "matmult", ast.Div: "div",
    ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow", ast.LShift: "lshift",
    ast.RShift: "rshift", ast.BitOr: "bitor", ast.BitXor: "bitxor", ast.BitAnd: "bitand",
    ast.USub: "usub", ast.UAdd: "uadd", ast.Not: "not", ast.Invert: "invert",
}  # fmt: skip
_CMPS = {
    ast.Eq: "eq", ast.NotEq: "noteq", ast.Lt: "lt", ast.LtE: "lte", ast.Gt: "gt",
    ast.GtE: "gte", ast.Is: "is", ast.IsNot: "isnot", ast.In: "in", ast.NotIn: "notin",
}  # fmt: skip


def op_name(op: ast.operator | ast.unaryop) -> str:
    """The tag name of an operator (``add``, ``matmult``, ``usub``, ...)."""
    return _OPS[type(op)]


def cmp_name(op: ast.cmpop) -> str:
    """The tag name of a comparison (``eq``, ``lt``, ...)."""
    return _CMPS[type(op)]


class Tagger:
    """Tags Python nodes, given the inference results (names need them)."""

    def __init__(self, *, analysis: PyAnalysis, functions: set[str]) -> None:
        """:param analysis: inference results. :param functions: names of the file's ``def``s."""
        self.analysis = analysis
        self.functions = functions

    @property
    def alias(self) -> str | None:
        """The numpy alias."""
        return self.analysis.numpy_alias

    def primary(self, node: ast.AST) -> str | None:
        """The tag whose rule lifts ``node`` itself (``None`` for core nodes)."""
        match node:
            case ast.Constant(value=value):
                if isinstance(value, str):
                    return "py:str"
                if isinstance(value, complex):
                    return "py:complex"
                if value is None:
                    return "py:none"
                return None
            case ast.Name():
                return None
            case ast.BinOp(op=op) | ast.UnaryOp(op=op):
                return f"py:op:{op_name(op)}"
            case ast.Compare(ops=ops):
                return f"py:cmp:{cmp_name(ops[0])}" if len(ops) == 1 else "py:cmp:chain"
            case ast.BoolOp(op=op):
                return "py:boolop:and" if isinstance(op, ast.And) else "py:boolop:or"
            case ast.Subscript():
                return "py:subscript"
            case ast.Attribute():
                name = dotted(node, self.alias)
                if name is not None:
                    return f"py:const:{name}"
                return f"py:attr:{node.attr}"
            case ast.Call():
                return self.call_tag(node)
            case ast.List():
                return "py:list"
            case ast.Tuple():
                return "py:tuple"
            case ast.JoinedStr():
                return "py:fstring"
            case ast.IfExp():
                return "py:ifexp"
            case ast.Lambda():
                return "py:lambda"
        return f"py:expr:{type(node).__name__}"

    def call_tag(self, node: ast.Call) -> str:
        """``py:call:np.zeros``, ``py:call:abs``, ``py:method:sum``, ``py:call-local``, ..."""
        func = node.func
        name = dotted(func, self.alias)
        if name is not None:
            return f"py:call:{name}"
        if isinstance(func, ast.Attribute):
            if func.attr == "copy" and not node.args and not node.keywords:
                return "py:method:copy"
            return f"py:method:{func.attr}"
        if isinstance(func, ast.Name):
            info = self.analysis.info(func)
            if info.kind is PyKind.FUNC:
                return "py:call-handle"
            if func.id in self.functions:
                return "py:call-local"
            return f"py:call:{func.id}"
        return "py:expr:call-expression"

    def expression_uses(self, node: ast.AST) -> list[PyUse]:
        """Every tag inside an expression tree (lambdas included)."""
        uses: list[PyUse] = []
        self._walk(node, uses)
        return uses

    def _walk(self, node: ast.AST, uses: list[PyUse]) -> None:
        if isinstance(node, ast.expr) and not isinstance(node, ast.Slice):
            tag = self.primary(node)
            if tag is not None and tag != "py:method:copy":
                uses.append(PyUse(tag, loc_of(node)))
        if isinstance(node, ast.Subscript):
            uses.extend(self.subscript_uses(node))
        if isinstance(node, ast.Attribute) and dotted(node, self.alias) is not None:
            return  # `np.linalg.solve` is one name, not three
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and dotted(func, self.alias) is None:
                self._walk(func.value, uses)  # the receiver of a method
            joining = isinstance(func, ast.Attribute) and func.attr == "join"
            for arg in node.args:
                if joining and isinstance(arg, ast.ListComp | ast.GeneratorExp):
                    # `"".join([... for v in x])` is one construct, the join's:
                    # only the pieces of the comprehension are tagged.
                    self._walk(arg.elt, uses)
                    for generator in arg.generators:
                        self._walk(generator.iter, uses)
                        for condition in generator.ifs:
                            self._walk(condition, uses)
                    continue
                self._walk(arg, uses)
            for keyword in node.keywords:
                self._walk(keyword.value, uses)
            return
        if isinstance(node, ast.JoinedStr):
            return  # its pieces belong to the f-string
        if isinstance(node, ast.Lambda):
            for default in node.args.defaults:
                self._walk(default, uses)
            self._walk(node.body, uses)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr | ast.keyword | ast.comprehension):
                self._walk(child, uses)

    def subscript_uses(self, node: ast.Subscript) -> list[PyUse]:
        """Extra tags of one subscript: slices, negative indices, masks, new axes."""
        uses = []
        parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        for part in parts:
            loc = loc_of(part)
            if isinstance(part, ast.Slice):
                uses.append(PyUse("py:index:slice", loc))
                bounds = [part.lower, part.upper]
                if any(b is not None and _negative(b) for b in bounds):
                    uses.append(PyUse("py:index:negative", loc))
            elif dotted(part, self.alias) == "np.newaxis" or (
                isinstance(part, ast.Constant) and part.value is None
            ):
                uses.append(PyUse("py:index:newaxis", loc))
            elif _negative(part):
                uses.append(PyUse("py:index:negative", loc))
            elif self.analysis.info(part).is_bool and not self.analysis.info(part).is_scalar:
                uses.append(PyUse("py:index:mask", loc))
        return uses

    def statement_uses(self, stmt: ast.stmt) -> list[PyUse]:
        """Tags of one statement itself (headers, not nested bodies)."""
        loc = loc_of(stmt)
        uses: list[PyUse] = []
        add = uses.append
        match stmt:
            case ast.Assign(targets=targets, value=value):
                if len(targets) > 1:
                    add(PyUse("py:stmt:chained-assign", loc))
                for target in targets:
                    if isinstance(target, ast.Tuple | ast.List):
                        add(PyUse("py:assign-multi", loc))
                    elif isinstance(target, ast.Subscript):
                        add(PyUse("py:assign-store", loc))
                        uses.extend(self.expression_uses(target))
                    elif not isinstance(target, ast.Name):
                        uses.extend(self.expression_uses(target))
                uses.extend(self.expression_uses(value))
            case ast.AugAssign(target=target, op=op, value=value):
                add(PyUse("py:augassign", loc))
                add(PyUse(f"py:op:{op_name(op)}", loc))
                if not isinstance(target, ast.Name):
                    uses.extend(self.expression_uses(target))
                uses.extend(self.expression_uses(value))
            case ast.Expr(value=value):
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    add(PyUse("py:comment", loc))  # a docstring
                else:
                    uses.extend(self.expression_uses(value))
            case ast.For(target=target, iter=iterable, orelse=orelse):
                add(PyUse("py:for", loc))
                if orelse:
                    add(PyUse("py:stmt:for-else", loc))
                if not isinstance(target, ast.Name):
                    add(PyUse("py:stmt:for-unpack", loc))
                uses.extend(self.expression_uses(iterable))
            case ast.While(test=test, orelse=orelse):
                add(PyUse("py:while", loc))
                if orelse:
                    add(PyUse("py:stmt:while-else", loc))
                uses.extend(self.expression_uses(test))
            case ast.If(test=test):
                add(PyUse("py:if", loc))
                uses.extend(self.expression_uses(test))
                # elif chains are nested If nodes in orelse: tag their tests here.
                orelse = stmt.orelse
                while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                    uses.extend(self.expression_uses(orelse[0].test))
                    orelse = orelse[0].orelse
            case ast.Match(subject=subject):
                add(PyUse("py:match", loc))
                uses.extend(self.expression_uses(subject))
            case ast.Break():
                add(PyUse("py:break", loc))
            case ast.Continue():
                add(PyUse("py:continue", loc))
            case ast.Return(value=value):
                add(PyUse("py:return", loc))
                if isinstance(value, ast.Tuple):
                    add(PyUse("py:return-multi", loc))
                    for item in value.elts:
                        uses.extend(self.expression_uses(item))
                elif value is not None:
                    uses.extend(self.expression_uses(value))
            case ast.FunctionDef(args=args, decorator_list=decorators):
                add(PyUse("py:def", loc))
                if args.defaults:
                    add(PyUse("py:def-default", loc))
                if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
                    add(PyUse("py:stmt:def-varargs", loc))
                if decorators:
                    add(PyUse("py:stmt:decorator", loc))
            case ast.Import(names=names):
                for alias in names:
                    if alias.name != "numpy":
                        add(PyUse(f"py:import:{alias.name}", loc))
            case ast.ImportFrom(module=module):
                add(PyUse(f"py:import:{module or '.'}", loc))
            case ast.Pass():
                pass
            case _:
                add(PyUse(f"py:stmt:{type(stmt).__name__.lower()}", loc))
        return uses


def _negative(node: ast.expr) -> bool:
    """A literal negative index, or an expression that starts with one (``-1 - n``)."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub):
        return _negative(node.left)
    return False
