# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Construct tags: the names the scanner gives to what a file uses.

Every statement is tagged with the constructs it uses (``op:mtimes``,
``index:end``, ``call:linspace``, ...). Each rule declares in its
``rule.yaml`` ``match`` list which tags it covers, so adding a rule for a new
built-in is a new folder with ``match: [call:foo]`` and no engine change. A tag
that no rule matches is reported as unsupported *by its tag*, which is the
"named unsupported error" of the design.

Core constructs that mean the same in both languages (numbers, variables,
plain assignment, expression statements) carry no tag and need no rule.
"""

from dataclasses import dataclass

from dualect import ir
from dualect.infer import Analysis, Kind, index_may_grow

#: Every tag the scanner can emit (besides ``call:<name>``), with a description.
#: ``dualect.validate`` rejects rule ``match`` entries outside this vocabulary.
VOCABULARY: dict[str, str] = {
    "syntax:comment": "a % or # comment",
    "syntax:hash-comment": "a # comment (Octave only)",
    "syntax:block-comment": "a %{ ... %} block comment",
    "syntax:continuation": "a ... line continuation",
    "syntax:command": "command syntax: name word1 word2",
    "syntax:display": "a statement without ; that prints its result",
    "syntax:script-marker": "a leading `1;` that marks a script defining functions",
    "literal:matrix": "a bracket literal [1 2; 3 4]",
    "literal:empty": "the empty matrix []",
    "literal:string": "a single-quoted char array 'abc'",
    "literal:dq-string": 'a double-quoted string "abc"',
    "literal:cell": "a cell array literal {a, b}",
    "expr:concat": "bracket concatenation of arrays [a, b] / [a; b]",
    "expr:string-concat": "bracket concatenation of strings ['ab', s]",
    "expr:range": "a colon range a:b or a:step:b",
    "expr:field": "struct field access s.f",
    "expr:cell-index": "cell indexing c{k}",
    "index:paren": "indexing with parentheses x(i), A(i, j)",
    "index:colon": "a : subscript selecting a whole dimension, A(:, j)",
    "index:colon-all": "A(:), all elements as one column",
    "index:end": "end inside a subscript",
    "index:logical": "indexing with a logical mask",
    "index:delete": "deleting elements: x(idx) = []",
    "assign:grow": "assignment past the end of an array, which grows it",
    "assign:multi": "multiple outputs: [a, b] = f(x)",
    "assign:ignore-output": "a discarded output: [~, i] = f(x)",
    "op:augassign": "x += e, x++ and friends (Octave only)",
    "stmt:if": "if / elseif / else",
    "stmt:for": "a for loop",
    "stmt:while": "a while loop",
    "stmt:do-until": "do ... until (Octave only)",
    "stmt:switch": "switch / case / otherwise",
    "stmt:break": "break",
    "stmt:continue": "continue",
    "stmt:return": "return",
    "stmt:try": "try / catch",
    "stmt:global": "global variables",
    "stmt:persistent": "persistent variables",
    "stmt:classdef": "a classdef block",
    "stmt:parfor": "a parfor loop",
    "stmt:spmd": "an spmd block",
    "func:definition": "a function definition",
    "func:multi-output": "a function with several outputs",
    "func:varargin": "variable inputs (varargin)",
    "func:varargout": "variable outputs (varargout)",
    "func:arguments-block": "an arguments validation block",
    "func:nargin": "nargin",
    "func:nargout": "nargout",
    "func:handle": "an anonymous function @(x) ...",
    "func:named-handle": "a handle to a named function @sin",
    "func:handle-expression": "a handle to an expression",
    "func:call-local": "a call to a function defined in the same file",
    "func:call-handle": "a call through a function handle variable",
    "parse:error": "text the parser could not read",
    "literal:number-format": "a numeric literal in an unsupported format",
    "expr:end-outside-index": "end used outside a subscript",
    "expr:call-expression": "calling the result of an expression",
}
for _op in sorted(ir.BINARY_OPS | {"uminus", "uplus", "not", "ctranspose", "transpose"}):
    VOCABULARY[f"op:{_op}"] = f"the {_op} operator"

#: Names tagged as something other than ``call:<name>``.
_SPECIAL_CALLS = {"nargin": "func:nargin", "nargout": "func:nargout"}


def is_known_tag(tag: str) -> bool:
    """Whether a tag is in the vocabulary (``call:<name>`` is always known)."""
    return tag in VOCABULARY or (tag.startswith("call:") and len(tag) > len("call:"))


@dataclass(frozen=True)
class Use:
    """One use of a construct.

    :param tag: the construct tag.
    :param loc: where it is.
    """

    tag: str
    loc: ir.Loc


def statement_uses(stmt: ir.Stmt, analysis: Analysis) -> list[Use]:
    """Tag a statement's own constructs (not those of nested statements).

    :param stmt: the statement.
    :param analysis: inference results, used where the construct depends on
        facts (a logical mask, an assignment that grows an array).
    """
    uses = [Use(tag, stmt.loc) for tag in stmt.syntax]
    add = uses.append
    match stmt:
        case ir.Assign(targets=targets, value=value):
            if len(targets) > 1:
                add(Use("assign:multi", stmt.loc))
            if any(isinstance(t, ir.Ignore) for t in targets):
                add(Use("assign:ignore-output", stmt.loc))
            deleting = isinstance(value, ir.Matrix) and not value.rows
            for target in targets:
                if isinstance(target, ir.Index):
                    uses.extend(_index_target_uses(stmt, target, analysis, deleting=deleting))
                elif not isinstance(target, ir.Name | ir.Ignore):
                    uses.extend(expression_uses(target, analysis))
            if deleting and isinstance(targets[0], ir.Index):
                add(Use("index:delete", stmt.loc))
            else:
                uses.extend(expression_uses(value, analysis))
        case ir.AugAssign(target=target, value=value):
            add(Use("op:augassign", stmt.loc))
            uses.extend(expression_uses(target, analysis))
            uses.extend(expression_uses(value, analysis))
        case ir.ExprStmt(value=value):
            uses.extend(expression_uses(value, analysis))
        case ir.If(branches=branches):
            add(Use("stmt:if", stmt.loc))
            for cond, _body in branches:
                uses.extend(expression_uses(cond, analysis))
        case ir.For(iterable=iterable):
            add(Use("stmt:for", stmt.loc))
            uses.extend(expression_uses(iterable, analysis))
        case ir.While(cond=cond):
            add(Use("stmt:while", stmt.loc))
            uses.extend(expression_uses(cond, analysis))
        case ir.DoUntil(cond=cond):
            add(Use("stmt:do-until", stmt.loc))
            uses.extend(expression_uses(cond, analysis))
        case ir.Switch(subject=subject, cases=cases):
            add(Use("stmt:switch", stmt.loc))
            uses.extend(expression_uses(subject, analysis))
            for values, _body in cases:
                for value in values:
                    uses.extend(expression_uses(value, analysis))
        case ir.Break():
            add(Use("stmt:break", stmt.loc))
        case ir.Continue():
            add(Use("stmt:continue", stmt.loc))
        case ir.Return():
            add(Use("stmt:return", stmt.loc))
        case ir.Command():
            add(Use("syntax:command", stmt.loc))
        case ir.FunctionDef(params=params, outputs=outputs):
            add(Use("func:definition", stmt.loc))
            if len(outputs) > 1:
                add(Use("func:multi-output", stmt.loc))
            if "varargin" in params:
                add(Use("func:varargin", stmt.loc))
            if "varargout" in outputs:
                add(Use("func:varargout", stmt.loc))
        case ir.OpaqueStmt(construct=construct):
            add(Use(construct, stmt.loc))
    return _dedupe(uses)


def _index_target_uses(
    stmt: ir.Assign, target: ir.Index, analysis: Analysis, *, deleting: bool
) -> list[Use]:
    uses = [Use("index:paren", target.loc)]
    for arg in target.args:
        uses.extend(_subscript_uses(arg, target, analysis))
    if deleting:
        return uses
    name = target.target.id if isinstance(target.target, ir.Name) else None
    before = analysis.env_before.get(id(stmt), {}).get(name) if name else None
    if before is None or before.kind is Kind.EMPTY or index_may_grow(target, before, analysis):
        uses.append(Use("assign:grow", target.loc))
    return uses


def _subscript_uses(arg: ir.Expr, index: ir.Index, analysis: Analysis) -> list[Use]:
    uses: list[Use] = []
    if isinstance(arg, ir.Colon):
        uses.append(Use("index:colon-all" if len(index.args) == 1 else "index:colon", arg.loc))
        return uses
    info = analysis.info(arg)
    if info.is_logical and not info.is_scalar:
        uses.append(Use("index:logical", arg.loc))
    uses.extend(expression_uses(arg, analysis))
    return uses


def expression_uses(node: ir.Expr, analysis: Analysis) -> list[Use]:
    """Tag the constructs inside one expression tree.

    :param node: the expression.
    :param analysis: inference results.
    """
    uses: list[Use] = []
    add = uses.append
    match node:
        case ir.Str(quote=quote):
            add(Use("literal:dq-string" if quote == '"' else "literal:string", node.loc))
        case ir.Matrix(rows=rows):
            if not rows:
                add(Use("literal:empty", node.loc))
            else:
                infos = [analysis.info(item) for row in rows for item in row]
                if all(info.kind is Kind.STRING for info in infos):
                    add(Use("expr:string-concat", node.loc))
                else:
                    add(Use("literal:matrix", node.loc))
                    if any(not info.is_scalar for info in infos):
                        add(Use("expr:concat", node.loc))
        case ir.CellLit():
            add(Use("literal:cell", node.loc))
        case ir.Range():
            add(Use("expr:range", node.loc))
        case ir.BinOp(op=op):
            add(Use(f"op:{op}", node.loc))
        case ir.UnOp(op=op):
            add(Use(f"op:{op}", node.loc))
        case ir.Transpose(conjugate=conjugate):
            add(Use("op:ctranspose" if conjugate else "op:transpose", node.loc))
        case ir.Index(args=args):
            add(Use("index:paren", node.loc))
            for arg in args:
                uses.extend(_subscript_uses(arg, node, analysis))
            return _dedupe(uses)
        case ir.End():
            add(Use("index:end", node.loc))
        case ir.Call(func=func, local=local, handle=handle):
            if local:
                add(Use("func:call-local", node.loc))
            elif handle:
                add(Use("func:call-handle", node.loc))
            else:
                add(Use(_SPECIAL_CALLS.get(func, f"call:{func}"), node.loc))
        case ir.Lambda():
            add(Use("func:handle", node.loc))
        case ir.FuncRef():
            add(Use("func:named-handle", node.loc))
        case ir.Opaque(construct=construct):
            add(Use(construct, node.loc))
    for child in ir.child_exprs(node):
        uses.extend(expression_uses(child, analysis))
    return _dedupe(uses)


def _dedupe(uses: list[Use]) -> list[Use]:
    seen: set[tuple[str, ir.Loc]] = set()
    unique = []
    for use in uses:
        key = (use.tag, use.loc)
        if key not in seen:
            seen.add(key)
            unique.append(use)
    return unique
