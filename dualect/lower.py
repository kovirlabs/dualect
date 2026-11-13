# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Lowering a tree-sitter ``.m`` syntax tree into the IR (:mod:`dualect.ir`).

Three things happen here that the grammar alone does not do:

* **Name resolution.** ``f(x)`` is an index when ``f`` is a variable and a call
  otherwise. A name counts as a variable in a scope when that scope assigns it
  anywhere (or it is a parameter or output); this is how ``.m`` code is
  written in practice, and it is checked by verification like everything else.
* **Operator precedence.** The grammar nests some operator chains wrongly:
  ``-2 ^ 2`` as ``(-2) ^ 2`` and ``a == b & c`` as ``a == (b & c)``. Every chain
  of prefix, binary and postfix operators is therefore flattened and rebuilt
  here with the documented precedence (see :data:`_BINDING`).
* **Honest gaps.** Constructs the IR does not model become :class:`~ir.Opaque`
  or :class:`~ir.OpaqueStmt` nodes with a construct tag, and text the grammar
  could not parse becomes ``parse:error``. Nothing is silently dropped.
"""

import re
from dataclasses import dataclass, field

from tree_sitter import Node as TSNode

from dualect import ir
from dualect.parse import ParsedSource, parse_m
from dualect.source import DO_MARKER, UNTIL_MARKER

_BINARY = {
    "+": "plus", "-": "minus", "*": "mtimes", ".*": "times", "/": "mrdivide",
    "./": "rdivide", "\\": "mldivide", ".\\": "ldivide", "^": "mpower", ".^": "power",
    "==": "eq", "~=": "ne", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge",
    "&": "and", "|": "or", "&&": "andand", "||": "oror",
}  # fmt: skip
_PREFIX = {"-": "uminus", "+": "uplus", "~": "not", "!": "not"}

#: Binding power of binary operators, loosest first, from the language
#: reference (Octave and MATLAB agree): || < && < | < & < comparisons < : <
#: + - < * / \ < unary prefix < ^ .^ and postfix transposes.
_BINDING = {
    "oror": 1, "andand": 2, "or": 3, "and": 4,
    "eq": 5, "ne": 5, "lt": 5, "le": 5, "gt": 5, "ge": 5,
    "plus": 7, "minus": 7,
    "mtimes": 8, "times": 8, "mrdivide": 8, "rdivide": 8, "mldivide": 8, "ldivide": 8,
    "mpower": 10, "power": 10,
}  # fmt: skip
_PREFIX_BINDING = 9
_POWER_BINDING = 10

_CHAIN_NODES = frozenset(
    {"binary_operator", "comparison_operator", "boolean_operator", "unary_operator",
     "not_operator", "postfix_operator"}
)  # fmt: skip

#: Statement node types whose meaning the IR does not model, with their tags.
_OPAQUE_STATEMENTS = {
    "try_statement": "stmt:try",
    "global_operator": "stmt:global",
    "persistent_operator": "stmt:persistent",
    "class_definition": "stmt:classdef",
    "spmd_statement": "stmt:spmd",
    "arguments_statement": "func:arguments-block",
}

_INT_LITERAL = re.compile(r"^[0-9]+$")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "a": "\a", "0": "\0"}


@dataclass
class _Scope:
    """Names visible as variables while lowering one function or script body.

    :param variables: names that are variables in this scope.
    :param handles: variables that only ever hold function handles.
    """

    variables: set[str]
    handles: set[str] = field(default_factory=set)


@dataclass
class _IndexContext:
    """The innermost index expression whose subscripts are being lowered."""

    target: ir.Expr
    dim: int
    ndims: int


class _Lowerer:
    """Walks one parsed file; see :func:`lower_source` for the entry point."""

    def __init__(self, *, parsed: ParsedSource, interpret_escapes: bool) -> None:
        self.parsed = parsed
        self.interpret_escapes = interpret_escapes
        self.functions: set[str] = set()
        self.scope = _Scope(variables=set())
        self.index_stack: list[_IndexContext] = []
        self.pending_syntax: list[str] = []
        self.pending_comments: list[str] = []

    # --- helpers -----------------------------------------------------------

    def loc(self, node: TSNode) -> ir.Loc:
        return self.parsed.loc(node)

    def text(self, node: TSNode) -> str:
        return self.parsed.text(node)

    def offset(self, node: TSNode) -> int:
        return self.parsed.char_offset(node.start_byte)

    @staticmethod
    def named(node: TSNode) -> list[TSNode]:
        """Named children, skipping comments and continuations (handled separately)."""
        return [c for c in node.named_children if c.type not in ("comment", "line_continuation")]

    def collect_continuations(self, node: TSNode) -> None:
        """Record ``...`` continuations inside a statement and keep their comments."""
        for child in node.children:
            if child.type == "line_continuation":
                self.pending_syntax.append("syntax:continuation")
                remark = self.text(child)[3:].strip()
                if remark:
                    self.pending_comments.append(remark)
            elif child.type == "comment" and child.parent is not None:
                self.pending_comments.append(_comment_text(self.text(child)))
            elif child.child_count and child.type not in ("function_definition",):
                self.collect_continuations(child)

    # --- scopes ------------------------------------------------------------

    def scan_assigned(self, node: TSNode, scope: _Scope) -> None:
        """Collect names assigned anywhere under ``node`` (not in nested functions)."""
        for child in node.named_children:
            if child.type == "function_definition":
                continue
            if child.type == "assignment":
                left = child.child_by_field_name("left")
                right = child.child_by_field_name("right")
                for name in self.target_names(left):
                    scope.variables.add(name)
                    if right is not None and right.type in ("lambda", "handle_operator"):
                        scope.handles.add(name)
                    elif name in scope.handles:
                        scope.handles.discard(name)
            elif child.type == "iterator":
                first = child.named_children[0] if child.named_children else None
                if first is not None and first.type == "identifier":
                    scope.variables.add(self.text(first))
            elif child.type in ("global_operator", "persistent_operator"):
                for ident in child.named_children:
                    if ident.type == "identifier":
                        scope.variables.add(self.text(ident))
            self.scan_assigned(child, scope)

    def target_names(self, left: TSNode | None) -> list[str]:
        if left is None:
            return []
        if left.type == "identifier":
            return [self.text(left)]
        if left.type == "function_call":
            name = left.child_by_field_name("name")
            return [self.text(name)] if name is not None else []
        if left.type == "multioutput_variable":
            names: list[str] = []
            for child in left.named_children:
                names.extend(self.target_names(child))
            return names
        return []

    # --- program and statements --------------------------------------------

    def program(self, root: TSNode, name: str) -> ir.Program:
        for child in root.named_children:
            if child.type == "function_definition":
                fname = child.child_by_field_name("name")
                if fname is not None:
                    self.functions.add(self.text(fname))
        script_scope = _Scope(variables=set())
        self.scan_assigned(root, script_scope)
        self.scope = script_scope
        body = self.block(root)
        _reject_stray_until(body)
        return ir.Program(loc=self.loc(root), body=body, source_name=name)

    def block(self, node: TSNode) -> list[ir.Stmt]:
        """Lower the statements among ``node``'s children, attaching comments."""
        body: list[ir.Stmt] = []
        children = node.children
        last_line = -1
        for index, child in enumerate(children):
            if child.type in ("comment", "line_continuation"):
                if child.type == "line_continuation":
                    continue
                loc = self.loc(child)
                # The grammar merges consecutive comment lines (and a following
                # %{ ... %} block) into one node; split it back into pieces.
                for first, piece in enumerate(_comment_pieces(self.text(child))):
                    line_offset, is_block, text, span = piece
                    line = loc.line + line_offset
                    tags = ["syntax:hash-comment"] if self._is_hash_line(line) else []
                    if (
                        first == 0
                        and body
                        and line == last_line
                        and not isinstance(body[-1], ir.Comment)
                        and not is_block
                    ):
                        previous = body[-1]
                        previous.comment = (
                            text if previous.comment is None else f"{previous.comment}; {text}"
                        )
                        previous.syntax.extend(["syntax:comment", *tags])
                        continue
                    piece_loc = ir.Loc(line, 1 if line_offset else loc.col, line + span - 1, 1)
                    comment = ir.Comment(loc=piece_loc, text=text, block=is_block)
                    comment.syntax.extend(
                        ["syntax:block-comment" if is_block else "syntax:comment", *tags]
                    )
                    body.append(comment)
                    last_line = line + span - 1
                continue
            if not child.is_named:
                continue
            following = children[index + 1] if index + 1 < len(children) else None
            display = following is None or following.type != ";"
            stmts = self.statement(child, display=display)
            body.extend(stmts)
            if stmts:
                last_line = self.loc(child).end_line
        return body

    def _is_hash_line(self, line: int) -> bool:
        """Whether a comment on this original line starts with ``#``."""
        lines = self.parsed.source.original.splitlines()
        if not 1 <= line <= len(lines):
            return False
        text = lines[line - 1]
        stripped = text.lstrip()
        if stripped.startswith("#"):
            return True
        return "#" in text and "%" not in text

    def statement(self, node: TSNode, *, display: bool) -> list[ir.Stmt]:
        """Lower one statement node; may return several (or none)."""
        kind = node.type
        loc = self.loc(node)
        if kind == "block":
            return self.block(node)
        if kind in _OPAQUE_STATEMENTS:
            return [
                ir.OpaqueStmt(loc=loc, construct=_OPAQUE_STATEMENTS[kind], text=self.text(node))
            ]
        if kind == "function_definition":
            return [self.function_definition(node)]
        if kind in ("if_statement", "for_statement", "while_statement", "switch_statement"):
            return [self.compound(node)]
        if node.has_error:
            return [ir.OpaqueStmt(loc=loc, construct="parse:error", text=self.text(node))]
        if kind == "break_statement":
            return [ir.Break(loc=loc)]
        if kind == "continue_statement":
            return [ir.Continue(loc=loc)]
        if kind == "return_statement":
            return [ir.Return(loc=loc)]
        if kind == "command":
            name_node = node.child_by_field_name("name") or node.named_children[0]
            words = [self.text(c) for c in node.named_children if c.type == "command_argument"]
            return [ir.Command(loc=loc, name=self.text(name_node), words=words)]

        self.pending_syntax = []
        self.pending_comments = []
        self.collect_continuations(node)
        stmt = self.simple_statement(node, display=display)
        if display and not isinstance(stmt, ir.OpaqueStmt):
            stmt.syntax.append("syntax:display")
        stmt.syntax.extend(self.pending_syntax)
        if self.pending_comments:
            stmt.comment = "; ".join(self.pending_comments)
        return [stmt]

    def simple_statement(self, node: TSNode, *, display: bool) -> ir.Stmt:
        loc = self.loc(node)
        if node.type == "assignment":
            return self.assignment(node, display=display)
        increment = self._increment(node)
        if increment is not None:
            ident, op = increment
            return ir.AugAssign(
                loc=loc,
                target=ir.Name(loc=self.loc(ident), id=self.text(ident)),
                op=op,
                value=ir.Num(loc=loc, text="1", value=1),
            )
        value = self.expr(node)
        # A lone number statement (`1;`) is how an Octave script declares that it
        # is a script before defining functions: it has no effect.
        if isinstance(value, ir.Num) and not display:
            stmt = ir.ExprStmt(loc=loc, value=value, display=display)
            stmt.syntax.append("syntax:script-marker")
            return stmt
        return ir.ExprStmt(loc=loc, value=value, display=display)

    def _increment(self, node: TSNode) -> tuple[TSNode, str] | None:
        """The identifier and operator of a normalized ``x++`` statement, if it is one."""
        if node.type != "binary_operator":
            return None
        for child in node.named_children:
            if child.type == "identifier":
                op = self.parsed.source.increments.get(self.offset(child))
                if op is not None:
                    return child, op
        return None

    def assignment(self, node: TSNode, *, display: bool) -> ir.Stmt:
        loc = self.loc(node)
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is None or right is None:
            return ir.OpaqueStmt(loc=loc, construct="parse:error", text=self.text(node))
        equals = next((c for c in node.children if c.type == "="), None)
        aug = self.parsed.source.aug_ops.get(self.offset(equals)) if equals is not None else None
        targets = self.targets(left)
        value = self.expr(right)
        if aug is not None:
            return ir.AugAssign(loc=loc, target=targets[0], op=aug, value=value)
        return ir.Assign(loc=loc, targets=targets, value=value, display=display)

    def targets(self, left: TSNode) -> list[ir.Expr]:
        loc = self.loc(left)
        if left.type == "multioutput_variable":
            found: list[ir.Expr] = []
            for child in left.named_children:
                if child.type == "ignored_argument":
                    found.append(ir.Ignore(loc=self.loc(child)))
                else:
                    found.extend(self.targets(child))
            return found
        if left.type == "identifier":
            return [ir.Name(loc=loc, id=self.text(left))]
        if left.type == "function_call" and not self._is_brace(left):
            return [self.index(left, force_index=True)]
        if left.type == "field_expression":
            return [ir.Opaque(loc=loc, construct="expr:field", text=self.text(left))]
        if left.type == "function_call":
            return [ir.Opaque(loc=loc, construct="expr:cell-index", text=self.text(left))]
        return [ir.Opaque(loc=loc, construct="parse:error", text=self.text(left))]

    def function_definition(self, node: TSNode) -> ir.Stmt:
        loc = self.loc(node)
        name_node = node.child_by_field_name("name")
        name = self.text(name_node) if name_node is not None else "<anonymous>"
        outputs: list[str] = []
        params: list[str] = []
        body_node: TSNode | None = None
        for child in node.named_children:
            if child.type == "function_output":
                for out in child.named_children:
                    if out.type == "identifier":
                        outputs.append(self.text(out))
                    elif out.type == "multioutput_variable":
                        outputs.extend(
                            self.text(c) for c in out.named_children if c.type == "identifier"
                        )
            elif child.type == "function_arguments":
                for arg in child.named_children:
                    if arg.type == "identifier":
                        params.append(self.text(arg))
                    elif arg.type == "ignored_argument":
                        params.append("~")
            elif child.type == "block":
                body_node = child
        header_errors = any(
            c.has_error
            for c in node.named_children
            if c.type in ("function_output", "function_arguments")
        )
        if header_errors:
            return ir.OpaqueStmt(loc=loc, construct="parse:error", text=self.text(node))
        outer = self.scope
        scope = _Scope(variables={*params, *outputs})
        if body_node is not None:
            self.scan_assigned(body_node, scope)
        self.scope = scope
        try:
            body = self.block(body_node) if body_node is not None else []
        finally:
            self.scope = outer
        stmt = ir.FunctionDef(loc=loc, name=name, params=params, outputs=outputs, body=body)
        # Comments and blank lines between the header and the first statement
        # belong to the body; nothing to do here.
        return stmt

    def compound(self, node: TSNode) -> ir.Stmt:
        stmt = self._compound(node)
        # A comment on the header line (`while x  % why`) is a direct child.
        remarks = [_comment_text(self.text(c)) for c in node.children if c.type == "comment"]
        if remarks and not isinstance(stmt, ir.OpaqueStmt):
            stmt.comment = "; ".join(r for r in [stmt.comment, *remarks] if r)
            stmt.syntax.append("syntax:comment")
        return stmt

    def _compound(self, node: TSNode) -> ir.Stmt:
        loc = self.loc(node)
        kind = node.type
        header = [c for c in node.named_children if c.type not in ("block", "comment")]
        if kind == "if_statement":
            cond = node.child_by_field_name("condition")
            if cond is None or cond.has_error:
                return ir.OpaqueStmt(loc=loc, construct="parse:error", text=self.text(node))
            branches = [(self.expr(cond), self.first_block(node))]
            orelse: list[ir.Stmt] | None = None
            for clause in node.named_children:
                if clause.type == "elseif_clause":
                    clause_cond = clause.child_by_field_name("condition")
                    if clause_cond is None or clause_cond.has_error:
                        return ir.OpaqueStmt(loc=loc, construct="parse:error", text=self.text(node))
                    branches.append((self.expr(clause_cond), self.first_block(clause)))
                elif clause.type == "else_clause":
                    orelse = self.first_block(clause)
            return ir.If(loc=loc, branches=branches, orelse=orelse)
        if kind == "for_statement":
            iterator = next((c for c in header if c.type == "iterator"), None)
            is_parfor = any(c.type == "parfor" for c in node.children)
            if iterator is None or iterator.has_error or is_parfor:
                construct = "stmt:parfor" if is_parfor else "parse:error"
                return ir.OpaqueStmt(loc=loc, construct=construct, text=self.text(node))
            parts = self.named(iterator)
            var = parts[0]
            iterable = self.expr(parts[-1])
            return ir.For(
                loc=loc,
                var=ir.Name(loc=self.loc(var), id=self.text(var)),
                iterable=iterable,
                body=self.first_block(node),
            )
        if kind == "while_statement":
            cond = node.child_by_field_name("condition")
            if cond is None or cond.has_error:
                return ir.OpaqueStmt(loc=loc, construct="parse:error", text=self.text(node))
            body = self.first_block(node)
            if cond.type == "identifier" and self.text(cond) == DO_MARKER:
                return self.do_until(node, body)
            return ir.While(loc=loc, cond=self.expr(cond), body=body)
        # switch_statement
        subject_node = node.child_by_field_name("condition")
        if subject_node is None or subject_node.has_error:
            return ir.OpaqueStmt(loc=loc, construct="parse:error", text=self.text(node))
        cases: list[tuple[list[ir.Expr], list[ir.Stmt]]] = []
        otherwise: list[ir.Stmt] | None = None
        for clause in node.named_children:
            if clause.type == "case_clause":
                value_node = clause.child_by_field_name("condition")
                if value_node is None or value_node.has_error:
                    return ir.OpaqueStmt(loc=loc, construct="parse:error", text=self.text(node))
                value = self.expr(value_node)
                values = value.items if isinstance(value, ir.CellLit) else [value]
                cases.append((values, self.first_block(clause)))
            elif clause.type == "otherwise_clause":
                otherwise = self.first_block(clause)
        return ir.Switch(loc=loc, subject=self.expr(subject_node), cases=cases, otherwise=otherwise)

    def first_block(self, node: TSNode) -> list[ir.Stmt]:
        """The statements of a clause: its ``block`` child, if any."""
        block = next((c for c in node.named_children if c.type == "block"), None)
        body = self.block(block) if block is not None else []
        # Comments that sit directly in the clause (an otherwise empty branch).
        for child in node.named_children:
            if child.type == "comment" and block is None:
                body.append(ir.Comment(loc=self.loc(child), text=_comment_text(self.text(child))))
        return body

    def do_until(self, node: TSNode, body: list[ir.Stmt]) -> ir.Stmt:
        loc = self.loc(node)
        last = next((s for s in reversed(body) if not isinstance(s, ir.Comment)), None)
        if (
            isinstance(last, ir.Assign)
            and len(last.targets) == 1
            and isinstance(last.targets[0], ir.Name)
            and last.targets[0].id == UNTIL_MARKER
        ):
            index = body.index(last)
            stmt = ir.DoUntil(loc=loc, body=body[:index] + body[index + 1 :], cond=last.value)
            stmt.comment = last.comment
            return stmt
        return ir.OpaqueStmt(loc=loc, construct="parse:error", text=self.text(node))

    # --- expressions -------------------------------------------------------

    def expr(self, node: TSNode) -> ir.Expr:
        """Lower an expression node."""
        loc = self.loc(node)
        kind = node.type
        if node.has_error or node.is_missing:
            return ir.Opaque(loc=loc, construct="parse:error", text=self.text(node))
        if kind == "_expression" and len(self.named(node)) == 1:
            # A wrapper the grammar sometimes leaves in place; it means nothing.
            return self.expr(self.named(node)[0])
        if kind in _CHAIN_NODES:
            return self.chain(node)
        if kind == "parenthesis":
            inner = self.named(node)
            if len(inner) != 1:
                return ir.Opaque(loc=loc, construct="parse:error", text=self.text(node))
            return self.expr(inner[0])
        if kind == "number":
            return _number(self.text(node), loc)
        if kind == "string":
            return self.string(node)
        if kind == "identifier":
            return self.identifier(node)
        if kind == "end_keyword":
            if not self.index_stack:
                return ir.Opaque(loc=loc, construct="expr:end-outside-index", text="end")
            context = self.index_stack[-1]
            return ir.End(loc=loc, target=context.target, dim=context.dim, ndims=context.ndims)
        if kind == "spread_operator":
            return ir.Colon(loc=loc)
        if kind == "range":
            parts = [self.expr(c) for c in self.named(node)]
            if len(parts) == 2:
                return ir.Range(loc=loc, start=parts[0], stop=parts[1])
            if len(parts) == 3:
                return ir.Range(loc=loc, start=parts[0], step=parts[1], stop=parts[2])
            return ir.Opaque(loc=loc, construct="parse:error", text=self.text(node))
        if kind == "matrix":
            rows = []
            for row in self.named(node):
                items = [self.expr(c) for c in self.named(row)]
                if items:
                    rows.append(items)
            return ir.Matrix(loc=loc, rows=rows)
        if kind == "cell":
            items = [self.expr(c) for row in self.named(node) for c in self.named(row)]
            return ir.CellLit(loc=loc, items=items)
        if kind == "function_call":
            if self._is_brace(node):
                return ir.Opaque(loc=loc, construct="expr:cell-index", text=self.text(node))
            return self.index(node)
        if kind == "lambda":
            return self.lambda_(node)
        if kind == "handle_operator":
            target = self.named(node)
            if len(target) == 1 and target[0].type == "identifier":
                return ir.FuncRef(loc=loc, name=self.text(target[0]))
            return ir.Opaque(loc=loc, construct="func:handle-expression", text=self.text(node))
        if kind == "field_expression":
            return ir.Opaque(loc=loc, construct="expr:field", text=self.text(node))
        if kind == "boolean":
            return ir.Call(loc=loc, func=self.text(node), args=[])
        return ir.Opaque(loc=loc, construct=f"expr:{kind}", text=self.text(node))

    def string(self, node: TSNode) -> ir.Expr:
        raw = self.text(node)
        loc = self.loc(node)
        quote = raw[0]
        body = raw[1:-1]
        if quote == "'":
            return ir.Str(loc=loc, value=body.replace("''", "'"), quote="'")
        if self.interpret_escapes:
            # Octave reads backslash escapes in double-quoted strings.
            value = re.sub(
                r"\\(.)", lambda m: _ESCAPES.get(m.group(1), "\\" + m.group(1)), body
            ).replace('""', '"')
        else:
            value = body.replace('""', '"')
        return ir.Str(loc=loc, value=value, quote='"')

    def identifier(self, node: TSNode) -> ir.Expr:
        name = self.text(node)
        loc = self.loc(node)
        if name in self.scope.variables:
            return ir.Name(loc=loc, id=name)
        return ir.Call(loc=loc, func=name, args=[], local=name in self.functions)

    @staticmethod
    def _is_brace(node: TSNode) -> bool:
        return any(c.type == "{" for c in node.children)

    def index(self, node: TSNode, *, force_index: bool = False) -> ir.Expr:
        """Lower ``name(args)``: an index of a variable, else a function call."""
        loc = self.loc(node)
        name_node = node.child_by_field_name("name")
        if name_node is None or name_node.type != "identifier":
            return ir.Opaque(loc=loc, construct="expr:call-expression", text=self.text(node))
        name = self.text(name_node)
        arg_nodes: list[TSNode] = []
        for child in node.named_children:
            if child.type == "arguments":
                arg_nodes = self.named(child)
        is_variable = force_index or name in self.scope.variables
        if is_variable and name not in self.scope.handles:
            target = ir.Name(loc=self.loc(name_node), id=name)
            args: list[ir.Expr] = []
            for dim, arg in enumerate(arg_nodes):
                self.index_stack.append(_IndexContext(target=target, dim=dim, ndims=len(arg_nodes)))
                try:
                    args.append(self.expr(arg))
                finally:
                    self.index_stack.pop()
            return ir.Index(loc=loc, target=target, args=args)
        # `end` inside a call's arguments would bind to an enclosing index (if any)
        # in the language; keep the stack as it is.
        call_args = [self.expr(arg) for arg in arg_nodes]
        return ir.Call(
            loc=loc,
            func=name,
            args=call_args,
            local=name in self.functions,
            handle=name in self.scope.handles,
        )

    def lambda_(self, node: TSNode) -> ir.Expr:
        loc = self.loc(node)
        params: list[str] = []
        body_node = node.child_by_field_name("expression")
        for child in node.named_children:
            if child.type == "arguments":
                params = [self.text(c) for c in child.named_children if c.type == "identifier"]
        if body_node is None:
            return ir.Opaque(loc=loc, construct="parse:error", text=self.text(node))
        outer = self.scope
        self.scope = _Scope(
            variables=outer.variables | set(params), handles=outer.handles - set(params)
        )
        try:
            body = self.expr(body_node)
        finally:
            self.scope = outer
        return ir.Lambda(loc=loc, params=params, body=body)

    # --- operator chains ---------------------------------------------------

    def flatten(self, node: TSNode, tokens: list[tuple[str, object, ir.Loc]]) -> None:
        """Flatten an operator chain into tokens: atoms, prefix, binary, postfix ops."""
        kind = node.type
        if kind == "_expression" and len(self.named(node)) == 1:
            self.flatten(self.named(node)[0], tokens)
            return
        if kind not in _CHAIN_NODES or node.has_error:
            tokens.append(("atom", self.expr(node), self.loc(node)))
            return
        if kind in ("unary_operator", "not_operator"):
            op_token = next(c for c in node.children if not c.is_named)
            tokens.append(("pre", _PREFIX[self.text(op_token)], self.loc(op_token)))
            for child in self.named(node):
                self.flatten(child, tokens)
            return
        if kind == "postfix_operator":
            for child in self.named(node):
                self.flatten(child, tokens)
            op_token = next(c for c in reversed(node.children) if not c.is_named)
            tokens.append(("post", self.text(op_token) == "'", self.loc(op_token)))
            return
        for child in node.children:
            if child.type in ("comment", "line_continuation"):
                continue
            if child.is_named:
                self.flatten(child, tokens)
            else:
                tokens.append(("bin", _BINARY[self.text(child)], self.loc(child)))

    def chain(self, node: TSNode) -> ir.Expr:
        tokens: list[tuple[str, object, ir.Loc]] = []
        try:
            self.flatten(node, tokens)
        except (KeyError, StopIteration):
            return ir.Opaque(loc=self.loc(node), construct="parse:error", text=self.text(node))
        parser = _PrattParser(tokens)
        result = parser.expression(0)
        if parser.position != len(tokens):
            return ir.Opaque(loc=self.loc(node), construct="parse:error", text=self.text(node))
        return result


class _PrattParser:
    """Rebuilds an operator chain with the language's precedence (see :data:`_BINDING`)."""

    def __init__(self, tokens: list[tuple[str, object, ir.Loc]]) -> None:
        self.tokens = tokens
        self.position = 0

    def peek(self) -> tuple[str, object, ir.Loc] | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> tuple[str, object, ir.Loc]:
        token = self.tokens[self.position]
        self.position += 1
        return token

    def expression(self, min_binding: int) -> ir.Expr:
        left = self.prefix()
        while (token := self.peek()) is not None:
            kind, value, loc = token
            if kind == "post":
                if min_binding > _POWER_BINDING:
                    break
                self.take()
                left = ir.Transpose(loc=_span(left.loc, loc), operand=left, conjugate=bool(value))
                continue
            if kind != "bin":
                break
            op = str(value)
            binding = _BINDING[op]
            if binding < min_binding:
                break
            self.take()
            # All binary operators are left-associative; a power's exponent may
            # carry its own prefix operators (`2^-1`), which bind only to it.
            right = (
                self.power_operand() if binding == _POWER_BINDING else self.expression(binding + 1)
            )
            left = ir.BinOp(loc=_span(left.loc, right.loc), op=op, left=left, right=right)
        return left

    def prefix(self) -> ir.Expr:
        kind, value, loc = self.take()
        if kind == "pre":
            operand = self.expression(_PREFIX_BINDING)
            return ir.UnOp(loc=_span(loc, operand.loc), op=str(value), operand=operand)
        if kind != "atom" or not isinstance(value, ir.Expr):
            raise ValueError("malformed operator chain")
        return value

    def power_operand(self) -> ir.Expr:
        kind, value, loc = self.take()
        if kind == "pre":
            operand = self.power_operand()
            return ir.UnOp(loc=_span(loc, operand.loc), op=str(value), operand=operand)
        if kind != "atom" or not isinstance(value, ir.Expr):
            raise ValueError("malformed operator chain")
        return value


def _reject_stray_until(body: list[ir.Stmt]) -> None:
    """Turn an ``until`` that closed no ``do`` into a parse error, in place."""
    for index, stmt in enumerate(body):
        if (
            isinstance(stmt, ir.Assign)
            and isinstance(stmt.targets[0], ir.Name)
            and stmt.targets[0].id == UNTIL_MARKER
        ):
            body[index] = ir.OpaqueStmt(loc=stmt.loc, construct="parse:error", text="until ...")
        for inner in ir.stmt_bodies(stmt):
            _reject_stray_until(inner)


def _span(first: ir.Loc, last: ir.Loc) -> ir.Loc:
    return ir.Loc(first.line, first.col, last.end_line, last.end_col)


def _number(text: str, loc: ir.Loc) -> ir.Expr:
    """Parse a numeric literal; imaginary literals end in ``i`` or ``j``."""
    body = text
    imaginary = body[-1] in "ijIJ" and not body.lower().startswith("0x")
    if imaginary:
        body = body[:-1]
    try:
        if body.lower().startswith("0x"):
            value: int | float = int(body, 16)
        elif _INT_LITERAL.match(body):
            value = int(body)
        else:
            value = float(body)
    except ValueError:
        return ir.Opaque(loc=loc, construct="literal:number-format", text=text)
    if imaginary:
        return ir.Num(loc=loc, text=text, value=complex(0, value))
    return ir.Num(loc=loc, text=text, value=value)


def _comment_text(text: str) -> str:
    """Comment text without its marker; ``%%`` cell dividers keep theirs."""
    stripped = text.strip()
    if stripped.startswith("%%"):
        return stripped
    body = stripped[1:]
    return body[1:] if body.startswith(" ") else body


def _comment_pieces(text: str) -> list[tuple[int, bool, str, int]]:
    """Split a comment node into ``(line offset, is block, text, line count)`` pieces."""
    lines = text.split("\n")
    pieces: list[tuple[int, bool, str, int]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "%{":
            depth, end = 1, index + 1
            while end < len(lines) and depth:
                marker = lines[end].strip()
                depth += 1 if marker == "%{" else -1 if marker == "%}" else 0
                end += 1
            inner = [line.strip() for line in lines[index + 1 : end - 1]]
            pieces.append((index, True, "\n".join(inner), end - index))
            index = end
            continue
        if stripped:
            pieces.append((index, False, _comment_text(stripped), 1))
        index += 1
    return pieces


def lower_source(*, text: str, name: str = "<source>", dialect: str = "octave") -> ir.Program:
    """Parse and lower ``.m`` source text into an :class:`ir.Program`.

    :param text: the source.
    :param name: file name for reports.
    :param dialect: ``"octave"`` (double-quoted strings read backslash escapes)
        or ``"matlab"`` (they do not).
    """
    parsed = parse_m(text)
    lowerer = _Lowerer(parsed=parsed, interpret_escapes=dialect == "octave")
    return lowerer.program(parsed.tree.root_node, name)


def lower_parsed(*, parsed: ParsedSource, name: str, dialect: str = "octave") -> ir.Program:
    """Lower an already parsed file (see :func:`lower_source`).

    :param parsed: output of :func:`dualect.parse.parse_m`.
    :param name: file name for reports.
    :param dialect: source dialect.
    """
    return _Lowerer(parsed=parsed, interpret_escapes=dialect == "octave").program(
        parsed.tree.root_node, name
    )
