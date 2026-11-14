# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Printing the IR as ``.m`` source (the Python -> ``.m`` direction, step 5).

By the time a program reaches this module it already has ``.m`` semantics:
the lifting step (:mod:`dualect.lift`, with each rule's ``lift``) produced
1-based :class:`~dualect.ir.Index` subscripts, inclusive ranges, ``mtimes``
for the matrix product and transposes where a 1-D NumPy array must be seen as
a column. What is left here is surface syntax: precedence and parentheses,
statement layout, and comments.

The output avoids anything newer than Octave 8 and MATLAB R2016b: every
statement ends with ``;``, functions are defined before use in a script that
starts with ``1;``, and matrix elements are separated by commas, so
``[1, -2]`` is never read as ``[1 - 2]``.
"""

from dualect import ir

INDENT = "  "

#: Binding power, loosest first (as in :mod:`dualect.lower`).
_BINDING = {
    "oror": 1, "andand": 2, "or": 3, "and": 4,
    "eq": 5, "ne": 5, "lt": 5, "le": 5, "gt": 5, "ge": 5,
    "range": 6,
    "plus": 7, "minus": 7,
    "mtimes": 8, "times": 8, "mrdivide": 8, "rdivide": 8, "mldivide": 8, "ldivide": 8,
    "unary": 9,
    "mpower": 10, "power": 10, "postfix": 10,
}  # fmt: skip
_ATOM = 11

_SYMBOLS = {
    "plus": "+", "minus": "-", "mtimes": "*", "times": ".*", "mrdivide": "/",
    "rdivide": "./", "mldivide": "\\", "ldivide": ".\\", "mpower": "^", "power": ".^",
    "eq": "==", "ne": "~=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
    "and": "&", "or": "|", "andand": "&&", "oror": "||",
}  # fmt: skip
_PREFIX = {"uminus": "-", "uplus": "+", "not": "~"}

#: Construct of an :class:`~dualect.ir.OpaqueStmt` whose text is printed verbatim
#: (an untranslated statement: its TODO comment and the ``error`` call).
HOLE = "dualect:hole"


def _binding(node: ir.Expr) -> int:
    match node:
        case ir.BinOp(op=op):
            return _BINDING[op]
        case ir.UnOp():
            return _BINDING["unary"]
        case ir.Transpose():
            return _BINDING["postfix"]
        case ir.Range():
            return _BINDING["range"]
        case ir.Lambda():
            return 0
        case ir.Num(value=value) if isinstance(value, int | float) and value < 0:
            return _BINDING["unary"]
    return _ATOM


def _wrap(node: ir.Expr, minimum: int) -> str:
    text = expr(node)
    return f"({text})" if _binding(node) < minimum else text


def number(node: ir.Num) -> str:
    """A numeric literal in ``.m`` spelling (``1i`` for an imaginary unit)."""
    value = node.value
    if isinstance(value, complex):
        if value.real == 0:
            return f"{_real(value.imag)}i"
        return f"({_real(value.real)} + {_real(value.imag)}i)"
    if node.text and not node.text.endswith(("j", "J")):
        return node.text
    return _real(value)


def _real(value: float) -> str:
    if isinstance(value, int) or float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def string(value: str) -> str:
    """A single-quoted char array literal."""
    return "'" + value.replace("'", "''") + "'"


def expr(node: ir.Expr) -> str:
    """Print one expression."""
    match node:
        case ir.Num():
            return number(node)
        case ir.Str(value=value):
            return string(value)
        case ir.Name(id=name):
            return name
        case ir.Colon():
            return ":"
        case ir.End():
            return "end"
        case ir.Ignore():
            return "~"
        case ir.Range(start=start, stop=stop, step=step):
            parts = [start, step, stop] if step is not None else [start, stop]
            return ":".join(_wrap(p, _BINDING["range"] + 1) for p in parts)
        case ir.Matrix(rows=rows):
            return "[" + "; ".join(", ".join(expr(item) for item in row) for row in rows) + "]"
        case ir.CellLit(items=items):
            return "{" + ", ".join(expr(item) for item in items) + "}"
        case ir.BinOp(op=op, left=left, right=right):
            binding = _BINDING[op]
            if op in ("mpower", "power"):
                # A prefix operator binds looser than a power: (-2)^2.
                left_text = _wrap(left, binding + 1)
                right_text = _wrap(right, _ATOM if isinstance(right, ir.BinOp) else binding)
            else:
                left_text = _wrap(left, binding)
                right_text = _wrap(right, binding + 1)
            return f"{left_text} {_SYMBOLS[op]} {right_text}"
        case ir.UnOp(op=op, operand=operand):
            return f"{_PREFIX[op]}{_wrap(operand, _BINDING['unary'])}"
        case ir.Transpose(operand=operand, conjugate=conjugate):
            mark = "'" if conjugate else ".'"
            return f"{_wrap(operand, _ATOM)}{mark}"
        case ir.Index(target=target, args=args):
            return f"{_wrap(target, _ATOM)}({', '.join(expr(a) for a in args)})"
        case ir.Call(func=func, args=args):
            if not args:
                return func
            return f"{func}({', '.join(expr(a) for a in args)})"
        case ir.Lambda(params=params, body=body):
            return f"@({', '.join(params)}) {expr(body)}"
        case ir.FuncRef(name=name):
            return f"@{name}"
        case ir.Opaque(text=text):
            return text
    raise ValueError(f"cannot print {type(node).__name__} as .m code")


def statements(body: list[ir.Stmt], depth: int = 0) -> list[str]:
    """Print a statement list at a nesting depth."""
    lines: list[str] = []
    for stmt in body:
        lines.extend(statement(stmt, depth))
    return lines


def _with_comment(line: str, stmt: ir.Stmt) -> str:
    return f"{line}  % {stmt.comment}" if stmt.comment else line


def statement(stmt: ir.Stmt, depth: int = 0) -> list[str]:
    """Print one statement (several lines for a compound one)."""
    pad = INDENT * depth
    match stmt:
        case ir.Comment(text=text):
            return [f"{pad}% {line}".rstrip() for line in text.split("\n")]
        case ir.OpaqueStmt(construct=construct, text=text) if construct == HOLE:
            return [f"{pad}{line}" for line in text.split("\n")]
        case ir.Assign(targets=targets, value=value):
            if len(targets) == 1:
                lhs = expr(targets[0])
            else:
                lhs = "[" + ", ".join(expr(t) for t in targets) + "]"
            return [_with_comment(f"{pad}{lhs} = {expr(value)};", stmt)]
        case ir.ExprStmt(value=value):
            return [_with_comment(f"{pad}{expr(value)};", stmt)]
        case ir.If(branches=branches, orelse=orelse):
            lines = []
            for k, (cond, body) in enumerate(branches):
                keyword = "if" if k == 0 else "elseif"
                header = f"{pad}{keyword} {expr(cond)}"
                lines.append(_with_comment(header, stmt) if k == 0 else header)
                lines.extend(statements(body, depth + 1))
            if orelse:
                lines.append(f"{pad}else")
                lines.extend(statements(orelse, depth + 1))
            lines.append(f"{pad}end")
            return lines
        case ir.For(var=var, iterable=iterable, body=body):
            header = _with_comment(f"{pad}for {var.id} = {expr(iterable)}", stmt)
            return [header, *statements(body, depth + 1), f"{pad}end"]
        case ir.While(cond=cond, body=body):
            header = _with_comment(f"{pad}while {expr(cond)}", stmt)
            return [header, *statements(body, depth + 1), f"{pad}end"]
        case ir.Switch(subject=subject, cases=cases, otherwise=otherwise):
            lines = [_with_comment(f"{pad}switch {expr(subject)}", stmt)]
            for values, body in cases:
                label = expr(values[0]) if len(values) == 1 else expr(ir.CellLit(ir.NO_LOC, values))
                lines.append(f"{pad}{INDENT}case {label}")
                lines.extend(statements(body, depth + 2))
            if otherwise is not None:
                lines.append(f"{pad}{INDENT}otherwise")
                lines.extend(statements(otherwise, depth + 2))
            lines.append(f"{pad}end")
            return lines
        case ir.Break():
            return [f"{pad}break;"]
        case ir.Continue():
            return [f"{pad}continue;"]
        case ir.Return():
            return [f"{pad}return;"]
        case ir.FunctionDef(name=name, params=params, outputs=outputs, body=body):
            if not outputs:
                out = ""
            elif len(outputs) == 1:
                out = f"{outputs[0]} = "
            else:
                out = f"[{', '.join(outputs)}] = "
            header = _with_comment(f"{pad}function {out}{name}({', '.join(params)})", stmt)
            return [header, *statements(body, depth + 1), f"{pad}end"]
    raise ValueError(f"cannot print {type(stmt).__name__} as .m code")


def program(body: list[ir.Stmt]) -> str:
    """Print a whole script; ``1;`` first when it defines functions."""
    lines: list[str] = []
    if any(isinstance(s, ir.FunctionDef) for s in body):
        # Octave reads a file whose first statement is `function` as a function
        # file; `1;` makes it a script that may define functions before use.
        lines.append("1;")
    previous: ir.Stmt | None = None
    for stmt in body:
        separate = isinstance(stmt, ir.FunctionDef) or isinstance(previous, ir.FunctionDef)
        gap = previous is not None and stmt.loc.line > previous.loc.end_line + 1
        if lines and (separate or gap) and lines[-1] != "":
            lines.append("")
        lines.extend(statement(stmt))
        previous = stmt
    return "\n".join(lines) + "\n"
