# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/num2str: numbers as text.

``num2str`` picks its format from the value: a whole number prints all its
digits (``f"{x:.0f}"``), any other number gets ``max(floor(log10(|x|)) + 5,
5)`` significant digits, which depends on the magnitude and so has no fixed
Python spelling: that form is declined. ``num2str(x, prec)`` is
``f"{x:.{prec}g}"`` and ``num2str(x, fmt)`` is the ``sprintf`` of ``fmt``
(rule io/printf) with surrounding blanks trimmed. Whole numbers of 1e15 or
more switch to exponent notation in the source, so a value not known to be
smaller is assumed so, out loud. Arrays, which print as padded rows, are
declined.

From Python, ``str(n)`` of an integer is ``num2str(n)`` (``str`` of a float
uses Python's shortest repr, which no ``num2str`` form reproduces), and
``s.strip()`` of a string is ``strtrim(s)``; ``int(x)`` truncates toward
zero, ``fix(x)``.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind

_LIMIT = 1e15


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``num2str(x)``, ``num2str(x, prec)`` or ``num2str(x, fmt)``."""
    if len(node.args) not in (1, 2):
        ctx.decline("num2str with these arguments")
    info = ctx.info(node.args[0])
    if info.kind is Kind.STRING:
        if len(node.args) == 2:
            ctx.decline("num2str of a string with a format")
        return ctx.expr(node.args[0])  # text passes through unchanged
    if not info.is_scalar:
        ctx.decline("num2str of an array (it prints padded rows)")
    if info.elem not in (Elem.REAL, Elem.LOGICAL):
        ctx.decline("num2str of a complex value or a value of unknown type")
    value = info.value
    if isinstance(value, int | float) and value != value or value in (float("inf"), -float("inf")):
        ctx.decline("num2str of Inf or NaN")
    if len(node.args) == 2:
        spec = node.args[1]
        if isinstance(spec, ir.Str):
            if "\\n" in spec.value or "\n" in spec.value:
                ctx.decline("num2str with a format that prints several rows")
            printf = ctx.module("io/printf")
            return ctx.method(printf.format_text(spec.value, [node.args[0]], ctx), "strip")
        digits = ctx.info(spec).value
        if not isinstance(digits, int | float) or not float(digits).is_integer() or digits < 0:
            ctx.decline("num2str with a precision that is not a literal whole number")
        if not info.finite:
            ctx.assume("a finite value (Octave prints Inf/NaN, Python inf/nan)")
        field = ast.FormattedValue(
            value=ctx.expr(node.args[0]),
            conversion=-1,
            format_spec=ast.JoinedStr(values=[ast.Constant(f".{int(digits)}g")]),
        )
        return ast.JoinedStr(values=[field])
    if not info.integer:
        ctx.decline(
            "num2str of a value not known to be a whole number "
            "(its significant digits depend on the magnitude; give a precision)"
        )
    if isinstance(value, int | float):
        if abs(value) >= _LIMIT:
            ctx.decline("num2str of a whole number of 1e15 or more (exponent notation)")
    elif not info.is_logical:
        ctx.assume("|value| < 1e15 in num2str (larger whole numbers print with an exponent)")
    # f"{x:.0f}": every digit of a whole number, int or float, and the sign of
    # a negative zero (`num2str(-0)` is '-0'; str(int(-0.0)) would lose it).
    field = ast.FormattedValue(
        value=ctx.expr(node.args[0]),
        conversion=-1,
        format_spec=ast.JoinedStr(values=[ast.Constant(".0f")]),
    )
    return ast.JoinedStr(values=[field])


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """A char row vector."""
    return Info(kind=Kind.STRING, rows=1, elem=Elem.CHAR)


# --- Python -> .m ------------------------------------------------------------


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``str(n)`` of an integer, ``s.strip()`` of a string."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    if node.keywords or len(node.args) > 1:
        ctx.decline("str()/strip() with these arguments")
    if isinstance(node.func, ast.Name) and node.func.id == "int":
        # int(x) truncates toward zero, as fix(x) does (a double on the .m side).
        if len(node.args) != 1:
            ctx.decline("int() without one argument")
        inner = ctx.info(node.args[0])
        if not inner.is_scalar or inner.dtype not in (DType.INT, DType.FLOAT, DType.BOOL):
            ctx.decline("int() of a value that is not a real number")
        return ctx.call("fix", ctx.expr(node.args[0]), loc=loc)
    if isinstance(node.func, ast.Attribute):  # s.strip()
        if node.args or ctx.info(node.func.value).kind is not PyKind.STRING:
            ctx.decline("strip() of a value that is not a string, or with characters")
        return ctx.call("strtrim", ctx.expr(node.func.value), loc=loc)
    if len(node.args) != 1:
        ctx.decline("str() without an argument")
    arg = node.args[0]
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "int":
        # str(int(x)): the digits of x truncated, which fix(x) gives num2str.
        if len(arg.args) != 1 or arg.keywords:
            ctx.decline("int() with these arguments")
        inner = ctx.info(arg.args[0])
        if not inner.is_scalar or inner.dtype not in (DType.INT, DType.FLOAT, DType.BOOL):
            ctx.decline("str(int(x)) of a value that is not a real number")
        if inner.dtype is not DType.BOOL:
            ctx.assume("|value| < 1e15 (num2str prints larger whole numbers with an exponent)")
        value = ctx.expr(arg.args[0])
        if inner.dtype is DType.FLOAT:
            value = ctx.call("fix", value, loc=loc)
        return ctx.call("num2str", value, loc=loc)
    info = ctx.info(arg)
    if info.kind is PyKind.STRING:
        return ctx.expr(arg)
    if not info.is_scalar or info.dtype is not DType.INT:
        ctx.decline("str() of a value that is not an integer (Python's repr has no .m form)")
    ctx.assume("|value| < 1e15 (num2str prints larger whole numbers with an exponent)")
    return ctx.call("num2str", ctx.expr(arg), loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """``str(...)`` and ``s.strip()`` are strings."""
    if tag == "py:call:str" or (receiver is not None and receiver.kind is PyKind.STRING):
        return PyInfo(kind=PyKind.STRING)
    if tag == "py:call:int" and len(args) == 1:
        return PyInfo(kind=PyKind.SCALAR, dtype=DType.INT, finite=True, nonneg=args[0].nonneg)
    return None
