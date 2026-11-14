# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for io/printf: ``printf``, ``fprintf``, ``sprintf`` and ``disp``.

A literal format becomes an f-string. The format is read at translation time:
its escapes (``\\n``, ``\\t``, ...) are resolved, each conversion is mapped to
a Python format spec, and the source's *recycling* is carried out statically:
the format is repeated while values remain and output stops at the first
conversion that has no value left (``fprintf('%d %d\\n', 1, 2, 3)`` prints
``1 2`` and ``3 ``). A single array argument recycles the format once per
element (column-major) or per column; that becomes a ``for`` loop (or a
``"".join`` for ``sprintf``).

Declined, by name: conversions without an exact Python counterpart (``%x``,
``%c``, ``%u``, ``*`` widths, ``%.3d``), ``%d`` of a value not known to be a
whole number (Octave silently switches to a ``%f``/``%g`` style there), ``%s``
of a number, a format that is not a literal, a file id other than 1, and
``disp`` of a number (its display format is not Python's ``str``). Values not
known to be finite are assumed finite out loud: Octave prints ``Inf``/``NaN``
where Python prints ``inf``/``nan``. Every translation that prints asks
verification to compare standard output (``# dualect: stdout``).
"""

import ast
import math
import re
from dataclasses import dataclass

from dualect import ir
from dualect.emit import Context, Line
from dualect.infer import Elem, Info, Kind
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyKind

# --- the format string -------------------------------------------------------

_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", "a": "\a", "r": "\r", '"': '"', "'": "'"}
_CONVERSION = re.compile(
    r"%(?P<flags>[-+ 0#]*)(?P<width>\d+|\*)?(?:\.(?P<precision>\d*|\*))?(?P<type>[A-Za-z%])"
)
_NUMERIC = frozenset("dieEfFgG")


@dataclass(frozen=True)
class Conversion:
    """One ``%`` conversion of a format.

    :param flags: the flag characters (``-+ 0#``).
    :param width: minimum field width, or ``None``.
    :param precision: digits after the point (or significant digits), or ``None``.
    :param kind: the conversion letter.
    """

    flags: str
    width: int | None
    precision: int | None
    kind: str


Piece = str | Conversion


def parse_format(text: str) -> list[Piece]:
    """Split a printf format into literal text and conversions.

    Escapes are resolved first, as printf does for single-quoted formats.

    :raises ValueError: for an escape or conversion with no exact Python form.
    """
    unescaped = re.sub(r"\\(.)", lambda m: _escape(m.group(1)), text)
    pieces: list[Piece] = []
    position = 0
    for match in _CONVERSION.finditer(unescaped):
        if match.start() > position:
            pieces.append(unescaped[position : match.start()])
        position = match.end()
        kind = match.group("type")
        if kind == "%":
            if match.group(0) != "%%":
                raise ValueError("a %% with flags")
            pieces.append("%")
            continue
        width, precision = match.group("width"), match.group("precision")
        if width == "*" or precision == "*":
            raise ValueError("a * width or precision (taken from the arguments)")
        if kind not in _NUMERIC and kind != "s":
            raise ValueError(f"the %{kind} conversion")
        pieces.append(
            Conversion(
                flags=match.group("flags"),
                width=int(width) if width else None,
                precision=int(precision or 0) if precision is not None else None,
                kind=kind,
            )
        )
    if "%" in unescaped[position:]:
        raise ValueError("an incomplete % conversion")
    if position < len(unescaped):
        pieces.append(unescaped[position:])
    return pieces


def _escape(char: str) -> str:
    if char not in _ESCAPES:
        raise ValueError(f"the escape \\{char}")
    return _ESCAPES[char]


def python_spec(conversion: Conversion, *, is_string: bool) -> str:
    """The Python format spec that prints like ``conversion``.

    :raises ValueError: when there is none.
    """
    flags, width, precision, kind = (
        conversion.flags,
        conversion.width,
        conversion.precision,
        conversion.kind,
    )
    left = "-" in flags
    if is_string:
        if "0" in flags or "+" in flags or " " in flags:
            raise ValueError("numeric flags on %s")
        # Python left-aligns strings by default, C right-aligns them.
        align = ("<" if left else ">") if width is not None else ""
        spec = f"{align}{width or ''}"
        return spec + (f".{precision}" if precision is not None else "")
    spec = "<" if left else ""
    spec += "+" if "+" in flags else " " if " " in flags else ""
    spec += "#" if "#" in flags and kind not in "di" else ""
    spec += "0" if "0" in flags and not left else ""
    spec += str(width) if width is not None else ""
    if kind in "di":
        if precision is not None:
            raise ValueError("a precision on %d (a minimum digit count)")
        return spec + ("d" if spec else "")
    spec += f".{precision}" if precision is not None else ""
    return spec + ("f" if kind == "F" else kind)


# --- .m -> Python ---------------------------------------------------------------


def _format_and_args(node: ir.Call, ctx: Context) -> tuple[list[Piece], list[ir.Expr]]:
    """The parsed format and the value arguments of a printing call."""
    args = list(node.args)
    if node.func == "fprintf" and args and not isinstance(args[0], ir.Str):
        fid = ctx.info(args[0]).value
        if fid != 1:
            ctx.decline("fprintf to a file id other than 1 (standard output)")
        args = args[1:]
    if not args or not isinstance(args[0], ir.Str):
        ctx.decline(f"{node.func} with a format that is not a literal string")
    try:
        pieces = parse_format(args[0].value)
    except ValueError as exc:
        ctx.decline(f"{node.func} with {exc}")
    return pieces, args[1:]


def _plan(pieces: list[Piece], count: int, ctx: Context) -> list[str | tuple[Conversion, int]]:
    """Walk the format the way printf does for ``count`` scalar values.

    The format repeats while values remain; output stops at the first
    conversion with no value left, and after a full pass that used the last one.
    """
    conversions = sum(isinstance(p, Conversion) for p in pieces)
    if conversions == 0:
        if count:
            ctx.decline("values given to a format without conversions")
        return [p for p in pieces if isinstance(p, str)]
    if count == 0:
        ctx.decline("a format with conversions but no values")
    planned: list[str | tuple[Conversion, int]] = []
    used = 0
    while True:
        for piece in pieces:
            if isinstance(piece, Conversion):
                if used == count:
                    return planned
                planned.append((piece, used))
                used += 1
            else:
                planned.append(piece)
        if used == count:
            return planned


def _field(conversion: Conversion, value: ast.expr, info: Info, ctx: Context) -> ast.expr | str:
    """One ``{value:spec}`` field of the f-string (a literal value: its text)."""
    is_string = info.kind is Kind.STRING
    if conversion.kind == "s" and not is_string:
        ctx.decline("%s of a value not known to be a string (a number prints as a character)")
    if conversion.kind != "s":
        if is_string or not info.is_scalar:
            ctx.decline(f"%{conversion.kind} of a value that is not a known numeric scalar")
        if not info.is_real:
            ctx.decline(f"%{conversion.kind} of a value that may be complex")
        if isinstance(info.value, int | float) and not math.isfinite(info.value):
            ctx.decline("printing Inf or NaN (Octave prints Inf/NaN, Python inf/nan)")
        if not info.finite:
            ctx.assume("finite values printed (Octave prints Inf/NaN, Python inf/nan)")
    if conversion.kind in "di":
        if not info.integer:
            ctx.decline(
                f"%{conversion.kind} of a value not known to be a whole number "
                "(Octave prints a fraction in another format)"
            )
        if isinstance(info.value, int | float) and abs(info.value) >= 2**63:
            ctx.decline(f"%{conversion.kind} of a whole number beyond 64-bit integers")
        if info.value is None and not info.int_dtype:
            # A computed double may be huge; Octave prints those as %g (1e+20).
            ctx.assume(f"|value| < 2^63 under %{conversion.kind}")
        if not (isinstance(value, ast.Constant) and isinstance(value.value, int | float)):
            value = ctx.call(ast.Name("int"), value)
        elif isinstance(value.value, float):
            value = ast.Constant(int(value.value))
    if _has_string(value) and not (isinstance(value, ast.Constant) and info.kind is Kind.STRING):
        ctx.decline("a string expression inside an f-string field")
    try:
        spec = python_spec(conversion, is_string=is_string)
    except ValueError as exc:
        ctx.decline(str(exc))
    if isinstance(value, ast.Constant) and isinstance(value.value, int | float | str):
        # A literal is formatted now: `fprintf('%d %s\n', 3, 'ok')` prints fixed text.
        return format(value.value, spec)
    format_spec = ast.JoinedStr(values=[ast.Constant(spec)]) if spec else None
    return ast.FormattedValue(value=value, conversion=-1, format_spec=format_spec)


def _has_string(node: ast.expr) -> bool:
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str) for n in ast.walk(node))


def _joined(parts: list[str | ast.expr]) -> ast.expr:
    """An f-string of literal text and fields; a plain string when there are no fields."""
    values: list[ast.expr] = []
    for part in parts:
        if isinstance(part, str):
            if values and isinstance(values[-1], ast.Constant):
                previous = values[-1].value
                assert isinstance(previous, str)
                values[-1] = ast.Constant(previous + part)
            elif part:
                values.append(ast.Constant(part))
        else:
            values.append(part)
    if not any(isinstance(v, ast.FormattedValue) for v in values):
        return ast.Constant("".join(str(v.value) for v in values if isinstance(v, ast.Constant)))
    return ast.JoinedStr(values=values)


def _scalars(pieces: list[Piece], args: list[ir.Expr], ctx: Context) -> list[str | ast.expr]:
    """The printed text for scalar (and string) arguments, recycled statically."""
    for arg in args:
        info = ctx.info(arg)
        if not (info.is_scalar or info.kind is Kind.STRING):
            ctx.decline("recycling a format over arrays mixed with other values")
    values = [ctx.expr(arg) for arg in args]
    parts: list[str | ast.expr] = []
    for item in _plan(pieces, len(args), ctx):
        if isinstance(item, str):
            parts.append(item)
        else:
            conversion, k = item
            parts.append(_field(conversion, values[k], ctx.info(args[k]), ctx))
    return parts


@dataclass
class _Loop:
    """A format recycled over one array: ``for <target> in <iterable>``."""

    target: str
    iterable: ast.expr
    parts: list[str | ast.expr]


def _array_loop(pieces: list[Piece], array: ir.Expr, ctx: Context) -> _Loop:
    """One pass of the format per element (one conversion) or per column (one per row)."""
    info = ctx.info(array)
    conversions = [p for p in pieces if isinstance(p, Conversion)]
    if not conversions:
        ctx.decline("an array given to a format without conversions")
    if info.numel == 0 or info.kind is Kind.EMPTY:
        ctx.decline("an empty array given to a format (the source prints it once, empty)")
    if info.numel is None:
        # An empty array prints the format once with empty fields; a loop, nothing.
        ctx.assume("a non-empty array")
    name = _fresh("value", ctx)
    element = Info(
        kind=Kind.SCALAR,
        rows=1,
        cols=1,
        elem=info.elem,
        finite=info.finite,
        nonneg=info.nonneg,
        integer=info.integer,
    )
    expr = ctx.expr(array)
    if len(conversions) == 1:
        if info.is_matrix:
            iterable = ctx.method(expr, "ravel", order=ast.Constant("F"))
        elif info.is_vector:
            iterable = expr
        else:
            ctx.decline("recycling a format over a value whose shape is unknown")
        fields: list[ast.expr] = [ast.Name(name)]
    else:
        if not (info.is_matrix and info.rows == len(conversions)):
            ctx.decline(
                "recycling a format with several conversions over an array that does not "
                "have one row per conversion"
            )
        iterable = ast.Attribute(value=expr, attr="T")
        name = _fresh("column", ctx)
        fields = [
            ast.Subscript(value=ast.Name(name), slice=ast.Constant(k))
            for k in range(len(conversions))
        ]
    if element.elem is Elem.UNKNOWN:
        ctx.decline("recycling a format over values of unknown type")
    parts: list[str | ast.expr] = []
    k = 0
    for piece in pieces:
        if isinstance(piece, Conversion):
            parts.append(_field(piece, fields[k], element, ctx))
            k += 1
        else:
            parts.append(piece)
    return _Loop(target=name, iterable=iterable, parts=parts)


def _fresh(base: str, ctx: Context) -> str:
    name = base
    while name in ctx.scope.variables:
        name += "_"
    return name


def _is_array_call(args: list[ir.Expr], ctx: Context) -> bool:
    return len(args) == 1 and ctx.info(args[0]).is_array


def _print(parts: list[str | ast.expr], ctx: Context) -> ast.expr:
    """``print(...)`` of the text, its trailing newline as print's own."""
    ends_line = bool(parts) and isinstance(parts[-1], str) and parts[-1].endswith("\n")
    if ends_line:
        last = parts[-1]
        assert isinstance(last, str)
        parts = [*parts[:-1], last[:-1]]
    text = _joined(parts)
    args = [] if isinstance(text, ast.Constant) and text.value == "" else [text]
    keywords = {} if ends_line else {"end": ast.Constant("")}
    if not args and not ends_line:
        ctx.decline("printing nothing")
    return ctx.call(ast.Name("print"), *args, **keywords)


def translate_call_statement(stmt: ir.ExprStmt, ctx: Context) -> list[Line]:
    """``printf``/``fprintf``/``disp`` as a statement; a loop when an array recycles."""
    node = stmt.value
    assert isinstance(node, ir.Call)
    ctx.file_directive("stdout")
    if node.func == "disp":
        return [ctx.line(ast.unparse(_disp(node, ctx)))]
    if node.func not in ("printf", "fprintf"):
        return [ctx.line(ast.unparse(translate(node, ctx)))]
    pieces, args = _format_and_args(node, ctx)
    if _is_array_call(args, ctx):
        loop = _array_loop(pieces, args[0], ctx)
        header = f"for {loop.target} in {ast.unparse(loop.iterable)}:"
        return [ctx.line(header), ctx.line(ast.unparse(_print(loop.parts, ctx)), extra=1)]
    return [ctx.line(ast.unparse(_print(_scalars(pieces, args, ctx), ctx)))]


def _disp(node: ir.Call, ctx: Context) -> ast.expr:
    if len(node.args) != 1:
        ctx.decline("disp with more than one argument")
    if ctx.info(node.args[0]).kind is not Kind.STRING:
        ctx.decline("disp of a value that is not a string (its number format is not Python's)")
    return ctx.call(ast.Name("print"), ctx.expr(node.args[0]))


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``sprintf`` as a value; the printing functions only as statements."""
    if node.func != "sprintf":
        ctx.decline(f"{node.func} used as a value")
    pieces, args = _format_and_args(node, ctx)
    if _is_array_call(args, ctx):
        loop = _array_loop(pieces, args[0], ctx)
        generator = ast.ListComp(
            elt=_joined(loop.parts),
            generators=[
                ast.comprehension(
                    target=ast.Name(loop.target), iter=loop.iterable, ifs=[], is_async=0
                )
            ],
        )
        return ctx.method(ast.Constant(""), "join", generator)
    return _joined(_scalars(pieces, args, ctx))


def format_text(fmt: str, args: list[ir.Expr], ctx: Context) -> ast.expr:
    """``sprintf(fmt, args...)`` for other rules (``num2str(x, fmt)``)."""
    try:
        pieces = parse_format(fmt)
    except ValueError as exc:
        ctx.decline(f"a format with {exc}")
    return _joined(_scalars(pieces, args, ctx))


# --- Python -> .m ------------------------------------------------------------
#
# print() of literal text, strings and f-strings becomes fprintf with the
# matching format (disp for a single string); an f-string used as a value is
# sprintf. Each {value:spec} field maps to a conversion when Python and printf
# format it the same way: d, f, e, g (and E, G) with width, precision, sign,
# zero padding and alignment, and strings. A number without a spec prints
# with Python's repr, which no printf format reproduces, so it is declined,
# except an integer, whose digits are the same under %d.

_PY_SPEC = re.compile(
    r"^(?:(?P<fill>.)?(?P<align>[<>=^]))?(?P<sign>[-+ ])?(?P<alt>#)?(?P<zero>0)?"
    r"(?P<width>\d+)?(?P<grouping>[,_])?(?:\.(?P<precision>\d+))?(?P<type>[A-Za-z%])?$"
)


def _m_literal(text: str, ctx: LiftContext) -> str:
    """Python text as printf format text: ``%`` doubled, control characters escaped."""
    out = []
    for char in text:
        if char == "%":
            out.append("%%")
        elif char == "\\":
            out.append("\\\\")
        elif char == "\n":
            out.append("\\n")
        elif char == "\t":
            out.append("\\t")
        elif ord(char) < 32 or ord(char) == 127:
            ctx.decline("printing a control character")
        else:
            out.append(char)
    return "".join(out)


def _m_conversion(spec: str, value: ast.expr, ctx: LiftContext) -> tuple[str, ir.Expr]:
    """``%...`` and the value for one f-string field."""
    info = ctx.info(value)
    is_string = info.kind is PyKind.STRING
    match = _PY_SPEC.match(spec)
    if match is None:
        ctx.decline(f"the format spec {spec!r}")
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "int":
        # f"{int(x):5d}": an integer, the same digits as %5d of fix(x).
        if match.group("type") not in (None, "d") or len(value.args) != 1 or value.keywords:
            ctx.decline("int() with this format spec inside an f-string")
        if ctx.info(value.args[0]).kind is PyKind.STRING:
            ctx.decline("int() of a string")
        inner = value.args[0]
        fixed = (
            ctx.expr(inner)
            if ctx.info(inner).dtype is DType.INT  # already whole: int() changes nothing
            else ctx.call("fix", ctx.expr(inner), loc=ctx.loc(value))
        )
        flags = "-" if match.group("align") == "<" else ""
        flags += match.group("sign") if match.group("sign") in ("+", " ") else ""
        flags += "0" if match.group("zero") and match.group("align") != "<" else ""
        align = match.group("align")
        if match.group("precision") or match.group("grouping") or align in ("=", "^"):
            ctx.decline(f"the format spec {spec!r} on an integer")
        return f"%{flags}{match.group('width') or ''}d", fixed
    fill, align, sign = match.group("fill"), match.group("align"), match.group("sign")
    kind, width, precision = match.group("type"), match.group("width"), match.group("precision")
    if (fill not in (None, " ")) or align in ("=", "^") or match.group("grouping"):
        ctx.decline(f"the format spec {spec!r} (fill, centring or grouping)")
    flags = ""
    if is_string:
        if kind not in (None, "s") or sign or match.group("zero") or match.group("alt"):
            ctx.decline(f"the format spec {spec!r} on a string")
        if align != ">" and width:
            flags = "-"  # Python left-aligns strings, printf right-aligns them
        dot = f".{precision}" if precision else ""
        return f"%{flags}{width or ''}{dot}s", ctx.expr(value)
    if not info.is_scalar or info.dtype is DType.COMPLEX:
        ctx.decline("formatting a value that is not a real scalar")
    if kind is None and not precision and info.dtype in (DType.INT, DType.BOOL):
        if info.dtype is DType.BOOL:
            ctx.decline("formatting a bool (Python prints True/False)")
        kind = "d"
    if kind not in ("d", "f", "e", "E", "g", "G"):
        ctx.decline(f"the format spec {spec!r} (Python's own number format)")
    if kind == "d" and info.dtype not in (DType.INT, DType.BOOL):
        ctx.decline("{:d} of a value that is not an integer")
    if kind == "d" and precision:
        ctx.decline("a precision on an integer")
    if not info.finite:
        ctx.assume("finite values printed (Python prints inf/nan, the .m side Inf/NaN)")
    flags += "-" if align == "<" else ""
    flags += sign if sign in ("+", " ") else ""
    flags += "#" if match.group("alt") else ""
    flags += "0" if match.group("zero") and align != "<" else ""
    dot = f".{precision}" if precision else ""
    return f"%{flags}{width or ''}{dot}{kind}", ctx.expr(value)


def _m_format(parts: list[ast.expr], ctx: LiftContext) -> tuple[str, list[ir.Expr]]:
    """The printf format and values for a sequence of printed Python values."""
    fmt: list[str] = []
    values: list[ir.Expr] = []
    for part in parts:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            fmt.append(_m_literal(part.value, ctx))
        elif isinstance(part, ast.JoinedStr):
            for piece in part.values:
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    fmt.append(_m_literal(piece.value, ctx))
                    continue
                assert isinstance(piece, ast.FormattedValue)
                if piece.conversion != -1:
                    ctx.decline("a !r/!s/!a conversion in an f-string")
                spec = ""
                if piece.format_spec is not None:
                    inner = piece.format_spec
                    assert isinstance(inner, ast.JoinedStr)
                    if not all(
                        isinstance(v, ast.Constant) and isinstance(v.value, str)
                        for v in inner.values
                    ):
                        ctx.decline("a computed format spec")
                    spec = "".join(
                        str(v.value) for v in inner.values if isinstance(v, ast.Constant)
                    )
                conversion, value = _m_conversion(spec, piece.value, ctx)
                fmt.append(conversion)
                values.append(value)
        else:
            conversion, value = _m_conversion("", part, ctx)
            fmt.append(conversion)
            values.append(value)
    return "".join(fmt), values


def _lift_join(node: ast.Call, ctx: LiftContext) -> ir.Expr:
    """``"".join([f"..." for v in values])``: ``sprintf`` recycles its format per element."""
    loc = ctx.loc(node)
    receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
    comprehension = node.args[0] if len(node.args) == 1 and not node.keywords else None
    if not (
        isinstance(receiver, ast.Constant)
        and receiver.value == ""
        and isinstance(comprehension, ast.ListComp | ast.GeneratorExp)
        and len(comprehension.generators) == 1
        and not comprehension.generators[0].ifs
        and isinstance(comprehension.generators[0].target, ast.Name)
        and isinstance(comprehension.elt, ast.JoinedStr)
    ):
        ctx.decline("str.join of anything but f-strings over one array")
    generator = comprehension.generators[0]
    assert isinstance(generator.target, ast.Name)
    if not ctx.info(generator.iter).is_vector:
        ctx.decline("joining over a value that is not a 1-D array")
    fields = [p for p in comprehension.elt.values if isinstance(p, ast.FormattedValue)]
    loop_name = generator.target.id
    for field in fields:
        value = field.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            value = value.args[0] if value.func.id == "int" and value.args else value
        if not (isinstance(value, ast.Name) and value.id == loop_name) or len(fields) != 1:
            ctx.decline("a joined f-string with fields other than the one loop value")
    fmt, _values = _m_format([comprehension.elt], ctx)
    values = ctx.module_for("syntax/array-literals").lift_items(generator.iter, ctx)
    return ctx.call("sprintf", ir.Str(loc=loc, value=fmt, quote="'"), values, loc=loc)


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``print(...)`` as ``fprintf``/``disp``; an f-string value as ``sprintf``."""
    loc = ctx.loc(node)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return _lift_join(node, ctx)
    if isinstance(node, ast.JoinedStr):
        fmt, values = _m_format([node], ctx)
        return ctx.call("sprintf", ir.Str(loc=loc, value=fmt, quote="'"), *values, loc=loc)
    assert isinstance(node, ast.Call)
    sep, end = " ", "\n"
    for keyword in node.keywords:
        if keyword.arg not in ("sep", "end") or not (
            isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str)
        ):
            ctx.decline(f"print with {keyword.arg}=")
        if keyword.arg == "sep":
            sep = str(keyword.value.value)
        else:
            end = str(keyword.value.value)
    ctx.file_directive("stdout")
    args = list(node.args)
    if (
        len(args) == 1
        and end == "\n"
        and ctx.info(args[0]).kind is PyKind.STRING
        and not isinstance(args[0], ast.JoinedStr)
    ):
        return ctx.call("disp", ctx.expr(args[0]), loc=loc)
    parts: list[ast.expr] = []
    for k, arg in enumerate(args):
        if k:
            parts.append(ast.Constant(sep))
        parts.append(arg)
    parts.append(ast.Constant(end))
    fmt, values = _m_format(parts, ctx)
    return ctx.call("fprintf", ir.Str(loc=loc, value=fmt, quote="'"), *values, loc=loc)
