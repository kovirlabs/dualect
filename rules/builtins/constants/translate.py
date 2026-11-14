# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/constants: ``pi``, ``Inf``, ``NaN``, ``eps``, ``true``, ...

Only the argument-free forms are constants. With dimensions, ``Inf``,
``NaN``, ``true`` and ``false`` fill an array: ``np.full`` with the same
value, 1-D when one dimension is 1 (like ``zeros``); ``true(n)`` is n x n.
``eps(x)`` (the spacing at ``x``) is declined.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind, array
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind

_NUMPY = {"pi": "pi", "e": "e", "Inf": "inf", "inf": "inf", "NaN": "nan", "nan": "nan"}


_FILLS = {"Inf", "inf", "NaN", "nan", "true", "false"}


def _is_one(node: ir.Expr, ctx: Context) -> bool:
    return ctx.info(node).value == 1


def _fill(node: ir.Call, ctx: Context) -> ast.expr:
    """``Inf(m, n)``, ``true(n)``, ...: ``np.full(shape, value)``."""
    args = node.args
    if any(not ctx.info(a).is_scalar or isinstance(a, ir.Str) for a in args) or len(args) > 2:
        ctx.decline(f"{node.func} with a size vector or a class argument")
    name = node.func
    value = ctx.np(_NUMPY[name]) if name in _NUMPY else ast.Constant(name == "true")
    full = ctx.np("full")
    if len(args) == 1:
        n = ctx.expr(args[0])
        return ctx.call(full, ast.Tuple(elts=[n, n]), value)
    rows, cols = args
    if _is_one(rows, ctx):
        return ctx.call(full, ctx.expr(cols), value)
    if _is_one(cols, ctx):
        return ctx.call(full, ctx.expr(rows), value)
    if ctx.info(rows).value is None or ctx.info(cols).value is None:
        ctx.assume("neither dimension is 1")
    return ctx.call(full, ast.Tuple(elts=[ctx.expr(rows), ctx.expr(cols)]), value)


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """Facts for the filled arrays (the argument-free constants are core facts)."""
    if node.func not in _FILLS or not args or len(args) > 2:
        return None
    dims = [int(a.value) if isinstance(a.value, int | float) else None for a in args]
    rows, cols = (dims[0], dims[0]) if len(dims) == 1 else (dims[0], dims[1])
    logical = node.func in ("true", "false")
    facts = {
        "elem": Elem.LOGICAL if logical else Elem.REAL,
        "finite": logical,
        "nonneg": node.func not in ("NaN", "nan"),
        "integer": logical,
    }
    if rows != 1 and cols != 1:
        return array(rows, cols, kind=Kind.MATRIX, **facts)  # type: ignore[arg-type]
    return array(rows, cols, **facts)  # type: ignore[arg-type]


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """One constant, or an array filled with one."""
    if node.args and node.func in _FILLS:
        return _fill(node, ctx)
    if node.args:
        ctx.decline(f"{node.func} with arguments")
    name = node.func
    if name in _NUMPY:
        return ctx.np(_NUMPY[name])
    if name == "eps":
        return ast.Attribute(value=ctx.call(ctx.np("finfo"), ast.Name("float")), attr="eps")
    if name in ("true", "false"):
        return ast.Constant(name == "true")
    if name in ("i", "j"):
        return ast.Constant(1j)
    ctx.decline(f"constant {name}")


# --- Python -> .m ------------------------------------------------------------
#
# np.pi is pi, np.inf Inf, np.nan NaN, np.finfo(float).eps eps, 2j is 2i.
# np.e is exp(1): a variable named e (common) would hide the function e.


_FROM_NUMPY = {"np.pi": "pi", "np.inf": "Inf", "np.nan": "NaN"}


_FILL_FROM = {"np.inf": "Inf", "np.nan": "NaN"}


def _lift_full(node: ast.Call, ctx: LiftContext) -> ir.Expr:
    """``np.full(shape, np.inf)`` and friends: ``Inf(m, n)``, ``true(1, n)``, ..."""
    loc = ctx.loc(node)
    if len(node.args) != 2 or node.keywords:
        ctx.decline("np.full with these arguments")
    shape, value = node.args
    if isinstance(value, ast.Constant) and isinstance(value.value, bool):
        name = "true" if value.value else "false"
    else:
        tag = ctx.tag(value) or ""
        name = _FILL_FROM.get(tag.removeprefix("py:const:"), "")
        if not name:
            ctx.decline("np.full with a value other than inf, nan, True or False")
    if isinstance(shape, ast.Tuple):
        if len(shape.elts) != 2:
            ctx.decline("np.full with other than two dimensions")
        dims = [ctx.expr(e) for e in shape.elts]
    else:
        if not ctx.info(shape).is_scalar:
            ctx.decline("np.full with a computed shape")
        dims = [ctx.num(1), ctx.expr(shape)]  # 1-D: a row
    return ctx.call(name, *dims, loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """``np.full``: an array of the shape given, holding the value given."""
    if tag != "py:call:np.full" or len(args) != 2:
        return None
    shape, value = args
    if shape.kind is PyKind.TUPLE and shape.length == 2:
        kind = PyKind.MATRIX
    elif shape.is_scalar:
        kind = PyKind.VECTOR
    else:
        return None
    dtype = value.dtype if value.dtype is not DType.UNKNOWN else DType.FLOAT
    length = shape.value if kind is PyKind.VECTOR and isinstance(shape.value, int) else None
    return PyInfo(kind=kind, dtype=dtype, finite=value.finite, nonneg=value.nonneg, length=length)


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """A NumPy constant, machine epsilon, an imaginary literal or ``np.full``."""
    loc = ctx.loc(node)
    if isinstance(node, ast.Call) and ctx.tag(node) == "py:call:np.full":
        return _lift_full(node, ctx)
    if isinstance(node, ast.Constant) and isinstance(node.value, complex):
        return ir.Num(loc=loc, text="", value=node.value)
    tag = ctx.tag(node)
    if tag is not None and tag.startswith("py:const:"):
        name = tag[len("py:const:") :]
        if name in _FROM_NUMPY:
            return ctx.call(_FROM_NUMPY[name], loc=loc)
        if name == "np.e":
            return ctx.call("exp", ctx.num(1), loc=loc)
        ctx.decline(f"the NumPy name {name}")
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "eps"
        and isinstance(node.value, ast.Call)
        and ctx.tag(node.value) == "py:call:np.finfo"
        and [ast.unparse(a) for a in node.value.args] in (["float"], ["np.float64"])
    ):
        return ctx.call("eps", loc=loc)
    ctx.decline(f"{ast.unparse(node)}")
