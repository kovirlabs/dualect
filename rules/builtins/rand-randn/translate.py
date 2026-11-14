# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/rand-randn: uniform and normal random numbers.

``rand(m, n)`` is ``np.random.default_rng().random((m, n))`` and
``randn(m, n)`` ``np.random.default_rng().standard_normal((m, n))``: NumPy's
Generator API (its legacy ``np.random.rand`` is discouraged, and flagged by
linters), kept as one expression so no generator object has to be threaded
through the code. ``rand(n)`` is square, ``rand(1, n)`` and ``rand(n, 1)``
are 1-D, and ``rand`` alone a scalar. A size that may be 1 is assumed not to be, out loud.
Seeding and class arguments (``rand('seed', 1)``, ``'single'``) are declined.

The two sides draw different numbers, so a random result can only be checked
for its shape: when a call is the whole right-hand side of an assignment to a
script variable, the translation adds a ``dualect: shape <name>`` directive.
Values computed from random numbers are compared as usual and fail
verification, loudly, unless they are annotated the same way.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind, array, scalar
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind
from dualect.pyinfer import array as py_array
from dualect.pyinfer import scalar as py_scalar


def _dims(node: ir.Call) -> list[ir.Expr] | None:
    args = node.args
    if len(args) == 1 and isinstance(args[0], ir.Matrix):
        rows = args[0].rows
        return list(rows[0]) if len(rows) == 1 and len(rows[0]) == 2 else None
    if len(args) == 1:
        return [args[0], args[0]]
    if len(args) in (0, 2):
        return list(args)
    return None


def _value(node: ir.Expr, info: Info | None) -> int | None:
    if isinstance(node, ir.Num) and isinstance(node.value, int | float):
        return int(node.value) if float(node.value).is_integer() else None
    if info is not None and isinstance(info.value, int | float):
        return int(info.value) if float(info.value).is_integer() else None
    return None


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """A scalar, a vector or a matrix of finite doubles."""
    dims = _dims(node)
    if dims is None or any(i.kind is Kind.STRING for i in args):
        return None
    nonneg = node.func == "rand"
    if not dims:
        return scalar(nonneg=nonneg)
    infos: list[Info | None] = [None, None]
    if len(args) == 2:
        infos = list(args)
    elif args and not isinstance(node.args[0], ir.Matrix):
        infos = [args[0], args[0]]
    rows, cols = (_value(d, i) for d, i in zip(dims, infos, strict=True))
    if rows == 1 and cols == 1:
        return scalar(nonneg=nonneg)
    kind = None if 1 in (rows, cols) else Kind.MATRIX
    return array(rows, cols, elem=Elem.REAL, kind=kind, finite=True, nonneg=nonneg)


def _shape_directive(node: ir.Call, ctx: Context) -> None:
    current = ctx.current
    if (
        ctx.scope.is_script
        and isinstance(current, ir.Assign)
        and current.value is node
        and len(current.targets) == 1
        and isinstance(current.targets[0], ir.Name)
    ):
        ctx.file_directive(f"shape {current.targets[0].id}")


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``rand``/``randn`` with zero, one or two sizes."""
    if any(isinstance(arg, ir.Str) for arg in node.args):
        ctx.decline(f"{node.func} with a seed, state or class argument")
    dims = _dims(node)
    if dims is None:
        ctx.decline(f"{node.func} with more than two dimensions")
    for dim in dims:
        if not ctx.info(dim).is_scalar:
            ctx.decline(f"{node.func} with a size that is not a number")
    generator = ctx.call(ctx.np("random.default_rng"))
    method = "random" if node.func == "rand" else "standard_normal"
    _shape_directive(node, ctx)
    if not dims:
        return ctx.method(generator, method)
    rows, cols = (_value(d, ctx.info(d)) for d in dims)
    if rows == 1 and cols == 1:
        return ctx.method(generator, method)
    if rows == 1:
        return ctx.method(generator, method, ctx.expr(dims[1]))
    if cols == 1:
        return ctx.method(generator, method, ctx.expr(dims[0]))
    if rows is None or cols is None:
        ctx.assume("neither dimension is 1")
    size = ast.Tuple(elts=[ctx.expr(dims[0]), ctx.expr(dims[1])])
    return ctx.method(generator, method, size)


# --- Python -> .m ------------------------------------------------------------


_METHODS = {"random": "rand", "standard_normal": "randn"}


def _is_generator(node: ast.expr, ctx: LiftContext) -> bool:
    """``np.random.default_rng()`` with no seed."""
    return (
        isinstance(node, ast.Call)
        and ctx.tag(node) == "py:call:np.random.default_rng"
        and not node.args
        and not node.keywords
    )


def _sizes(node: ast.Call, ctx: LiftContext) -> list[ast.expr]:
    """The size arguments of a generator method: ``n`` or ``(m, n)``."""
    if node.keywords or len(node.args) > 1:
        ctx.decline("a generator method with these arguments")
    if not node.args:
        return []
    size = node.args[0]
    parts = list(size.elts) if isinstance(size, ast.Tuple | ast.List) else [size]
    if len(parts) > 2:
        ctx.decline("random numbers in more than two dimensions")
    return parts


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.random.default_rng().random(size)`` and friends; the legacy functions."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    tag = ctx.tag(node) or ""
    if tag == "py:call:np.random.default_rng":
        ctx.decline("a random generator object (only its random/standard_normal draws)")
    if tag.startswith("py:method:"):
        assert isinstance(node.func, ast.Attribute)
        if not _is_generator(node.func.value, ctx) or node.func.attr not in _METHODS:
            ctx.decline(f"{node.func.attr} of a generator other than np.random.default_rng()")
        func = _METHODS[node.func.attr]
        sizes = _sizes(node, ctx)
    else:
        func = tag.removeprefix("py:call:np.random.")
        if node.keywords or len(node.args) > 2:
            ctx.decline(f"np.random.{func} with these arguments")
        sizes = list(node.args)
    for arg in sizes:
        if not ctx.info(arg).is_scalar:
            ctx.decline(f"{func} with a size that is not a number")
    current = ctx.current
    if (
        ctx.function is None
        and isinstance(current, ast.Assign)
        and current.value is node
        and len(current.targets) == 1
        and isinstance(current.targets[0], ast.Name)
    ):
        ctx.file_directive(f"shape {current.targets[0].id}")
    args = [ctx.expr(arg) for arg in sizes]
    if len(args) == 1:
        return ctx.call(func, ctx.num(1), args[0], loc=loc)  # 1-D: a row
    return ctx.call(func, *args, loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """No size: a float; one: 1-D; two: 2-D."""
    if tag in ("py:method:random", "py:method:standard_normal"):
        if args and args[0].kind is PyKind.TUPLE:
            args = list(args[0].items or ())
        nonneg = tag == "py:method:random"
    elif tag in ("py:call:np.random.rand", "py:call:np.random.randn"):
        nonneg = tag.endswith(".rand")
    else:
        return None
    if not args:
        return py_scalar(DType.FLOAT, nonneg=nonneg)
    kind = PyKind.VECTOR if len(args) == 1 else PyKind.MATRIX
    length = args[0].value if len(args) == 1 and isinstance(args[0].value, int) else None
    return py_array(kind, DType.FLOAT, finite=True, nonneg=nonneg, length=length)
