# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/meshgrid: ``[X, Y] = meshgrid(x, y)``.

``np.meshgrid`` with its default ``indexing="xy"`` builds the same grids
(given two vectors: ``np.meshgrid(x)`` returns one 1-D grid):
``X`` repeats ``x`` along each row and ``Y`` repeats ``y`` down each column,
both ``numel(y) x numel(x)``. ``meshgrid(x)`` is ``meshgrid(x, x)``. The
inputs must be vectors (a matrix is flattened in a different order on each
side); a grid with a single row or column would be 1-D on the ``.m`` side
and 2-D in NumPy, so a length known to be 1 is declined and an unknown one
assumed not to be, out loud. Three-dimensional grids are declined.

From Python, ``X, Y = np.meshgrid(x, y)`` is ``[X, Y] = meshgrid(x, y)``;
``indexing="ij"`` (the transposed grids) is declined.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context, Line
from dualect.infer import Info, Kind, array
from dualect.lift import LiftContext
from dualect.pyinfer import PyInfo, PyKind
from dualect.pyinfer import array as py_array


def _grid(source: Info, rows: Info, cols: Info) -> Info:
    # np.meshgrid keeps the dtype: integer inputs give integer grids, which
    # element stores and powers must know about.
    grid = array(
        rows.numel,
        cols.numel,
        elem=source.elem,
        finite=source.finite,
        nonneg=source.nonneg,
        integer=source.integer,
        kind=Kind.MATRIX,
    )
    return replace(grid, int_dtype=source.int_dtype)


def infer_outputs(node: ir.Call, args: list[Info], nargout: int) -> list[Info] | None:
    """``X`` holds ``x`` and ``Y`` holds ``y``, both ``numel(y) x numel(x)``."""
    if node.func != "meshgrid" or len(args) not in (1, 2):
        return None
    x = args[0]
    y = args[1] if len(args) == 2 else x
    return [_grid(x, y, x), _grid(y, y, x)]


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """The first output alone."""
    outputs = infer_outputs(node, args, nargout)
    return outputs[0] if outputs else None


def _arguments(node: ir.Call, ctx: Context) -> list[ast.expr]:
    if len(node.args) not in (1, 2):
        ctx.decline("meshgrid with three inputs (a 3-D grid)")
    for arg in node.args:
        info = ctx.info(arg)
        if not (info.is_vector or info.is_scalar):
            ctx.decline("meshgrid of a value not known to be a vector")
        if info.numel == 1 or info.is_scalar:
            ctx.decline("meshgrid with a single point along an axis (a 1-D grid in the source)")
        if info.numel is None:
            ctx.assume("grid vectors longer than one element")
    if len(node.args) == 1 and any(
        isinstance(sub, ir.Call) and (sub.func in ("rand", "randn") or sub.local or sub.handle)
        for sub in ir.walk_expr(node.args[0])
    ):
        ctx.decline("meshgrid(x) of an expression that cannot be evaluated twice")
    args = [ctx.arg(arg) for arg in node.args]
    # np.meshgrid(x) returns a single 1-D grid; meshgrid(x) is meshgrid(x, x).
    return args * 2 if len(args) == 1 else args


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``X = meshgrid(x, y)``: the first grid."""
    call = ctx.call(ctx.np("meshgrid"), *_arguments(node, ctx))
    return ast.Subscript(value=call, slice=ast.Constant(0))


def translate_multi(stmt: ir.Assign, ctx: Context) -> list[Line]:
    """``[X, Y] = meshgrid(x, y)``."""
    call = stmt.value
    assert isinstance(call, ir.Call)
    if len(stmt.targets) != 2:
        ctx.decline("meshgrid with three outputs")
    names = []
    for target in stmt.targets:
        if isinstance(target, ir.Name):
            names.append(target.id)
        elif isinstance(target, ir.Ignore):
            names.append("_")
        else:
            ctx.decline("assigning a grid into an element")
    value = ctx.call(ctx.np("meshgrid"), *_arguments(call, ctx))
    return [ctx.line(f"{', '.join(names)} = {ast.unparse(value)}")]


# --- Python -> .m ------------------------------------------------------------


def _lift_call(node: ast.Call, ctx: LiftContext) -> ir.Call:
    for keyword in node.keywords:
        value = keyword.value
        if not (keyword.arg == "indexing" and isinstance(value, ast.Constant)):
            ctx.decline(f"np.meshgrid with {keyword.arg}=")
        if value.value != "xy":
            ctx.decline('np.meshgrid with indexing="ij" (the transposed grids)')
    if len(node.args) != 2:
        ctx.decline("np.meshgrid of other than two vectors")
    literals = ctx.module_for("syntax/array-literals")
    for arg in node.args:
        if not (ctx.info(arg).is_vector or isinstance(arg, ast.List)):
            ctx.decline("np.meshgrid of a value not known to be 1-D")
    args = [literals.lift_items(arg, ctx) for arg in node.args]
    return ctx.call("meshgrid", *args, loc=ctx.loc(node))


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.meshgrid`` used as a value (a tuple of grids) is declined."""
    ctx.decline("np.meshgrid used other than unpacked into names")


def lift_multi(stmt: ast.Assign, ctx: LiftContext) -> list[ir.Stmt]:
    """``X, Y = np.meshgrid(x, y)``."""
    target = stmt.targets[0]
    value = stmt.value
    assert isinstance(target, ast.Tuple | ast.List) and isinstance(value, ast.Call)
    if len(target.elts) != 2 or len(value.args) != 2:
        ctx.decline("unpacking a different number of grids than np.meshgrid returns")
    targets: list[ir.Expr] = []
    for element in target.elts:
        if not isinstance(element, ast.Name):
            ctx.decline("unpacking a grid into an element")
        if element.id == "_":
            targets.append(ir.Ignore(loc=ctx.loc(element)))
        else:
            targets.append(ir.Name(loc=ctx.loc(element), id=element.id))
    return [ir.Assign(loc=ctx.loc(stmt), targets=targets, value=_lift_call(value, ctx))]


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """``np.meshgrid``: a tuple of 2-D grids."""
    if tag != "py:call:np.meshgrid" or not args:
        return None
    grids = tuple(py_array(PyKind.MATRIX, a.dtype, finite=a.finite, nonneg=a.nonneg) for a in args)
    return PyInfo(kind=PyKind.TUPLE, items=grids, length=len(grids))
