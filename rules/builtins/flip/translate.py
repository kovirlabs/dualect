# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/flip: ``fliplr``, ``flipud`` and ``flip``.

The orientation of the source vector decides: ``fliplr`` of a row and
``flipud`` of a column reverse the 1-D array (``np.flip(v)``), while
``fliplr`` of a column and ``flipud`` of a row return it unchanged. A matrix
is ``np.fliplr``/``np.flipud``; ``flip`` works along the first non-singleton
dimension (any vector, the rows of a matrix: ``np.flip(A, axis=0)``) or the
dimension given. A vector of unknown orientation is declined for ``fliplr``
and ``flipud``. A string is reversed with ``s[::-1]``. NumPy's flips are
views, so the result is copied when it or its input is later written into.

From Python, ``np.flip`` of a 1-D array is ``flip`` of the row; of a 2-D
array it needs an axis (without one it reverses both). ``np.flipud`` of a
1-D array reverses it, which on the ``.m`` side (a row) is ``fliplr``.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Info, Kind
from dualect.lift import LiftContext, view_root
from dualect.pyinfer import PyInfo, as_array

#: The dimension each function reverses (None: the first non-singleton one).
_DIM = {"fliplr": 2, "flipud": 1}


def _reversed(data: ir.Expr, info: Info, ctx: Context) -> ast.expr:
    if info.kind is Kind.STRING:
        step = ast.Slice(step=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1)))
        return ast.Subscript(value=ctx.expr(data), slice=step)
    return _private(ctx.call(ctx.np("flip"), ctx.arg(data)), data, ctx)


def _private(result: ast.expr, data: ir.Expr, ctx: Context) -> ast.expr:
    """``result.copy()`` when the view NumPy returns could be written through.

    ``np.flip`` (and ``np.fliplr``/``np.flipud``) return views of their input;
    the source's result is a separate array.
    """
    stmt = ctx.current
    target = stmt.targets[0] if isinstance(stmt, ir.Assign) and len(stmt.targets) == 1 else None
    names = {data.id} if isinstance(data, ir.Name) else set()
    if isinstance(target, ir.Name):
        names.add(target.id)
    if names & ctx.scope.mutated:
        return ctx.method(result, "copy")
    return result


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """One flip."""
    if not node.args or len(node.args) > (2 if node.func == "flip" else 1):
        ctx.decline(f"{node.func} with these arguments")
    data = node.args[0]
    info = ctx.info(data)
    dim = _DIM.get(node.func)
    if len(node.args) == 2:
        value = ctx.info(node.args[1]).value
        if value not in (1, 2):
            ctx.decline("flip along a computed or third dimension")
        assert isinstance(value, int | float)
        dim = int(value)
    if info.is_scalar or info.kind is Kind.EMPTY:
        return ctx.expr(data)  # nothing to reverse
    if info.is_matrix:
        if dim is None:
            result = ctx.call(ctx.np("flip"), ctx.expr(data), axis=ast.Constant(0))
        else:
            result = ctx.call(ctx.np("fliplr" if dim == 2 else "flipud"), ctx.expr(data))
        return _private(result, data, ctx)
    if info.kind in (Kind.ROW, Kind.STRING):
        return _reversed(data, info, ctx) if dim in (None, 2) else ctx.expr(data)
    if info.kind is Kind.COL:
        return _reversed(data, info, ctx) if dim in (None, 1) else ctx.expr(data)
    if info.kind is Kind.VECTOR and dim is None:
        return _reversed(data, info, ctx)
    ctx.decline(f"{node.func} of a value whose shape or orientation is unknown")


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """The data's shape and type."""
    if not args:
        return None
    return replace(args[0], value=None, func=None, env=None)


# --- Python -> .m ------------------------------------------------------------


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.flip(x, axis=...)``, ``np.fliplr(A)``, ``np.flipud(x)``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    name = (ctx.tag(node) or "").removeprefix("py:call:np.")
    if not node.args or len(node.args) > (2 if name == "flip" else 1):
        ctx.decline(f"np.{name} with these arguments")
    axis_node = node.args[1] if len(node.args) > 1 else None
    for keyword in node.keywords:
        if keyword.arg != "axis" or name != "flip":
            ctx.decline(f"np.{name} with {keyword.arg}=")
        axis_node = keyword.value
    axis = ctx.info(axis_node).value if axis_node is not None else None
    if axis_node is not None and not isinstance(axis, int):
        ctx.decline("np.flip along a computed axis")
    data = node.args[0]
    root = view_root(data)
    if root is not None and root in ctx.scope.mutated:
        # The flipped array is a view: a later write into the original shows
        # through it in NumPy, never on the .m side.
        ctx.decline(f"np.{name} of {root}, which is written into (NumPy returns a view)")
    info = ctx.info(data)
    value = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    if info.is_vector or isinstance(data, ast.List):
        if name == "fliplr":
            ctx.decline("np.fliplr of a 1-D array (NumPy raises)")
        if axis not in (None, 0, -1):
            ctx.decline("np.flip of a 1-D array along axis 1")
        return ctx.call("fliplr" if name == "flipud" else "flip", value, loc=loc)
    if not info.is_matrix:
        ctx.decline(f"np.{name} of a value whose dimensions are unknown")
    if name == "flip":
        if axis is None:
            ctx.decline("np.flip of a 2-D array without an axis (it reverses both axes)")
        name = "flipud" if axis in (0, -2) else "fliplr"
    return ctx.call(name, value, loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """The data's facts; the result is a view of it."""
    if not args:
        return None
    data = as_array(args[0])
    if not data.is_array:
        return None
    return replace(data, value=None, view=True)
