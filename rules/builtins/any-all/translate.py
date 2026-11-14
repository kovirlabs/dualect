# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/any-all: ``any`` and ``all``.

They reduce like ``sum`` (builtins/sum-mean-max does the work): a vector to a
single logical (``v.any()``), a matrix down its columns (``axis=0``) unless a
dimension is given, ``any(A(:))`` over every element. ``any`` does not count
NaN as true in the source while NumPy does, so ``any`` of data that may hold
NaN is declined; ``all`` treats NaN as nonzero in both.

From Python, ``x.any()``/``np.any(x)`` and ``all`` reduce the same way (the
same helpers as ``sum``), with the same NaN check for ``any``.
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind, array, scalar
from dualect.lift import LiftContext


def _data(node: ir.Call) -> ir.Expr:
    """The reduced data (``A`` of ``any(A(:))``)."""
    data = node.args[0]
    if isinstance(data, ir.Index) and len(data.args) == 1 and isinstance(data.args[0], ir.Colon):
        return data.target
    return data


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``any(x)``, ``all(x)``, ``any(x, dim)``."""
    if not node.args or len(node.args) > 2:
        ctx.decline(f"{node.func} with these arguments")
    info = ctx.info(_data(node))
    if node.func == "any" and not info.finite:
        ctx.decline("any of data that may hold NaN (the source ignores NaN, NumPy counts it)")
    if info.elem not in (Elem.REAL, Elem.LOGICAL):
        ctx.decline(f"{node.func} of values that may be complex or characters")
    result: ast.expr = ctx.module("builtins/sum-mean-max").translate(node, ctx)
    return result


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """A logical scalar per vector, a row per matrix (a column along dimension 2)."""
    if not args:
        return None
    first = args[0]
    facts = {"elem": Elem.LOGICAL, "finite": True, "nonneg": True, "integer": True}
    dim = args[1].value if len(args) > 1 else None
    if first.is_matrix and dim in (None, 1):
        return array(1, first.cols, kind=Kind.ROW, **facts)  # type: ignore[arg-type]
    if first.is_matrix and dim == 2:
        return array(first.rows, 1, kind=Kind.COL, **facts)  # type: ignore[arg-type]
    if first.is_vector and dim is None:
        return scalar(**facts)  # type: ignore[arg-type]
    if first.is_scalar:
        return scalar(**facts)  # type: ignore[arg-type]
    return None


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``x.any()``, ``np.all(x, axis=...)``: the reduction rule's lift."""
    assert isinstance(node, ast.Call)
    reductions = ctx.module_for("builtins/sum-mean-max")
    func, data, _rest = reductions._name(node, ctx)
    if data is None:
        ctx.decline(f"{func} without arguments")
    if func == "any" and not reductions._finite(data, ctx):
        ctx.decline("any of data that may hold NaN (NumPy counts NaN as true, the .m any not)")
    result: ir.Expr = reductions.lift(node, ctx)
    return result
