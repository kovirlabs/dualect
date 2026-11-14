# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/norm: vector and matrix norms.

``norm(v)`` of a vector is the Euclidean length, ``np.linalg.norm(v)``, and
``norm(v, p)`` the p-norm, ``np.linalg.norm(v, p)`` (``Inf`` is ``np.inf``;
``'fro'`` of a vector is its 2-norm). For a matrix the defaults differ:
``norm(A)`` is the *spectral* norm (largest singular value),
``np.linalg.norm(A, 2)``, while NumPy's default for a 2-D array is the
Frobenius norm, which is ``norm(A, 'fro')``. ``norm(A, 1)`` and ``norm(A,
Inf)`` are the largest column and row sums on both sides; other matrix
p-norms are declined (NumPy has none).

From Python, ``np.linalg.norm(x[, ord])`` maps back the same way; ``axis=``
and the nuclear norm are declined.
"""

import ast
import math

from dualect import ir
from dualect.emit import Context
from dualect.infer import Info, scalar
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo
from dualect.pyinfer import scalar as py_scalar


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """A non-negative real scalar."""
    if not args:
        return None
    return scalar(finite=args[0].finite, nonneg=True)


def _order(node: ir.Expr, ctx: Context) -> float | str:
    if isinstance(node, ir.Str):
        text = node.value.lower()
        if text in ("fro", "inf", "-inf"):
            return text
        ctx.decline(f"norm with the option {node.value!r}")
    value = ctx.info(node).value
    if not isinstance(value, int | float) or isinstance(value, bool):
        ctx.decline("norm with a computed order")
    return float(value)


def _ord(value: float, ctx: Context) -> ast.expr:
    if math.isinf(value):
        return ctx.np("inf") if value > 0 else ast.UnaryOp(op=ast.USub(), operand=ctx.np("inf"))
    return ctx.const(int(value) if value.is_integer() else value)


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``norm(x)`` / ``norm(x, p)``."""
    if len(node.args) not in (1, 2):
        ctx.decline("norm with these arguments")
    info = ctx.info(node.args[0])
    order: float | str | None = _order(node.args[1], ctx) if len(node.args) == 2 else None
    if order in ("inf", "-inf"):
        order = math.inf if order == "inf" else -math.inf
    func = ctx.np("linalg.norm")
    x = ctx.arg(node.args[0])
    if info.is_vector or info.is_scalar:
        if order is None or order == "fro" or order == 2:
            return ctx.call(func, x)
        assert isinstance(order, float)
        if order <= 0 and not math.isinf(order):
            ctx.decline("a vector norm of order 0 or below")
        return ctx.call(func, x, _ord(order, ctx))
    if not info.is_matrix:
        ctx.decline("norm of a value whose shape (vector or matrix) is unknown")
    if order == "fro":
        return ctx.call(func, x)
    if order is None or order == 2:
        return ctx.call(func, x, ast.Constant(2))  # the spectral norm, not NumPy's default
    assert isinstance(order, float)
    if order in (1, math.inf):
        return ctx.call(func, x, _ord(order, ctx))
    ctx.decline("a matrix p-norm other than 1, 2, Inf and 'fro'")


# --- Python -> .m ------------------------------------------------------------


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.linalg.norm(x[, ord])``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    args = list(node.args)
    for keyword in node.keywords:
        if keyword.arg != "ord" or len(args) != 1:
            ctx.decline(f"np.linalg.norm with {keyword.arg}=")
        args.append(keyword.value)
    if len(args) not in (1, 2):
        ctx.decline("np.linalg.norm with these arguments")
    data = args[0]
    info = ctx.info(data)
    value = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    order_node = args[1] if len(args) == 2 else None
    order: object = None
    if order_node is not None:
        if isinstance(order_node, ast.Constant) and order_node.value is None:
            order_node = None
        else:
            order = ctx.info(order_node).value
            if isinstance(order_node, ast.Constant) and isinstance(order_node.value, str):
                order = order_node.value
            if order is None:
                ctx.decline("np.linalg.norm with a computed ord")
    one_d = info.is_vector or isinstance(data, ast.List)
    if not (one_d or info.is_matrix):
        ctx.decline("np.linalg.norm of a value whose dimensions are unknown")
    if order is None:
        if one_d:
            return ctx.call("norm", value, loc=loc)
        return ctx.call("norm", value, ir.Str(loc=loc, value="fro", quote="'"), loc=loc)
    if order == "fro" and not one_d:
        return ctx.call("norm", value, ir.Str(loc=loc, value="fro", quote="'"), loc=loc)
    if not isinstance(order, int | float) or isinstance(order, bool):
        ctx.decline(f"np.linalg.norm with ord={order!r}")
    if not one_d and order not in (1, 2, math.inf):
        ctx.decline(f"a matrix norm with ord={order!r}")
    if one_d and order <= 0 and not math.isinf(order):
        ctx.decline("a vector norm of order 0 or below")
    if order == 2:
        return ctx.call("norm", value, loc=loc)
    if math.isinf(order):
        inf = ctx.call("Inf", loc=loc)
        p = inf if order > 0 else ir.UnOp(loc=loc, op="uminus", operand=inf)
        return ctx.call("norm", value, p, loc=loc)
    return ctx.call("norm", value, ctx.num(order), loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """A non-negative float."""
    if tag != "py:call:np.linalg.norm" or not args:
        return None
    return py_scalar(DType.FLOAT, finite=args[0].finite, nonneg=True)
