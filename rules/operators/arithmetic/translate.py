# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for operators/arithmetic: ``+``, ``-``, unary minus, comparisons.

These are element-wise in both languages. The one difference is implicit
expansion against a *column*: NumPy vectors are 1-D, so a column operand that
must stretch across columns gets an explicit axis, ``c[:, np.newaxis]``.
:func:`expand` is shared with the other element-wise operators.

Logical values are numbers in the source (``true + true`` is 2) but NumPy
keeps ``bool + bool`` as ``bool`` (a logical OR) and ``bool * bool`` as an
AND, so when both operands are logical :func:`numeric` turns the left one
into integers first (``m.astype(int) + m``).
"""

import ast

from dualect import ir
from dualect.emit import Context
from dualect.infer import Info, Kind
from dualect.lift import LiftContext

_BINARY: dict[str, ast.operator] = {"plus": ast.Add(), "minus": ast.Sub()}
_COMPARE: dict[str, ast.cmpop] = {
    "eq": ast.Eq(),
    "ne": ast.NotEq(),
    "lt": ast.Lt(),
    "le": ast.LtE(),
    "gt": ast.Gt(),
    "ge": ast.GtE(),
}


def _column(expr: ast.expr, ctx: Context) -> ast.expr:
    """``expr[:, np.newaxis]``: a 1-D vector as a column that broadcasts across columns."""
    return ast.Subscript(value=expr, slice=ast.Tuple(elts=[ast.Slice(), ctx.np("newaxis")]))


def numeric(expr: ast.expr, left: Info, right: Info, ctx: Context) -> ast.expr:
    """The left operand of an arithmetic operator, as numbers when both sides are logical.

    :param expr: the translated left operand.
    :param left: facts about the left operand.
    :param right: facts about the right operand.
    """
    if not (left.is_logical and right.is_logical):
        return expr
    if left.is_array:
        return ctx.method(expr, "astype", ast.Name("int"))
    if left.is_scalar:
        # A NumPy bool scalar (`m(1)`) adds like a bool array; int() is exact.
        return ctx.call(ast.Name("int"), expr)
    ctx.decline("arithmetic on logical values of unknown shape")


def expand(node: ir.BinOp, ctx: Context) -> tuple[ast.expr, ast.expr]:
    """Both operands of an element-wise operator, with columns given their axis.

    A column against a row (or a matrix) needs ``[:, np.newaxis]``; two vectors
    of unknown orientation are assumed to have the same one.
    """
    left, right = ctx.info(node.left), ctx.info(node.right)
    a, b = ctx.expr(node.left), ctx.expr(node.right)
    if left.is_scalar or right.is_scalar:
        return a, b
    kinds = {left.kind, right.kind}
    if kinds == {Kind.ROW, Kind.COL} or kinds == {Kind.MATRIX, Kind.COL}:
        if left.kind is Kind.COL:
            a = _column(a, ctx)
        else:
            b = _column(b, ctx)
        return a, b
    if left.kind is right.kind and left.kind in (Kind.ROW, Kind.COL, Kind.MATRIX, Kind.EMPTY):
        return a, b
    if kinds <= {Kind.MATRIX, Kind.ROW, Kind.EMPTY}:
        return a, b
    if Kind.VECTOR in kinds and kinds & {Kind.ROW, Kind.COL, Kind.MATRIX}:
        # Orientation differs between uses; broadcasting would differ too.
        ctx.decline("an element-wise operation on a vector whose orientation is not known")
    if left.kind is Kind.STRING or right.kind is Kind.STRING:
        ctx.decline("arithmetic or comparison on char arrays")
    ctx.assume("operands of matching shape")
    return a, b


def translate(node: ir.BinOp | ir.UnOp, ctx: Context) -> ast.expr:
    """``a + b``, ``a - b``, ``-a``, ``+a`` and the comparisons."""
    if isinstance(node, ir.UnOp):
        info = ctx.info(node.operand)
        # `-m` and `+m` of a logical value are numbers in the source.
        operand = numeric(ctx.expr(node.operand), info, info, ctx)
        if node.op == "uplus":
            return operand
        if node.op == "uminus":
            return ast.UnaryOp(op=ast.USub(), operand=operand)
        ctx.decline(f"unary {node.op}")
    a, b = expand(node, ctx)
    if node.op in _BINARY:
        a = numeric(a, ctx.info(node.left), ctx.info(node.right), ctx)
        return ast.BinOp(left=a, op=_BINARY[node.op], right=b)
    if node.op in _COMPARE:
        return ast.Compare(left=a, ops=[_COMPARE[node.op]], comparators=[b])
    ctx.decline(f"operator {node.op}")


# --- Python -> .m ------------------------------------------------------------
#
# The same operators. NumPy's bool + bool is a logical OR (and -mask raises),
# so arithmetic on two boolean arrays is declined. String + string is a
# concatenation and == on strings a comparison of the whole text: both go to
# types/char-strings. c[:, np.newaxis] is a 1-D vector stood up as a column.

_FROM_PY = {ast.Add: "plus", ast.Sub: "minus"}
_FROM_PY_CMP = {
    ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge",
}  # fmt: skip


def both_bool_arrays(left: ast.expr, right: ast.expr, ctx: LiftContext) -> bool:
    """Whether NumPy would combine two boolean values as booleans, not numbers."""
    a, b = ctx.info(left), ctx.info(right)
    return a.is_bool and b.is_bool and (a.is_array or b.is_array)


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``+``, ``-``, unary ``-``/``+``, the comparisons and ``c[:, np.newaxis]``."""
    loc = ctx.loc(node)
    if isinstance(node, ast.UnaryOp):
        info = ctx.info(node.operand)
        if info.is_bool and info.is_array:
            ctx.decline("negating a boolean array")
        operand = ctx.expr(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        return ir.UnOp(loc=loc, op="uminus", operand=operand)
    if isinstance(node, ast.BinOp):
        a, b = ctx.info(node.left), ctx.info(node.right)
        if a.kind.value == "string" or b.kind.value == "string":
            return ctx.module_for("types/char-strings").lift_concat(node, ctx)
        if both_bool_arrays(node.left, node.right, ctx):
            ctx.decline("arithmetic on boolean arrays (NumPy keeps bool + bool as a logical OR)")
        op = _FROM_PY.get(type(node.op))
        if op is None:
            ctx.decline(f"operator {type(node.op).__name__}")
        return ctx.binop(op, ctx.expr(node.left), ctx.expr(node.right), loc=loc)
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            ctx.decline("a chained comparison")
        left, right = node.left, node.comparators[0]
        if ctx.info(left).kind.value == "string" or ctx.info(right).kind.value == "string":
            return ctx.module_for("types/char-strings").lift_compare(node, ctx)
        op = _FROM_PY_CMP.get(type(node.ops[0]))
        if op is None:
            ctx.decline(f"comparison {type(node.ops[0]).__name__}")
        return ctx.binop(op, ctx.expr(left), ctx.expr(right), loc=loc)
    if isinstance(node, ast.Attribute):  # np.newaxis on its own
        ctx.decline("np.newaxis outside a subscript")
    ctx.decline(f"{type(node).__name__}")


def lift_newaxis(node: ast.Subscript, ctx: LiftContext) -> ir.Expr:
    """``c[:, np.newaxis]`` is ``c.'`` (a column); ``c[np.newaxis, :]`` is ``c`` (a row)."""
    parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    if not ctx.info(node.value).is_vector or len(parts) != 2:
        ctx.decline("a new axis on a value that is not 1-D")
    whole = [
        isinstance(p, ast.Slice) and p.lower is None and p.upper is None and p.step is None
        for p in parts
    ]
    if whole[0] and not whole[1]:
        return ctx.transpose(ctx.expr(node.value))
    if whole[1] and not whole[0]:
        return ctx.expr(node.value)
    ctx.decline("this new-axis subscript")
