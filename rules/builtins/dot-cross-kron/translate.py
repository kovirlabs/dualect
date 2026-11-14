# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/dot-cross-kron: ``dot``, ``cross`` and ``kron``.

``dot(a, b)`` of two real vectors (either orientation) is ``np.dot(a, b)``.
The source conjugates its first argument, ``np.dot`` does not
(``np.vdot`` does), and ``dot`` of matrices works column by column where
``np.dot`` is a matrix product, so complex data and matrices are declined.
``cross(a, b)`` of two 3-element vectors of the same orientation is
``np.cross(a, b)``; NumPy also accepts 2-element vectors (returning a
scalar) where the source raises, so an unknown length is assumed to be 3,
out loud.

``kron`` needs the orientation NumPy's 1-D arrays drop: two rows, two columns
or two matrices are ``np.kron(a, b)``; a column and a row (in either order)
make the matrix ``np.outer(c, r)``; a column next to a matrix gets its column
axis back, ``c[:, np.newaxis]``.

From Python, ``np.dot`` and ``np.cross`` of 1-D arrays and ``np.kron`` (a 1-D
array is a row on both sides) map back.
"""

import ast
from dataclasses import replace

from dualect import ir
from dualect.emit import Context
from dualect.infer import Elem, Info, Kind, array, scalar
from dualect.lift import LiftContext
from dualect.pyinfer import DType, PyInfo, PyKind
from dualect.pyinfer import array as py_array
from dualect.pyinfer import scalar as py_scalar


def _times(a: int | None, b: int | None) -> int | None:
    return a * b if a is not None and b is not None else None


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """A scalar for ``dot``, the shape of ``a`` for ``cross``, the product for ``kron``."""
    if len(args) != 2:
        return None
    a, b = args
    result = _shape(node, a, b)
    # NumPy keeps integer inputs integer (np.kron, np.cross, np.outer, np.dot).
    return replace(result, int_dtype=a.int_dtype and b.int_dtype) if result else None


def _shape(node: ir.Call, a: Info, b: Info) -> Info | None:
    elem = Elem.REAL if a.is_real and b.is_real else Elem.UNKNOWN
    finite = a.finite and b.finite
    if node.func == "dot":
        return scalar(elem=elem, finite=finite) if a.is_vector and b.is_vector else None
    if node.func == "cross":
        return array(a.rows, a.cols, elem=elem, finite=finite, kind=a.kind)
    if node.func == "kron":
        if a.kind in (Kind.UNKNOWN, Kind.ARRAY, Kind.VECTOR):
            return None
        if b.kind in (Kind.UNKNOWN, Kind.ARRAY, Kind.VECTOR):
            return None
        rows, cols = _times(a.rows, b.rows), _times(a.cols, b.cols)
        kind = None
        if {a.kind, b.kind} == {Kind.ROW, Kind.COL} or Kind.MATRIX in (a.kind, b.kind):
            kind = Kind.MATRIX if rows != 1 and cols != 1 else None
        return array(rows, cols, elem=elem, finite=finite, nonneg=a.nonneg and b.nonneg, kind=kind)
    return None


def _column(expr: ast.expr, ctx: Context) -> ast.expr:
    return ast.Subscript(value=expr, slice=ast.Tuple(elts=[ast.Slice(), ctx.np("newaxis")]))


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``dot``, ``cross`` or ``kron`` of two arguments."""
    if len(node.args) != 2:
        ctx.decline(f"{node.func} with a dimension argument")
    a_node, b_node = node.args
    a, b = ctx.info(a_node), ctx.info(b_node)
    x, y = ctx.arg(a_node), ctx.arg(b_node)
    if node.func == "dot":
        if not (a.is_vector and b.is_vector):
            ctx.decline("dot of values not known to be vectors (matrices go column by column)")
        if not (a.is_real and b.is_real):
            ctx.decline("dot of values that may be complex (the source conjugates the first)")
        return ctx.call(ctx.np("dot"), x, y)
    if node.func == "cross":
        if not (a.kind is b.kind and a.kind in (Kind.ROW, Kind.COL)):
            ctx.decline("cross of vectors not known to share an orientation")
        if a.numel not in (3, None) or b.numel not in (3, None):
            ctx.decline("cross of vectors that do not have 3 elements")
        if a.numel is None or b.numel is None:
            ctx.assume("3-element vectors (NumPy also accepts 2)")
        return ctx.call(ctx.np("cross"), x, y)
    # kron
    known = (Kind.SCALAR, Kind.ROW, Kind.COL, Kind.MATRIX)
    if a.kind not in known or b.kind not in known:
        ctx.decline("kron of values whose orientation (row, column or matrix) is unknown")
    if {a.kind, b.kind} == {Kind.ROW, Kind.COL}:
        column, row = (x, y) if a.kind is Kind.COL else (y, x)
        return ctx.call(ctx.np("outer"), column, row)
    if a.kind is Kind.COL and b.kind is Kind.MATRIX:
        x = _column(x, ctx)
    if b.kind is Kind.COL and a.kind is Kind.MATRIX:
        y = _column(y, ctx)
    return ctx.call(ctx.np("kron"), x, y)


# --- Python -> .m ------------------------------------------------------------


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.dot`` and ``np.cross`` of 1-D arrays, ``np.kron``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    func = (ctx.tag(node) or "").removeprefix("py:call:np.")
    if len(node.args) != 2 or node.keywords:
        ctx.decline(f"np.{func} with these arguments")
    literals = ctx.module_for("syntax/array-literals")
    infos = [ctx.info(arg) for arg in node.args]
    one_d = [
        i.is_vector or isinstance(arg, ast.List) for i, arg in zip(infos, node.args, strict=True)
    ]
    values = [literals.lift_items(arg, ctx) for arg in node.args]
    if func in ("dot", "cross"):
        if not all(one_d):
            ctx.decline(f"np.{func} of values not known to be 1-D")
        if (
            func == "dot"
            and any(i.dtype in (DType.COMPLEX, DType.UNKNOWN) for i in infos)
            and not all(isinstance(arg, ast.List) for arg in node.args)
        ):
            ctx.decline("np.dot of data that may be complex (the .m dot conjugates)")
        if func == "cross":
            lengths = {i.length for i in infos}
            if lengths - {3, None}:
                ctx.decline("np.cross of vectors that do not have 3 elements")
            if None in lengths:
                ctx.assume("3-element vectors (the .m cross rejects others)")
        return ctx.call(func, *values, loc=loc)
    if not all(o or i.is_matrix or i.is_scalar for o, i in zip(one_d, infos, strict=True)):
        ctx.decline("np.kron of values whose dimensions are unknown")
    return ctx.call("kron", *values, loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """``np.dot`` of 1-D arrays is a scalar; ``np.cross`` keeps the shape; ``np.kron``."""
    if len(args) != 2:
        return None
    a, b = args
    # NumPy's result type: integer (or bool) inputs stay integer, which a later
    # store of a fraction must know about.
    order = [DType.BOOL, DType.INT, DType.FLOAT, DType.COMPLEX]
    if DType.UNKNOWN in (a.dtype, b.dtype):
        dtype = DType.UNKNOWN
    else:
        dtype = max(a.dtype, b.dtype, key=order.index)
    if tag == "py:call:np.dot" and a.kind is PyKind.VECTOR and b.kind is PyKind.VECTOR:
        return py_scalar(dtype, finite=a.finite and b.finite)
    if tag == "py:call:np.cross":
        return py_array(PyKind.VECTOR, dtype, length=a.length)
    if tag == "py:call:np.kron":
        kind = PyKind.MATRIX if PyKind.MATRIX in (a.kind, b.kind) else PyKind.VECTOR
        return py_array(kind, dtype, finite=a.finite and b.finite)
    return None
