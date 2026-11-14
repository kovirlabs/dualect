# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/linear-algebra: ``inv``, ``det``, ``trace``, ``rank``.

Each is the NumPy function of the same meaning on a square (``rank``: any)
matrix: ``np.linalg.inv``, ``np.linalg.det``, ``np.trace`` and
``np.linalg.matrix_rank``. ``rank`` uses the same default tolerance on both
sides (``max(size(A)) * eps`` times the largest singular value), and
``rank(A, tol)`` is ``matrix_rank(A, tol)``. A singular matrix makes ``inv``
warn and return ``Inf`` in the source but raise ``LinAlgError`` in NumPy,
which fails loudly. Values not known to be matrices are declined: ``inv`` and
``det`` of a scalar would be ``1 / x`` and ``x``, but NumPy's functions need
2-D input.
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

_NUMPY = {"inv": "linalg.inv", "det": "linalg.det", "trace": "trace", "rank": "linalg.matrix_rank"}
_TO_M = {
    "np.linalg.inv": "inv",
    "np.linalg.det": "det",
    "np.trace": "trace",
    "np.linalg.matrix_rank": "rank",
}


def infer_call(node: ir.Call, args: list[Info], nargout: int) -> Info | None:
    """A matrix for ``inv``, a scalar for the rest."""
    if not args or node.func not in _NUMPY:
        return None
    a = args[0]
    if node.func == "inv":
        return array(a.rows, a.cols, elem=a.elem, kind=Kind.MATRIX if a.is_matrix else None)
    if node.func == "rank":
        return replace(scalar(nonneg=True, integer=True), int_dtype=True)
    result = scalar(elem=a.elem if a.elem is not Elem.CHAR else Elem.UNKNOWN, finite=False)
    # np.trace of an integer matrix is an integer; det is always a float.
    return replace(result, int_dtype=node.func == "trace" and a.int_dtype)


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """``inv(A)``, ``det(A)``, ``trace(A)``, ``rank(A[, tol])``."""
    limit = 2 if node.func == "rank" else 1
    if not 1 <= len(node.args) <= limit:
        ctx.decline(f"{node.func} with these arguments")
    info = ctx.info(node.args[0])
    if not info.is_matrix:
        ctx.decline(f"{node.func} of a value not known to be a matrix")
    square = info.rows is None or info.cols is None or info.rows == info.cols
    if not square and node.func != "rank":
        ctx.decline(f"{node.func} of a matrix that is not square")
    args = [ctx.expr(arg) for arg in node.args]
    return ctx.call(ctx.np(_NUMPY[node.func]), *args)


# --- Python -> .m ------------------------------------------------------------


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.linalg.inv/det/matrix_rank`` and ``np.trace`` of a 2-D array."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    name = (ctx.tag(node) or "").removeprefix("py:call:")
    func = _TO_M.get(name)
    limit = 2 if func == "rank" else 1
    if func is None or node.keywords or not 1 <= len(node.args) <= limit:
        ctx.decline(f"{name} with these arguments")
    if not ctx.info(node.args[0]).is_matrix:
        ctx.decline(f"{name} of a value not known to be 2-D")
    args = [ctx.expr(arg) for arg in node.args]
    return ctx.call(func, *args, loc=loc)


def pyinfer_call(
    tag: str,
    node: ast.Call,
    args: list[PyInfo],
    keywords: dict[str, PyInfo],
    receiver: PyInfo | None,
) -> PyInfo | None:
    """A 2-D array for ``inv``; a scalar otherwise."""
    name = tag.removeprefix("py:call:")
    if name not in _TO_M or not args:
        return None
    a = args[0]
    if name == "np.linalg.inv":
        return py_array(PyKind.MATRIX, DType.COMPLEX if a.dtype is DType.COMPLEX else DType.FLOAT)
    if name == "np.linalg.matrix_rank":
        return py_scalar(DType.INT, nonneg=True)
    if name == "np.trace":
        return py_scalar(a.dtype, finite=a.finite)
    return py_scalar(DType.COMPLEX if a.dtype is DType.COMPLEX else DType.FLOAT, finite=False)
