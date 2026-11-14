# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for builtins/sum-mean-max: ``sum``, ``mean``, ``prod``, ``max``, ``min``.

A vector reduces to a scalar (``x.sum()``); a matrix reduces down its columns
(``axis=0``) unless a dimension is given (``d`` is ``axis=d - 1``). ``sum(A(:))``
is ``A.sum()``. ``max``/``min`` skip NaN, so data not known to be NaN-free uses
``np.nanmax``/``np.nanmin``; the two-argument form is ``np.maximum`` (or
``np.fmax``, which also skips NaN), and ``min(max(x, lo), hi)`` is
``np.clip(x, lo, hi)``. The position output of ``[m, i] = max(v)`` is
``np.argmax``, 0-based when the emitter stores ``i`` 0-based. ``prod`` of
an integer array is taken in floating point (``dtype=float``): NumPy integers
wrap around on overflow where the source computes in doubles.
"""

import ast

from dualect import ir
from dualect.emit import Context, Line
from dualect.infer import Info, Kind
from dualect.lift import LiftContext

_NAN_SKIPPING = {"max": "nanmax", "min": "nanmin"}
_PAIRWISE = {"max": ("maximum", "fmax"), "min": ("minimum", "fmin")}


def position_outputs(call: ir.Call, nargout: int) -> set[int]:
    """The second output of ``max``/``min`` of one argument is a position."""
    if call.func in ("max", "min") and len(call.args) == 1 and nargout >= 2:
        return {1}
    return set()


def _colon_all(node: ir.Expr) -> ir.Expr | None:
    """``A`` when ``node`` is ``A(:)``."""
    if isinstance(node, ir.Index) and len(node.args) == 1 and isinstance(node.args[0], ir.Colon):
        return node.target
    return None


def _dim(node: ir.Call, position: int, ctx: Context) -> int | None:
    if len(node.args) <= position:
        return None
    value = ctx.info(node.args[position]).value
    if not isinstance(value, int | float) or not float(value).is_integer():
        ctx.decline(f"{node.func} along a computed dimension")
    return int(value)


def _reduce(func: str, data: ir.Expr, info: Info, dim: int | None, ctx: Context) -> ast.expr:
    """``func`` of ``data`` over ``dim`` (``None``: the default dimension)."""
    expr = ctx.expr(data)
    nan_name = _NAN_SKIPPING.get(func) if not info.finite else None
    # A product of integers wraps around in NumPy; the source multiplies doubles.
    keywords = {"dtype": ast.Name("float")} if func == "prod" and info.int_dtype else {}
    if info.is_matrix:
        axis = 0 if dim in (None, 1) else 1 if dim == 2 else None
        if axis is None:
            ctx.decline(f"{func} along dimension {dim}")
        if nan_name:
            return ctx.call(ctx.np(nan_name), expr, axis=ast.Constant(axis))
        return ctx.method(expr, func, axis=ast.Constant(axis), **keywords)
    reduces = (
        dim is None or (dim == 2 and info.kind is Kind.ROW) or (dim == 1 and info.kind is Kind.COL)
    )
    if info.is_vector and reduces:
        if func == "sum" and info.is_logical:
            return ctx.call(ctx.np("count_nonzero"), expr)
        if nan_name:
            return ctx.call(ctx.np(nan_name), expr)
        return ctx.method(expr, func, **keywords)
    ctx.decline(f"{func} of a value whose shape is unknown or along a singleton dimension")


def translate(node: ir.Call, ctx: Context) -> ast.expr:
    """One reduction, or a two-argument ``max``/``min``."""
    func, args = node.func, node.args
    if not args:
        ctx.decline(f"{func} without arguments")
    if func in _PAIRWISE and len(args) == 2:
        return _pairwise(node, ctx)
    if func in ("max", "min"):
        if len(args) == 3 and not (isinstance(args[1], ir.Matrix) and not args[1].rows):
            ctx.decline(f"{func} with a second argument and a dimension")
        dim = _dim(node, 2, ctx)
    else:
        dim = _dim(node, 1, ctx)
    data = args[0]
    whole = _colon_all(data)
    if whole is not None and dim is None:
        info = ctx.info(whole)
        if not (info.is_array or info.kind is Kind.UNKNOWN):
            ctx.decline(f"{func}(x(:)) of a non-array")
        nan_name = _NAN_SKIPPING.get(func) if not info.finite else None
        if nan_name:
            return ctx.call(ctx.np(nan_name), ctx.expr(whole))
        if func == "prod" and info.int_dtype:
            return ctx.method(ctx.expr(whole), func, dtype=ast.Name("float"))
        return ctx.method(ctx.expr(whole), func)
    return _reduce(func, data, ctx.info(data), dim, ctx)


def _pairwise(node: ir.Call, ctx: Context) -> ast.expr:
    a, b = node.args
    finite = ctx.info(a).finite and ctx.info(b).finite
    # min(max(x, lo), hi) clamps x: np.clip, when no NaN can change the meaning.
    inner = a if node.func == "min" else None
    if (
        isinstance(inner, ir.Call)
        and inner.func == "max"
        and len(inner.args) == 2
        and all(ctx.info(arg).finite for arg in [*inner.args, b])
    ):
        x, lo = inner.args
        return ctx.call(ctx.np("clip"), ctx.arg(x), ctx.arg(lo), ctx.arg(b))
    plain, nan_skipping = _PAIRWISE[node.func]
    return ctx.call(ctx.np(plain if finite else nan_skipping), ctx.arg(a), ctx.arg(b))


def translate_multi(stmt: ir.Assign, ctx: Context) -> list[Line]:
    """``[m, i] = max(v)``: the value and its first position."""
    call = stmt.value
    assert isinstance(call, ir.Call)
    if call.func not in ("max", "min") or len(call.args) != 1 or len(stmt.targets) != 2:
        ctx.decline(f"{call.func} with these outputs")
    data = call.args[0]
    info = ctx.info(data)
    if not info.is_vector:
        ctx.decline(f"[m, i] = {call.func}(x) of a matrix or unknown shape")
    value_target, position_target = stmt.targets
    expr = ctx.expr(data)
    arg_func = ("nanarg" if not info.finite else "arg") + call.func
    position = ctx.call(ctx.np(arg_func), expr)
    lines: list[Line] = []
    position_name = position_target.id if isinstance(position_target, ir.Name) else None
    stored_zero_based = position_name is not None and ctx.is_index_var(position_name)
    if position_name is not None:
        if stored_zero_based:
            lines.extend(ctx.index_directive(position_name))
            lines.append(ctx.line(f"{position_name} = {ast.unparse(position)}"))
        else:
            lines.append(ctx.line(f"{position_name} = {ast.unparse(position)} + 1"))
    if isinstance(value_target, ir.Name):
        value: ast.expr
        if stored_zero_based and position_name is not None:
            value = ast.Subscript(value=expr, slice=ast.Name(position_name))
        else:
            value = _reduce(call.func, data, info, None, ctx)
        lines.append(ctx.line(f"{value_target.id} = {ast.unparse(value)}"))
    elif not isinstance(value_target, ir.Ignore):
        ctx.decline("assigning a reduction into an element")
    if isinstance(position_target, ir.Index):
        ctx.decline("assigning a position into an element")
    return lines


# --- Python -> .m ------------------------------------------------------------
#
# x.sum() of a 1-D array is sum(x); of a 2-D array (all elements) sum(A(:)).
# axis=0 reduces down the columns, sum(A, 1), a row; axis=1 is sum(A, 2), a
# column on the .m side and 1-D in NumPy, so it is turned into a row. The
# .m max/min skip NaN where NumPy's max propagates it, so .max() of data not
# known to be NaN-free is declined and np.nanmax is plain max. np.argmax has no
# expression form: i = np.argmax(v) becomes [~, i] = max(v); i = i - 1, kept
# 0-based so the value equals Python's. np.fmax (NaN-skipping) is max(a, b);
# np.maximum is too when no NaN can occur; np.clip(x, lo, hi) is
# min(max(x, lo), hi); np.count_nonzero is nnz.

_SKIPS_NAN = {"nanmax": "max", "nanmin": "min", "fmax": "max", "fmin": "min"}
_ARG = {"argmax": "max", "argmin": "min", "nanargmax": "max", "nanargmin": "min"}


def _name(node: ast.Call, ctx: LiftContext) -> tuple[str, ast.expr | None, list[ast.expr]]:
    """``(function, receiver or first argument, remaining arguments)`` of a call."""
    tag = ctx.tag(node) or ""
    if tag.startswith("py:method:"):
        assert isinstance(node.func, ast.Attribute)
        return tag.removeprefix("py:method:"), node.func.value, list(node.args)
    name = tag.split(":")[-1].removeprefix("np.")
    if not node.args:
        return name, None, []
    return name, node.args[0], list(node.args[1:])


def _axis(node: ast.Call, rest: list[ast.expr], ctx: LiftContext) -> int | None:
    axis_node = rest[0] if rest else None
    for keyword in node.keywords:
        if keyword.arg == "axis":
            axis_node = keyword.value
        elif keyword.arg == "dtype" and isinstance(keyword.value, ast.Name):
            # dtype=float is the .m side's own double arithmetic (and what the
            # other direction writes for integer products); others are not.
            if keyword.value.id != "float":
                ctx.decline(f"dtype={keyword.value.id} in a reduction")
        else:
            ctx.decline(f"{keyword.arg}= in a reduction")
    if axis_node is None:
        return None
    value = ctx.info(axis_node).value
    if value not in (0, 1, -1, -2) or isinstance(value, bool):
        ctx.decline("a reduction along a computed axis")
    assert isinstance(value, int)
    return value % 2


def reduce(func: str, data: ast.expr, axis: int | None, ctx: LiftContext, loc: ir.Loc) -> ir.Expr:
    """``func`` of ``data`` (over ``axis``) as the ``.m`` reduction of the same values."""
    info = ctx.info(data)
    value = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    if info.is_vector or isinstance(data, ast.List):
        if axis not in (None, 0):
            ctx.decline("a reduction of a 1-D array along axis 1")
        return ctx.call(func, value, loc=loc)
    if not info.is_matrix:
        ctx.decline(f"{func} of a value whose dimensions are unknown")
    if axis is None:
        return ctx.call(func, ir.Index(loc=loc, target=value, args=[ir.Colon(loc=loc)]), loc=loc)
    if func in ("max", "min"):
        result = ctx.call(func, value, ir.Matrix(loc=loc, rows=[]), ctx.num(axis + 1), loc=loc)
    else:
        result = ctx.call(func, value, ctx.num(axis + 1), loc=loc)
    return result if axis == 0 else ctx.transpose(result)


def _finite(node: ast.expr, ctx: LiftContext) -> bool:
    if isinstance(node, ast.List):
        return all(_finite(e, ctx) for e in node.elts)
    return ctx.info(node).finite


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """A reduction, a pairwise max/min, ``np.clip`` or ``np.count_nonzero``."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    literals = ctx.module_for("syntax/array-literals")
    func, data, rest = _name(node, ctx)
    tag = ctx.tag(node) or ""
    if func in _ARG:
        ctx.decline(f"np.{func} inside an expression (it needs its own statement)")
    if tag in ("py:call:max", "py:call:min"):
        return _builtin(node, func, ctx)
    if data is None:
        ctx.decline(f"{func} without arguments")
    if func in ("maximum", "minimum", "fmax", "fmin"):
        if len(rest) != 1 or node.keywords:
            ctx.decline(f"np.{func} with these arguments")
        if func in ("maximum", "minimum") and not (_finite(data, ctx) and _finite(rest[0], ctx)):
            ctx.decline(f"np.{func} of data that may hold NaN (the .m {func[:3]} skips NaN)")
        name = func[:3] if func in ("maximum", "minimum") else _SKIPS_NAN[func]
        return ctx.call(
            name, literals.lift_items(data, ctx), literals.lift_items(rest[0], ctx), loc=loc
        )
    if func == "clip":
        if len(rest) != 2 or node.keywords or not _finite(data, ctx):
            ctx.decline("np.clip of data that may hold NaN, or with these arguments")
        if any(isinstance(b, ast.Constant) and b.value is None for b in rest):
            ctx.decline("np.clip with an open bound")
        low = ctx.call("max", literals.lift_items(data, ctx), ctx.expr(rest[0]), loc=loc)
        return ctx.call("min", low, ctx.expr(rest[1]), loc=loc)
    if func == "count_nonzero":
        if rest or node.keywords:
            ctx.decline("np.count_nonzero along an axis")
        return ctx.call("nnz", literals.lift_items(data, ctx), loc=loc)
    axis = _axis(node, rest, ctx)
    if func in ("max", "min") and not _finite(data, ctx):
        ctx.decline(
            f"{func} of data that may hold NaN (NumPy propagates it, the .m {func} skips it)"
        )
    return reduce(_SKIPS_NAN.get(func, func), data, axis, ctx, loc)


def _builtin(node: ast.Call, func: str, ctx: LiftContext) -> ir.Expr:
    """Python's own ``max``/``min``: of ``A.shape``, of a list, or of scalars."""
    loc = ctx.loc(node)
    if node.keywords or not node.args:
        ctx.decline(f"{func}() with these arguments")
    if len(node.args) == 1:
        arg = node.args[0]
        if isinstance(arg, ast.Attribute) and arg.attr == "shape":
            shape = ctx.module_for("builtins/size-numel-length").lift_shape(arg, ctx)
            return ctx.call(func, shape, loc=loc)
        if not _finite(arg, ctx):
            ctx.decline(f"{func}() of data that may hold NaN")
        return reduce(func, arg, None, ctx, loc)
    if not all(ctx.info(a).is_scalar and ctx.info(a).finite for a in node.args):
        ctx.decline(f"{func}() of values that are not finite scalars")
    result = ctx.expr(node.args[0])
    for arg in node.args[1:]:
        result = ctx.call(func, result, ctx.expr(arg), loc=loc)
    return result


def lift_assign(stmt: ast.Assign, ctx: LiftContext) -> list[ir.Stmt] | None:
    """``i = np.argmax(v)``: ``[~, i] = max(v); i = i - 1`` (0-based, as in Python)."""
    value = stmt.value
    target = stmt.targets[0]
    assert isinstance(value, ast.Call)
    func, data, rest = _name(value, ctx)
    if func not in _ARG:
        return None
    if data is None or rest or value.keywords or not isinstance(target, ast.Name):
        ctx.decline(f"{func} with these arguments")
    info = ctx.info(data)
    if not (info.is_vector or isinstance(data, ast.List)):
        ctx.decline(f"{func} of a value that is not 1-D (NumPy flattens it in row order)")
    if not func.startswith("nan") and not _finite(data, ctx):
        ctx.decline(f"{func} of data that may hold NaN (NumPy stops at the first NaN)")
    loc = ctx.loc(stmt)
    name = ir.Name(loc=ctx.loc(target), id=target.id)
    values = ctx.module_for("syntax/array-literals").lift_items(data, ctx)
    found = ir.Assign(
        loc=loc, targets=[ir.Ignore(loc=loc), name], value=ctx.call(_ARG[func], values, loc=loc)
    )
    shifted = ir.Assign(
        loc=loc,
        targets=[ir.Name(loc=loc, id=target.id)],
        value=ctx.binop("minus", ir.Name(loc=loc, id=target.id), ctx.num(1)),
    )
    return [found, shifted]
