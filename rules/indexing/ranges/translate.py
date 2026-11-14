# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Translator for indexing/ranges: ``a:b`` and ``a:step:b``.

As a value, an integer range is ``np.arange(a, b + 1)`` (the stop moves past
the inclusive end); a range with a fractional step is ``np.linspace`` with the
point count computed here, because ``np.arange`` with a float step can gain or
lose its end point. As a subscript, :func:`to_slice` gives the Python slice.
"""

import ast

from dualect import ir
from dualect.emit import Context, offset
from dualect.infer import range_count
from dualect.lift import LiftContext


def literal_int(node: ir.Expr | None) -> int | None:
    """The value of an integer literal, ``-k`` included."""
    if isinstance(node, ir.Num) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ir.UnOp) and node.op == "uminus":
        inner = literal_int(node.operand)
        return -inner if inner is not None else None
    return None


def translate(node: ir.Range, ctx: Context) -> ast.expr:
    """A range used as a value."""
    start, stop = ctx.info(node.start), ctx.info(node.stop)
    step_value = literal_int(node.step) if node.step is not None else 1
    if node.step is not None and step_value is None:
        step_info = ctx.info(node.step)
        if not (step_info.integer and step_info.is_scalar):
            return _linspace(node, ctx)
        ctx.decline("a range with a variable step")
    if start.integer and step_value is not None:
        if not stop.integer:
            if stop.value is not None:
                ctx.decline("a range whose bound is not an integer")
            ctx.assume("integer range bound")
        upper = offset(ctx.expr(node.stop), 1 if step_value > 0 else -1)
        args = [ctx.expr(node.start), upper]
        if step_value != 1:
            args.append(ctx.const(step_value))
        return ctx.call(ctx.np("arange"), *args)
    return _linspace(node, ctx)


def _linspace(node: ir.Range, ctx: Context) -> ast.expr:
    start = ctx.info(node.start).value
    stop = ctx.info(node.stop).value
    step = ctx.info(node.step).value if node.step is not None else 1
    count = range_count(start, step, stop)
    if count is None or not all(isinstance(v, int | float) for v in (start, step, stop)):
        ctx.decline("a range with a fractional or unknown step and unknown bounds")
    assert isinstance(start, int | float) and isinstance(step, int | float)
    last = start + (count - 1) * step
    if abs(last - float(stop)) > 1e-12 * max(1.0, abs(float(stop))):  # type: ignore[arg-type]
        ctx.decline("a fractional-step range that stops short of its bound")
    return ctx.call(
        ctx.np("linspace"), ctx.expr(node.start), ctx.expr(node.stop), ast.Constant(count)
    )


def to_slice(node: ir.Range, ctx: Context) -> ast.expr:
    """A range used as a subscript: the Python slice with the same elements."""
    ends = ctx.module("indexing/end-in-index")
    one_based = ctx.module("indexing/one-based")
    step = literal_int(node.step) if node.step is not None else 1
    if step is None or step == 0:
        ctx.decline("a subscript range with a variable step")
    lower: ast.expr | None
    upper: ast.expr | None
    if step > 0:
        lower = None if literal_int(node.start) == 1 else one_based.zero_based(node.start, ctx)
        stop_end = ends.end_offset(node.stop)
        if stop_end is not None:
            upper = None if stop_end == 0 else ctx.const(-stop_end)
        elif ends.end_minus(node.stop) is not None:
            upper = ast.UnaryOp(op=ast.USub(), operand=ctx.expr(ends.end_minus(node.stop)))
        else:
            # A 1-based inclusive stop is the same number as a 0-based exclusive one.
            upper = one_based.position_value(node.stop, ctx)
    else:
        lower = None if ends.end_offset(node.start) == 0 else one_based.zero_based(node.start, ctx)
        stop = literal_int(node.stop)
        if stop is None or stop < 1:
            ctx.decline("a descending subscript range with a non-literal stop")
        # Exclusive stop one below the 0-based inclusive stop; none when it is 0.
        upper = None if stop == 1 else ctx.const(stop - 2)
    return ast.Slice(
        lower=lower,
        upper=upper,
        step=None if step == 1 else ctx.const(step),
    )


# --- Python -> .m ------------------------------------------------------------
#
# np.arange(a, b) stops before b: a:b-1 (with a step s > 0, a:s:b-1; s < 0,
# a:s:b+1). np.arange with a float step can gain or lose its end point, so only
# integer arguments are translated. A slice x[a:b] is x(a+1:b): the 0-based
# start rises by one and the exclusive 0-based stop is the inclusive 1-based
# one. A missing stop is `end`; negative bounds count from the end.


def _step(node: ast.expr | None, ctx: LiftContext) -> int:
    if node is None:
        return 1
    value = node.value if isinstance(node, ast.Constant) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = node.operand
        value = (
            -inner.value if isinstance(inner, ast.Constant) and type(inner.value) is int else None
        )
    if type(value) is not int or value == 0:
        ctx.decline("a step that is not a non-zero integer literal")
    return value


def lift(node: ast.expr, ctx: LiftContext) -> ir.Expr:
    """``np.arange`` with integer arguments."""
    assert isinstance(node, ast.Call)
    loc = ctx.loc(node)
    args = node.args
    if node.keywords or not 1 <= len(args) <= 3:
        ctx.decline("np.arange with these arguments")
    if not all(ctx.info(a).dtype.value == "int" for a in args):
        ctx.decline("np.arange with arguments not known to be integers")
    if len(args) == 1:
        start: ir.Expr = ctx.num(0)
        stop_node = args[0]
    else:
        start = ctx.expr(args[0])
        stop_node = args[1]
    step = _step(args[2] if len(args) == 3 else None, ctx)
    stop = ctx.shift(ctx.expr(stop_node), -1 if step > 0 else 1)
    if step == 1:
        return ir.Range(loc=loc, start=start, stop=stop)
    return ir.Range(loc=loc, start=start, stop=stop, step=ctx.num(step))


def slice_range(
    part: ast.Slice, target: ir.Expr, dim: int, ndims: int, ctx: LiftContext
) -> ir.Expr:
    """A Python slice as the ``.m`` range of the same positions."""
    loc = ctx.loc(part)
    step = _step(part.step, ctx)
    one_based = ctx.module_for("indexing/one-based")
    ends = ctx.module_for("indexing/end-in-index")

    def bound(node: ast.expr, adjust: int) -> ir.Expr:
        """A 0-based (possibly negative) bound plus ``adjust``, as a 1-based value."""
        if one_based._negative(node):
            return ctx.shift(ends.from_end(node, target, dim, ndims, ctx), adjust - 1)
        info = ctx.info(node)
        if not info.nonneg:
            ctx.assume(f"non-negative slice bound {ast.unparse(node)}")
        return ctx.shift(ctx.expr(node), adjust)

    end = ir.End(loc=loc, target=target, dim=dim, ndims=ndims)
    if step > 0:
        start = ctx.num(1) if part.lower is None else bound(part.lower, 1)
        stop = end if part.upper is None else bound(part.upper, 0)
    else:
        start = end if part.lower is None else bound(part.lower, 1)
        stop = ctx.num(1) if part.upper is None else bound(part.upper, 2)
    if step == 1:
        return ir.Range(loc=loc, start=start, stop=stop)
    return ir.Range(loc=loc, start=start, stop=stop, step=ctx.num(step))
