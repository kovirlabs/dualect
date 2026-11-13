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
