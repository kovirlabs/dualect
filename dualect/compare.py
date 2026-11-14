# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Deciding whether a ``.m`` value and a Python value are the same result.

The comparison is deliberately about *values*, not representation:

- Numbers are compared with :func:`numpy.allclose` (NaN equals NaN).
- Singleton dimensions are ignored: Octave has no 1-D arrays, so its ``1xN`` row
  vector and an idiomatic NumPy 1-D array of length N are the same result.
  Non-singleton shapes must match exactly (a 2x3 is not a 3x2).
- Two empty arrays are equal whatever their shapes (``[]`` is 0x0 in Octave,
  ``np.array([])`` is shape ``(0,)``).
- Logical vs numeric *is* a difference: indexing with a logical mask and with
  the numbers 0/1 do different things, so a translation that loses the type is
  wrong even when the digits agree.
- Integer vs float is not: Octave computes in double by default where NumPy
  idiom often yields integers.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from dualect.runners import CallableValue

#: Default relative tolerance. Both sides use IEEE doubles and often the same
#: BLAS/LAPACK, so honest translations agree to near machine precision; the
#: margin covers different summation orders.
DEFAULT_RTOL = 1e-9
#: Default absolute tolerance, for values that should be exactly zero.
DEFAULT_ATOL = 1e-12

_NUMERIC_KINDS = frozenset("biufc")


@dataclass(frozen=True)
class Tolerance:
    """Tolerances passed to :func:`numpy.allclose`.

    :param rtol: relative tolerance [-].
    :param atol: absolute tolerance, in the units of the compared values.
    """

    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL


def _is_cell(value: Any) -> bool:
    """Whether a pulled value is an Octave cell array (an object ndarray)."""
    return isinstance(value, np.ndarray) and value.dtype == object


def _squeezed_shape(array: np.ndarray) -> tuple[int, ...]:
    """Shape with every length-1 dimension removed."""
    return tuple(dim for dim in array.shape if dim != 1)


def compare_values(
    *, m_value: Any, py_value: Any, tolerance: Tolerance, index: bool = False
) -> str | None:
    """Compare one ``.m`` result with one Python result.

    :param m_value: value pulled from Octave.
    :param py_value: value from the Python namespace.
    :param tolerance: allclose tolerances.
    :param index: the value holds 1-based indices on the ``.m`` side and 0-based
        indices on the Python side, so ``m_value - 1`` is compared.
    :returns: ``None`` if equal, otherwise a short reason.
    """
    if isinstance(m_value, CallableValue) or isinstance(py_value, CallableValue):
        if isinstance(m_value, CallableValue) and isinstance(py_value, CallableValue):
            return None
        return "callable on one side only"

    if isinstance(m_value, str) or isinstance(py_value, str):
        if isinstance(m_value, str) and isinstance(py_value, str):
            return None if m_value == py_value else f"strings differ: {m_value!r} vs {py_value!r}"
        return f"type differs: {type(m_value).__name__} vs {type(py_value).__name__}"

    if isinstance(m_value, dict) or isinstance(py_value, dict):
        return _compare_mappings(m_value=m_value, py_value=py_value, tolerance=tolerance)

    if _is_cell(m_value):
        return _compare_sequences(m_value=m_value, py_value=py_value, tolerance=tolerance)

    return _compare_arrays(m_value=m_value, py_value=py_value, tolerance=tolerance, index=index)


def compare_shapes(*, m_value: Any, py_value: Any) -> str | None:
    """Compare only what two random results must share: shape and logical-ness.

    ``rand(3)`` and ``np.random.rand(3, 3)`` never agree by value; a
    translation is still wrong if it draws a different number of values or
    returns a different kind of array.

    :param m_value: value pulled from Octave.
    :param py_value: value from the Python namespace.
    :returns: ``None`` if they agree, otherwise a short reason.
    """
    m_array, py_array = np.asarray(m_value), np.asarray(py_value)
    if m_array.dtype.kind not in _NUMERIC_KINDS or py_array.dtype.kind not in _NUMERIC_KINDS:
        return f"cannot compare {m_array.dtype} with {py_array.dtype}"
    if (m_array.dtype.kind == "b") != (py_array.dtype.kind == "b"):
        return f"logical vs numeric: {m_array.dtype} vs {py_array.dtype}"
    if m_array.size == 0 and py_array.size == 0:
        return None
    if _squeezed_shape(m_array) != _squeezed_shape(py_array):
        return f"shape differs: {m_array.shape} vs {py_array.shape}"
    return None


def _compare_mappings(*, m_value: Any, py_value: Any, tolerance: Tolerance) -> str | None:
    """Compare a struct with a dict field by field."""
    if not (isinstance(m_value, dict) and isinstance(py_value, dict)):
        return f"type differs: {type(m_value).__name__} vs {type(py_value).__name__}"
    if set(m_value) != set(py_value):
        return f"fields differ: {sorted(m_value)} vs {sorted(py_value)}"
    for key in sorted(m_value):
        reason = compare_values(m_value=m_value[key], py_value=py_value[key], tolerance=tolerance)
        if reason is not None:
            return f"field {key!r}: {reason}"
    return None


def _compare_sequences(*, m_value: np.ndarray, py_value: Any, tolerance: Tolerance) -> str | None:
    """Compare a cell array with a list/tuple element by element.

    Elements are taken in column-major order, the order Octave stores them.
    """
    if not isinstance(py_value, list | tuple | np.ndarray):
        return f"cell array vs {type(py_value).__name__}"
    m_items = list(m_value.ravel(order="F"))
    if isinstance(py_value, list | tuple):
        py_items = list(py_value)
    else:
        py_items = list(np.asarray(py_value, dtype=object).ravel())
    if len(m_items) != len(py_items):
        return f"length differs: {len(m_items)} vs {len(py_items)}"
    for position, (m_item, py_item) in enumerate(zip(m_items, py_items, strict=True)):
        reason = compare_values(m_value=m_item, py_value=py_item, tolerance=tolerance)
        if reason is not None:
            return f"element {position}: {reason}"
    return None


def _compare_arrays(
    *, m_value: Any, py_value: Any, tolerance: Tolerance, index: bool
) -> str | None:
    """Compare numeric/logical scalars and arrays."""
    m_array = np.asarray(m_value)
    py_array = np.asarray(py_value)
    if m_array.dtype.kind not in _NUMERIC_KINDS or py_array.dtype.kind not in _NUMERIC_KINDS:
        return f"cannot compare {m_array.dtype} with {py_array.dtype}"

    if index:
        if m_array.dtype.kind == "b" or py_array.dtype.kind == "b":
            return "declared as indices but holds logical values"
        m_array = m_array - 1
    elif (m_array.dtype.kind == "b") != (py_array.dtype.kind == "b"):
        return f"logical vs numeric: {m_array.dtype} vs {py_array.dtype}"

    if m_array.size == 0 and py_array.size == 0:
        return None
    if _squeezed_shape(m_array) != _squeezed_shape(py_array):
        return f"shape differs: {m_array.shape} vs {py_array.shape}"

    m_float = m_array.astype(complex if m_array.dtype.kind == "c" else float).reshape(
        _squeezed_shape(m_array)
    )
    py_float = py_array.astype(complex if py_array.dtype.kind == "c" else float).reshape(
        _squeezed_shape(py_array)
    )
    if np.allclose(m_float, py_float, rtol=tolerance.rtol, atol=tolerance.atol, equal_nan=True):
        return None
    with np.errstate(invalid="ignore"):
        difference = np.abs(m_float - py_float)
    finite = difference[np.isfinite(difference)]
    if finite.size and float(finite.max()) > 0:
        return f"values differ (max abs difference {float(finite.max()):.3g})"
    return "values differ (NaN or Inf in different positions)"
