# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Value comparison rules (no Octave needed: values are built by hand)."""

import numpy as np
import pytest

from dualect.compare import Tolerance, compare_values
from dualect.runners import CallableValue

TOL = Tolerance()


def same(m_value: object, py_value: object, *, index: bool = False) -> bool:
    return compare_values(m_value=m_value, py_value=py_value, tolerance=TOL, index=index) is None


def test_row_vector_equals_1d_array() -> None:
    assert same(np.array([[1.0, 2.0, 3.0]]), np.array([1, 2, 3]))


def test_column_vector_equals_1d_array() -> None:
    assert same(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))


def test_matrix_shape_must_match() -> None:
    assert not same(np.ones((2, 3)), np.ones((3, 2)))


def test_values_within_tolerance() -> None:
    assert same(1.0, 1.0 + 1e-13)
    assert not same(1.0, 1.001)


def test_nan_equals_nan_but_not_number() -> None:
    assert same(np.array([np.nan, 1.0]), np.array([np.nan, 1.0]))
    reason = compare_values(m_value=np.nan, py_value=1.0, tolerance=TOL)
    assert reason is not None and "NaN" in reason


def test_empty_arrays_equal_regardless_of_shape() -> None:
    assert same([], np.array([]))
    assert same(np.zeros((1, 0)), np.zeros((0,)))


def test_logical_vs_numeric_is_a_difference() -> None:
    reason = compare_values(
        m_value=np.array([True, False]), py_value=np.array([1, 0]), tolerance=TOL
    )
    assert reason is not None and "logical" in reason
    assert same(np.array([True, False]), np.array([True, False]))


def test_int_vs_float_is_not_a_difference() -> None:
    assert same(3.0, 3)


def test_index_values_are_shifted() -> None:
    assert same(np.array([[1.0, 4.0]]), np.array([0, 3]), index=True)
    assert not same(np.array([[1.0, 4.0]]), np.array([1, 4]), index=True)


def test_strings() -> None:
    assert same("abc", "abc")
    assert not same("abc", "abd")
    assert not same("1", 1.0)


def test_struct_and_dict() -> None:
    assert same({"a": 1.0, "b": "x"}, {"a": 1, "b": "x"})
    assert not same({"a": 1.0}, {"a": 1.0, "b": 2.0})


def test_cell_and_list() -> None:
    cell = np.empty((1, 2), dtype=object)
    cell[0, 0], cell[0, 1] = 1.0, "a"
    assert same(cell, [1, "a"])
    assert not same(cell, [1])


def test_callables_only_agree_on_being_callable() -> None:
    assert same(CallableValue(description="@(x) x"), CallableValue(description="lambda"))
    assert not same(CallableValue(description="@(x) x"), 1.0)


@pytest.mark.parametrize("value", [1 + 2j, np.array([1 + 1j, 2])])
def test_complex(value: object) -> None:
    assert same(value, value)
