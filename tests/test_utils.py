import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from familarity.utils import (
    clipped_cosine_similarity,
    combine_counters,
    cumsum_until,
    iterate_dict_in_batches,
    make_output_path,
)


def test_make_output_path(tmp_path):
    output_path = make_output_path(base_path=tmp_path)

    assert isinstance(output_path, Path)
    assert os.path.exists(output_path)
    assert os.path.isdir(output_path)


@pytest.mark.parametrize(
    "values, threshold, expected",
    [
        ([10, 20, 30], 25, [10, 15]),  # Threshold reached within two elements
        ([5, 10, 15, 20], 30, [5, 10, 15]),  # Accumulated sum stops before reaching the last element
        ([5, 10, 15], 10, [5, 5]),  # Threshold reached exactly with two elements
        ([50, 20], 40, [40]),  # Single element truncated to meet the threshold
        ([5, 5, 5], 50, [5, 5, 5]),  # Threshold not reached, all values returned
    ],
)
def test_cumsum_until(values, threshold, expected):
    assert cumsum_until(values, threshold) == expected


@pytest.mark.parametrize(
    "vec1, vec2, expected",
    [
        (np.array([1, 0]), np.array([1, 0]), 1.0),  # Identical vectors
        (np.array([1, 0]), np.array([-1, 0]), 0.0),  # Opposite vectors, should clip to 0
        (np.array([1, 1]), np.array([1, 1]), 1.0),  # Same direction
        (np.array([1, 0]), np.array([0, 1]), 0.0),  # Orthogonal vectors
        (np.array([0, 0]), np.array([1, 1]), 0.0),  # vec1 is zero vector, expect 0
        (np.array([1, 1]), np.array([0, 0]), 0.0),  # vec2 is zero vector, expect 0
        (np.array([0, 0]), np.array([0, 0]), 0.0),  # Both are zero vectors
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 0.974631),  # General positive case
        (np.array([1, -1]), np.array([-1, 1]), 0.0),  # Negative similarity, should clip to 0
    ],
)
def test_clipped_cosine_similarity(vec1, vec2, expected):
    result = clipped_cosine_similarity(vec1, vec2)
    assert result == pytest.approx(expected, 0.00001)


@pytest.mark.parametrize(
    "train_counter, test_counter, expected",
    [
        # Basic test case with some overlapping words
        (
            Counter({"apple": 3, "banana": 2}),
            Counter({"banana": 4, "cherry": 1}),
            {"apple": (3, 0), "banana": (2, 4), "cherry": (0, 1)},
        ),
        # Case where train_counter has unique words
        (
            Counter({"apple": 3}),
            Counter({"banana": 4}),
            {"apple": (3, 0), "banana": (0, 4)},
        ),
        # Case where both counters are empty
        (
            Counter(),
            Counter(),
            {},
        ),
        # Case where test_counter is empty
        (
            Counter({"apple": 3, "banana": 2}),
            Counter(),
            {"apple": (3, 0), "banana": (2, 0)},
        ),
        # Case where train_counter is empty
        (
            Counter(),
            Counter({"banana": 4, "cherry": 1}),
            {"banana": (0, 4), "cherry": (0, 1)},
        ),
        # Case with more complex counters
        (
            Counter({"apple": 2, "banana": 1, "cherry": 5}),
            Counter({"banana": 3, "cherry": 2, "date": 4}),
            {"apple": (2, 0), "banana": (1, 3), "cherry": (5, 2), "date": (0, 4)},
        ),
    ],
)
def test_combine_counters(train_counter, test_counter, expected):
    result = combine_counters(train_counter, test_counter)
    assert result == expected


@pytest.mark.parametrize(
    "d, batch_size, expected_batches",
    [
        # Basic case: dictionary with 5 items and batch size of 2
        (
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            2,
            [{"a": 1, "b": 2}, {"c": 3, "d": 4}, {"e": 5}],
        ),
        # Dictionary with fewer items than batch size
        (
            {"a": 1, "b": 2},
            5,
            [{"a": 1, "b": 2}],
        ),
        # Dictionary with exact multiples of batch size
        (
            {"a": 1, "b": 2, "c": 3, "d": 4},
            2,
            [{"a": 1, "b": 2}, {"c": 3, "d": 4}],
        ),
        # Dictionary with batch size of 1 (each item in its own batch)
        (
            {"a": 1, "b": 2, "c": 3},
            1,
            [{"a": 1}, {"b": 2}, {"c": 3}],
        ),
        # Empty dictionary should yield no batches
        (
            {},
            2,
            [],
        ),
    ],
)
def test_iterate_dict_in_batches(d: Dict[Any, Any], batch_size: int, expected_batches: list[Dict[Any, Any]]):
    result_batches = list(iterate_dict_in_batches(d, batch_size))
    assert result_batches == expected_batches
