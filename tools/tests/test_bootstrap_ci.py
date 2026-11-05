# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The interval helpers must be conservative where it matters.

The thesis verdicts hang on whether an interval excludes a threshold, so these
tests pin the two properties that decide it: the interval contains the point
estimate, and the exact interval is never narrower than Wilson.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pytest
from bootstrap_ci import (
    binom_test_less,
    bootstrap_items,
    clopper_pearson,
    edit_distance,
    macro_f1,
    percentile_ci,
    wilson,
)


def test_wilson_contains_the_point_estimate() -> None:
    for k, n in [(0, 10), (1, 10), (5, 10), (9, 10), (10, 10), (129, 152)]:
        low, high = wilson(k, n)
        assert low <= k / n <= high
        assert 0.0 <= low <= high <= 1.0


def test_wilson_matches_a_known_value() -> None:
    # 8/9 is the H1.5 measurement in the thesis.
    low, high = wilson(8, 9)
    assert low == pytest.approx(0.565, abs=1e-3)
    assert high == pytest.approx(0.980, abs=1e-3)


def test_exact_interval_is_not_narrower_than_wilson() -> None:
    for k, n in [(129, 152), (52, 99), (173, 178)]:
        w_low, w_high = wilson(k, n)
        c_low, c_high = clopper_pearson(k, n)
        assert c_low <= w_low
        assert c_high >= w_high


def test_exact_interval_brackets_the_point_estimate() -> None:
    for k, n in [(0, 20), (3, 20), (10, 20), (20, 20)]:
        low, high = clopper_pearson(k, n)
        assert low <= k / n <= high


def test_one_sided_p_values_agree_with_the_interval() -> None:
    # 129/152 sits just below 0.90: the lower-tail test rejects, the upper does not.
    assert binom_test_less(129, 152, 0.90) < 0.05
    assert binom_test_less(8, 9, 0.90) > 0.05


def test_macro_f1_of_a_perfect_matrix_is_one() -> None:
    matrix = np.diag([10, 20, 30])
    assert macro_f1(matrix, [0, 1, 2]) == pytest.approx(1.0)


def test_macro_f1_ignores_classes_outside_the_evaluated_set() -> None:
    matrix = np.array([[5, 5], [0, 10]])
    # Class 1 alone: precision 10/15, recall 1.
    assert macro_f1(matrix, [1]) == pytest.approx(0.8)
    # Class 0 adds precision 1.0 and recall 0.5, so F1 = 2/3.
    assert macro_f1(matrix, [0, 1]) == pytest.approx((2 / 3 + 0.8) / 2)


def test_edit_distance_handles_substitutions_and_indels() -> None:
    assert edit_distance("1234ABC", "1234ABC") == 0
    assert edit_distance("1234ABC", "1234ABD") == 1
    assert edit_distance("1234ABC", "123ABC") == 1
    assert edit_distance("", "ABC") == 3


def test_percentile_ci_brackets_the_sample() -> None:
    samples = np.linspace(0.0, 1.0, 1000)
    low, high = percentile_ci(samples, alpha=0.05)
    assert low < high
    assert low == pytest.approx(0.025, abs=0.01)
    assert high == pytest.approx(0.975, abs=0.01)


def test_macro_by_cluster_weights_plates_not_crops() -> None:
    rows = [
        {"truth": "AAA111", "pred": "AAA111", "plate": "AAA111"},
        {"truth": "AAA111", "pred": "AAA111", "plate": "AAA111"},
        {"truth": "AAA111", "pred": "XXX999", "plate": "AAA111"},
        {"truth": "BBB222", "pred": "BBB222", "plate": "BBB222"},
    ]
    args = SimpleNamespace(
        task="text", cluster_col="plate", cluster_regex=None, key_col="stem",
        min_support=1, replicates=20, seed=1, macro_by_cluster=True,
    )
    result = bootstrap_items(rows, args)
    # Per crop the mean would be 0.75; per plate it is ((2/3) + 1) / 2.
    assert result["primary"] == pytest.approx((2 / 3 + 1) / 2)
    assert result["n_clusters"] == 2
