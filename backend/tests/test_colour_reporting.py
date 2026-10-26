# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Colour reporting rules.

What gets its own number and what gets aggregated is a published claim, so the
rule is pinned here rather than left to the endpoint.
"""

from __future__ import annotations

from backend.domain.colours import (
    MINORITY_BUCKET,
    REPORTABLE_COLORS,
    colour_counts,
    report_bucket,
)


def test_a_supported_colour_is_reported_under_its_own_name() -> None:
    assert report_bucket("plata") == "plata"


def test_a_rare_colour_falls_into_the_aggregate() -> None:
    assert report_bucket("amarillo") == MINORITY_BUCKET
    assert report_bucket("marron") == MINORITY_BUCKET


def test_a_missing_colour_is_not_counted_as_a_colour() -> None:
    assert report_bucket(None) == MINORITY_BUCKET


def test_every_reportable_colour_is_present_even_at_zero() -> None:
    counts = colour_counts({"blanco": 3})

    assert list(counts) == [*REPORTABLE_COLORS, MINORITY_BUCKET]
    assert counts["blanco"] == 3
    assert counts["plata"] == 0
    assert counts[MINORITY_BUCKET] == 0
