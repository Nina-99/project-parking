# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Colour vocabulary shared by the search and the statistics endpoints.

Two decisions live here so both endpoints cannot drift apart.

**Which colours are reported.** Only six classes carry enough evidence to publish
a per-colour count: blanco, plata, rojo, gris, negro and azul cover 95% of the
observed fleet. The rest (verde, naranja, beige, marron, amarillo) are real
labels but each has a handful of samples, so their counts are aggregated into a
single "minority" bucket for display only -- never as a training class. Grouping
them as a *label* was rejected: their hues are 30-55 degrees apart, so one class
would be multi-modal.

**Which labels to trust.** ``MIN_REPORT_CONFIDENCE`` is the floor below which a
predicted colour is noise whichever colour it names.

The neighbourhood map is measurement, not intuition: measured on the test split
``plata`` and ``gris`` steal about 12% of each other's predictions, and ``gris``
and ``negro`` about 10-12%. Only pairs with at least 5% mutual confusion are
neighbours.
"""

from __future__ import annotations

REPORTABLE_COLORS: tuple[str, ...] = (
    "blanco",
    "plata",
    "rojo",
    "gris",
    "negro",
    "azul",
)

MINORITY_BUCKET = "minority"

MIN_REPORT_CONFIDENCE = 0.3

COLOR_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "blanco": ("plata",),
    "plata": ("gris", "blanco"),
    "gris": ("plata", "negro"),
    "negro": ("gris",),
}


def expand_color(color: str | None) -> tuple[str, ...]:
    """Requested colour plus its measured confusion neighbourhood, exact first."""
    if not color:
        return ()
    return (color, *COLOR_NEIGHBOURS.get(color, ()))


def report_bucket(color: str | None) -> str:
    """Colour name for a published count, or the aggregate bucket."""
    if color in REPORTABLE_COLORS:
        return color
    return MINORITY_BUCKET


def colour_counts(counter: dict[str, int]) -> dict[str, int]:
    """Fixed key order: the reportable colours, then the minority aggregate."""
    counts = {name: counter.get(name, 0) for name in REPORTABLE_COLORS}
    counts[MINORITY_BUCKET] = counter.get(MINORITY_BUCKET, 0)
    return counts
