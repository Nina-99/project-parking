# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Vehicle values shared by the gate policies.

A classifier result without its confidence is not usable: the reporting code
refuses low-confidence colours and the matching code refuses low-confidence
plates. Keeping the number next to the label stops either from being dropped on
the way to the database.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleAttributes:
    """What the classifiers said about one vehicle, and how sure they were."""

    vehicle_type: str | None = None
    vehicle_type_confidence: float | None = None
    color: str | None = None
    color_confidence: float | None = None


@dataclass(frozen=True)
class PlateChar:
    """One emitted character, with how sure the model was and its runners-up.

    The alternatives are the model's own second and third choices at that
    timestep. They are not a correction: they are what the operator sees when the
    read is plausible but not certain, so a single wrong character can be fixed
    without retyping the whole plate.
    """

    char: str
    confidence: float
    alternatives: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Plate:
    """A read plate. ``confidence`` is the OCR score, not a guess."""

    text: str
    confidence: float
    characters: tuple[PlateChar, ...] = ()
