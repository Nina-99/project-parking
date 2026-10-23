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
class Plate:
    """A read plate. ``confidence`` is the OCR score, not a guess."""

    text: str
    confidence: float
