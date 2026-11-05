# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Matching an exit to its entry.

This decides whether a parking session gets closed, so a wrong match corrupts the
occupancy count -- the one number the operator trusts. The policy is therefore
strict: match on an **exact identifier** or not at all.

Measuring first ruled out the cheap option. Using the type classifier's backbone
as an embedding, on 400 crops of the frozen dataset:

    same vehicle      average similarity 0.860
    different vehicles  p50 0.786, p90 0.973, p99 0.991
    P(two different vehicles above 0.95) = 19.9%

One in five pairs of unrelated vehicles looks more alike than the average pair of
the same vehicle, so no threshold separates them. The backbone is trained for
silhouette, not identity; it cannot decide who a vehicle is.

That leaves the plate, which is what the field is named after in the schema. The
matching below is ready for it and refuses to guess until it exists: an exit with
no plate, or a plate whose confidence is too low, leaves the session open. The
operator sees the lot fill up rather than see it silently emptied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

# Below this, a plate is not evidence of identity; it is a guess.
MIN_PLATE_CONFIDENCE = 0.6

_NON_ALNUM = re.compile(r"[^A-Z0-9]")
_PLATE_TEXT = re.compile(r"^[A-Z0-9]{4,9}$")


def normalize_plate(text: str | None) -> str:
    """Comparable form of a plate: uppercase, letters and digits only.

    OCR output arrives with separators and case noise ("1234-abc", "1234 abc"),
    and the same plate should not read as two different ones because of spacing.
    """
    if not text:
        return ""
    return _NON_ALNUM.sub("", text.upper())


def looks_like_plate(text: str | None) -> bool:
    """A short alphanumeric string that mixes digits and letters.

    Brand lettering like "TOYOTA" is all letters and must not become a plate.
    This is a domain rule: both OCR adapters and the matching policy use it.
    """
    cleaned = normalize_plate(text)
    return bool(_PLATE_TEXT.match(cleaned)) and any(ch.isdigit() for ch in cleaned)


@dataclass(frozen=True)
class ExitVehicle:
    """The vehicle leaving: only its identifiers matter here."""

    license_plate: str | None = None
    plate_confidence: float | None = None


@dataclass(frozen=True)
class OpenSession:
    """An active session, with what is known about the vehicle inside."""

    session_id: str
    vehicle_id: str
    entry_time: datetime
    license_plate: str | None = None
    plate_confidence: float | None = None


def find_match(
    exit_vehicle: ExitVehicle,
    open_sessions: list[OpenSession],
    min_confidence: float = MIN_PLATE_CONFIDENCE,
) -> OpenSession | None:
    """The session this exit closes, or None when nothing proves it.

    Returns None rather than the best guess: an unmatched exit is visible, a
    wrong match is invisible and destroys the count.

    When several sessions carry the same plate -- the same vehicle entered twice
    without an exit in between -- the most recent entry wins, which is the one
    this exit most likely closes.
    """
    plate = normalize_plate(exit_vehicle.license_plate)
    if not plate:
        return None
    if (exit_vehicle.plate_confidence or 0.0) < min_confidence:
        return None

    matches = [
        session for session in open_sessions
        if normalize_plate(session.license_plate) == plate
        and (session.plate_confidence or 0.0) >= min_confidence
    ]
    if not matches:
        return None
    return max(matches, key=lambda session: session.entry_time)
