# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""One operator decision about one gate event.

The machine proposes and the person disposes. An entry the operator confirms is
still the machine's claim; a plate the operator corrects is a different fact,
and the audit row keeps both. That distinction is why this module exists instead
of an "update vehicle" endpoint: the correction is not a field edit, it is a
decision with a name and a time attached.

No event is published. A decision changes the queue, not the parking count; the
session close that follows an unmatched exit has its own use case and its own
announcement.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.domain.errors import InvalidValidationDecisionError
from backend.domain.matching import looks_like_plate, normalize_plate
from backend.domain.ports import Clock, UnitOfWork
from backend.domain.validation import DECISIONS


@dataclass(frozen=True)
class ValidationOutcome:
    vehicle_id: str
    decision: str
    original_plate: str | None
    corrected_plate: str | None
    operator: str


def record_validation(
    *,
    vehicle_id: str,
    decision: str,
    operator: str,
    uow: UnitOfWork,
    clock: Clock,
    corrected_plate: str | None = None,
    session_id: str | None = None,
    notes: str | None = None,
) -> ValidationOutcome:
    """Store the operator's decision, correcting the plate when asked.

    ``session_closed`` is for an unmatched exit the operator resolved by closing
    the session it belonged to. The close itself happened in ``close_session``;
    this row is the trace that the event was reviewed.
    """
    if decision not in DECISIONS:
        raise InvalidValidationDecisionError(
            f"decision must be one of {list(DECISIONS)}"
        )

    corrected: str | None = None
    if decision == "plate_corrected":
        cleaned = normalize_plate(corrected_plate)
        if not looks_like_plate(cleaned):
            raise InvalidValidationDecisionError(
                "corrected_plate must be a plate-like value"
            )
        original = uow.vehicles.correct_plate(vehicle_id, cleaned)
        corrected = cleaned
    else:
        # Still touches the vehicle: a decision about a vehicle that does not
        # exist must fail before it writes an orphan audit row.
        original = uow.vehicles.plate_for(vehicle_id)

    uow.validations.add(
        vehicle_id=vehicle_id,
        session_id=session_id,
        decision=decision,
        original_plate=original,
        corrected_plate=corrected,
        operator=operator,
        notes=notes,
        at=clock.now(),
    )
    uow.commit()

    return ValidationOutcome(
        vehicle_id=vehicle_id,
        decision=decision,
        original_plate=original,
        corrected_plate=corrected,
        operator=operator,
    )
