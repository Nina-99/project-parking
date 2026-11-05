# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Operator decisions: the audit row, the plate correction and its limits."""
from __future__ import annotations

import pytest

from backend.application.decide_validation import record_validation
from backend.domain.errors import InvalidValidationDecisionError, VehicleNotFoundError
from backend.tests.fakes import FakeUnitOfWork, FixedClock


def uow_with(plate: str | None = "ABC123") -> FakeUnitOfWork:
    uow = FakeUnitOfWork()
    if plate is not None:
        uow.vehicles.plates["vehicle-1"] = plate
    return uow


def test_confirming_records_the_decision_with_the_operator() -> None:
    uow = uow_with()

    outcome = record_validation(
        vehicle_id="vehicle-1", decision="confirmed", operator="ana",
        uow=uow, clock=FixedClock(),
    )

    assert outcome.decision == "confirmed"
    assert uow.validations.added == [{
        "vehicle_id": "vehicle-1",
        "session_id": None,
        "decision": "confirmed",
        "original_plate": "ABC123",
        "corrected_plate": None,
        "operator": "ana",
        "notes": None,
        "at": FixedClock().now(),
    }]
    assert uow.log == ["commit"], "the audit row is durable before the response"


def test_correcting_a_plate_keeps_both_values() -> None:
    uow = uow_with("ABD123")

    record_validation(
        vehicle_id="vehicle-1", decision="plate_corrected", operator="ana",
        corrected_plate="ABC-123", uow=uow, clock=FixedClock(),
    )

    assert uow.vehicles.corrected == [("vehicle-1", "ABC123")]
    row = uow.validations.added[0]
    assert row["original_plate"] == "ABD123"
    assert row["corrected_plate"] == "ABC123"


def test_a_correction_that_is_not_a_plate_is_rejected() -> None:
    uow = uow_with()

    with pytest.raises(InvalidValidationDecisionError):
        record_validation(
            vehicle_id="vehicle-1", decision="plate_corrected", operator="ana",
            corrected_plate="TOYOTA", uow=uow, clock=FixedClock(),
        )

    assert uow.validations.added == []


def test_an_unknown_decision_is_rejected() -> None:
    with pytest.raises(InvalidValidationDecisionError):
        record_validation(
            vehicle_id="vehicle-1", decision="approved", operator="ana",
            uow=uow_with(), clock=FixedClock(),
        )


def test_a_decision_about_a_missing_vehicle_fails_before_auditing() -> None:
    uow = FakeUnitOfWork()
    uow.vehicles.missing.add("ghost")

    with pytest.raises(VehicleNotFoundError):
        record_validation(
            vehicle_id="ghost", decision="dismissed", operator="ana",
            uow=uow, clock=FixedClock(),
        )

    assert uow.validations.added == []


def test_a_resolved_unmatched_exit_names_the_session_it_closed() -> None:
    uow = uow_with(None)

    record_validation(
        vehicle_id="vehicle-1", decision="session_closed", operator="ana",
        session_id="session-9", uow=uow, clock=FixedClock(),
    )

    assert uow.validations.added[0]["session_id"] == "session-9"
