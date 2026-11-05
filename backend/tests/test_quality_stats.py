# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Decision-quality aggregates, against a real database.

The rates are the argument the thesis needs about the running system, so the test
checks them from rows rather than trusting the query to say what it looks like it
says.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from backend.app.api.auth import issue_token
from backend.app.core.config import settings
from backend.app.models.detection_event import DetectionEvent
from backend.app.models.validation_decision import ValidationDecision
from backend.app.models.vehicle import Vehicle


async def test_quality_rates_come_from_reviewed_events(api) -> None:
    client, factory = api
    now = datetime.now(UTC)

    async with factory() as session:
        await session.execute(text(
            "TRUNCATE validation_decisions, detection_events, "
            "parking_sessions, vehicles RESTART IDENTITY CASCADE"
        ))
        readable = Vehicle(license_plate="ABC123", created_at=now)
        unreadable = Vehicle(license_plate=None, created_at=now)
        corrected = Vehicle(license_plate="ABD123", created_at=now)
        old = Vehicle(
            license_plate="OLD999",
            created_at=now - timedelta(days=30),
        )
        session.add_all([readable, unreadable, corrected, old])
        await session.flush()

        session.add(DetectionEvent(
            vehicle_id=readable.id, camera_id="gate-2", model_name="yolo",
            object_class="vehicle", confidence=0.9, bbox={}, direction="exit",
            session_id=None, frame_timestamp=now,
        ))
        session.add(DetectionEvent(
            vehicle_id=unreadable.id, camera_id="gate-2", model_name="yolo",
            object_class="vehicle", confidence=0.9, bbox={}, direction="exit",
            session_id=None, frame_timestamp=now,
        ))

        session.add_all([
            ValidationDecision(
                vehicle_id=readable.id, decision="confirmed", operator="ana",
                original_plate="ABC123", created_at=now,
            ),
            ValidationDecision(
                vehicle_id=unreadable.id, decision="confirmed", operator="ana",
                original_plate=None, created_at=now,
            ),
            ValidationDecision(
                vehicle_id=corrected.id, decision="plate_corrected", operator="ana",
                original_plate="ABD123", corrected_plate="ABC123", created_at=now,
            ),
            ValidationDecision(
                vehicle_id=corrected.id, decision="dismissed", operator="beto",
                original_plate="ABC123", created_at=now,
            ),
        ])
        await session.commit()

    client.cookies.set(settings.AUTH_COOKIE_NAME, issue_token("duenio", "owner"))
    response = await client.get("/api/v1/stats/quality?days=7")

    assert response.status_code == 200
    body = response.json()

    assert body["events_total"] == 3, "the 30-day-old vehicle is outside the window"
    assert body["events_with_plate"] == 2
    assert body["plate_read_rate"] == 0.667
    assert body["unmatched_exits"] == 2
    assert body["decisions_total"] == 4
    assert body["by_decision"] == {
        "confirmed": 2,
        "plate_corrected": 1,
        "dismissed": 1,
        "session_closed": 0,
    }
    assert body["correction_rate"] == 0.25
    assert body["operators"] == [
        {"operator": "ana", "decisions": 3, "corrected": 1},
        {"operator": "beto", "decisions": 1, "corrected": 0},
    ]


async def test_an_empty_window_has_no_rates(api) -> None:
    client, factory = api

    async with factory() as session:
        await session.execute(text(
            "TRUNCATE validation_decisions, detection_events, "
            "parking_sessions, vehicles RESTART IDENTITY CASCADE"
        ))
        await session.commit()

    client.cookies.set(settings.AUTH_COOKIE_NAME, issue_token("duenio", "owner"))
    body = (await client.get("/api/v1/stats/quality")).json()

    assert body["events_total"] == 0
    assert body["plate_read_rate"] is None, "no events means no rate, not zero"
    assert body["correction_rate"] is None
    assert body["operators"] == []
