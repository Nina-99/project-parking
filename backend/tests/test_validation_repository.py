# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The queue query and the whitelist check against a real database.

The unit tests use recorders, which prove the route maps what it receives but not
that the SQL runs. The `regexp_replace` comparison and the window-function
subquery are exactly the kind of thing that only fails on PostgreSQL.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text

from backend.adapters.outbound.db.repositories import (
    SqlAlchemyAuthorizationStore,
    SqlAlchemyVehicleStore,
)
from backend.app.api.auth import issue_token
from backend.app.core.config import settings
from backend.app.models.authorized_vehicle import AuthorizedVehicle
from backend.app.models.detection_event import DetectionEvent
from backend.app.models.vehicle import Vehicle


def test_the_whitelist_check_ignores_separators(db) -> None:
    with db() as session:
        session.add(AuthorizedVehicle(license_plate="1234-ABC", is_active=True))
        session.add(AuthorizedVehicle(license_plate="9999-ZZZ", is_active=False))
        session.commit()

        store = SqlAlchemyAuthorizationStore(session)

        assert store.is_authorized("1234 ABC") is True
        assert store.is_authorized("1234abc") is True
        assert store.is_authorized("9999ZZZ") is False, "inactive is not authorized"
        assert store.is_authorized(None) is None, "no plate is unknown, not denied"
        assert store.is_authorized("") is None


def test_a_corrected_plate_becomes_usable_for_matching(db) -> None:
    with db() as session:
        vehicle = Vehicle(license_plate="ABD123", license_plate_confidence=0.55)
        session.add(vehicle)
        session.commit()

        store = SqlAlchemyVehicleStore(session)
        previous = store.correct_plate(str(vehicle.id), "abc-123")

        assert previous == "ABD123"
        session.refresh(vehicle)
        assert vehicle.license_plate == "ABC123"
        assert vehicle.license_plate_confidence == 1.0, (
            "a human-verified plate must pass the matching confidence floor"
        )


async def test_pending_returns_one_row_per_unreviewed_event(api) -> None:
    client, factory = api
    now = datetime.now(UTC)

    async with factory() as session:
        await session.execute(text(
            "TRUNCATE validation_decisions, detection_events, "
            "parking_sessions, vehicles, authorized_vehicles "
            "RESTART IDENTITY CASCADE"
        ))
        known = Vehicle(
            license_plate="ABC123",
            vehicle_type="car",
            plate_characters=[
                {"char": "A", "confidence": 0.9,
                 "alternatives": [{"char": "4", "confidence": 0.05}]},
            ],
        )
        unknown = Vehicle(license_plate=None, vehicle_type="suv")
        session.add_all([known, unknown])
        await session.flush()
        session.add(AuthorizedVehicle(license_plate="ABC-123", is_active=True))
        session.add(DetectionEvent(
            vehicle_id=known.id, camera_id="gate-1", model_name="yolo",
            object_class="vehicle", confidence=0.9, bbox={}, direction="entry",
            frame_timestamp=now,
        ))
        session.add(DetectionEvent(
            vehicle_id=unknown.id, camera_id="gate-2", model_name="yolo",
            object_class="vehicle", confidence=0.9, bbox={}, direction="exit",
            frame_timestamp=now,
        ))
        await session.commit()
        known_id = str(known.id)

    client.cookies.set(settings.AUTH_COOKIE_NAME, issue_token("ana", "operator"))
    response = await client.get("/api/v1/validations/pending")

    assert response.status_code == 200
    items = response.json()
    assert len(items) == 2

    by_id = {item["vehicle_id"]: item for item in items}
    assert by_id[known_id]["direction"] == "entry"
    assert by_id[known_id]["authorized"] is True
    assert by_id[known_id]["image_url"] is None
    assert by_id[known_id]["plate_characters"][0]["char"] == "A"

    unknown_item = next(item for item in items if item["vehicle_id"] != known_id)
    assert unknown_item["direction"] == "exit"
    assert unknown_item["authorized"] is None
