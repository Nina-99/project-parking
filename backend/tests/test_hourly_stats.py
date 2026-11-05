# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Hourly distribution, against a real database.

RF-12 asks for the temporal distribution of entries. The test fixes two hours and
checks that the other twenty-two come back as zero, because the chart depends on a
complete axis.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import text

from backend.app.api.auth import issue_token
from backend.app.core.config import settings
from backend.app.models.parking_session import ParkingSession
from backend.app.models.vehicle import Vehicle

LOCAL = ZoneInfo("America/La_Paz")


async def test_hourly_returns_a_full_day_axis(api) -> None:
    client, factory = api
    local_today = datetime.now(LOCAL).replace(minute=0, second=0, microsecond=0)
    morning = local_today.replace(hour=9).astimezone(UTC)
    evening = local_today.replace(hour=9, minute=30).astimezone(UTC)
    late = local_today.replace(hour=18).astimezone(UTC)

    async with factory() as session:
        await session.execute(text(
            "TRUNCATE validation_decisions, detection_events, "
            "parking_sessions, vehicles RESTART IDENTITY CASCADE"
        ))
        first = Vehicle(license_plate="ABC123")
        second = Vehicle(license_plate="DEF456")
        session.add_all([first, second])
        await session.flush()
        session.add_all([
            ParkingSession(vehicle_id=first.id, entry_gate="gate-1", entry_time=morning),
            ParkingSession(vehicle_id=second.id, entry_gate="gate-1", entry_time=evening),
            ParkingSession(vehicle_id=first.id, entry_gate="gate-1", entry_time=late),
        ])
        await session.commit()

    client.cookies.set(settings.AUTH_COOKIE_NAME, issue_token("duenio", "owner"))
    response = await client.get("/api/v1/stats/hourly?days=7")

    assert response.status_code == 200
    body = response.json()
    assert body["window_days"] == 7
    assert body["timezone"] == "America/La_Paz"
    hours = {item["hour"]: item["vehicles"] for item in body["hours"]}
    assert len(hours) == 24, "the chart needs every hour, including the empty ones"
    assert hours[9] == 2, "09:00 in La Paz is 13:00 UTC and still hour 9"
    assert hours[18] == 1
    assert hours[3] == 0


async def test_hourly_is_readable_by_an_operator(api) -> None:
    client, factory = api

    async with factory() as session:
        await session.execute(text(
            "TRUNCATE validation_decisions, detection_events, "
            "parking_sessions, vehicles RESTART IDENTITY CASCADE"
        ))
        await session.commit()

    client.cookies.set(settings.AUTH_COOKIE_NAME, issue_token("oper", "operator"))
    response = await client.get("/api/v1/stats/hourly")

    assert response.status_code == 200
    assert sum(item["vehicles"] for item in response.json()["hours"]) == 0


async def test_a_late_entry_counts_on_its_local_day(api) -> None:
    """21:00 in La Paz is 01:00 UTC of the next day; the report must not move it."""
    client, factory = api
    local_day = (datetime.now(LOCAL) - timedelta(days=1)).replace(
        hour=21, minute=0, second=0, microsecond=0,
    )
    entry_utc = local_day.astimezone(UTC)

    async with factory() as session:
        await session.execute(text(
            "TRUNCATE validation_decisions, detection_events, "
            "parking_sessions, vehicles RESTART IDENTITY CASCADE"
        ))
        vehicle = Vehicle(license_plate="ABC123")
        session.add(vehicle)
        await session.flush()
        session.add(ParkingSession(
            vehicle_id=vehicle.id, entry_gate="gate-1", entry_time=entry_utc,
        ))
        await session.commit()

    client.cookies.set(settings.AUTH_COOKIE_NAME, issue_token("duenio", "owner"))
    response = await client.get("/api/v1/stats/daily?days=7")

    assert response.status_code == 200
    rows = {row["date"]: row["total_vehicles"] for row in response.json()}
    assert rows[local_day.date().isoformat()] == 1
    if entry_utc.date() != local_day.date():
        assert entry_utc.date().isoformat() not in rows, (
            "the UTC day must not receive a local evening entry"
        )
