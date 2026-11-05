# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The per-character read survives the trip to the database and back."""
from __future__ import annotations

import uuid

from backend.adapters.outbound.db.repositories import SqlAlchemyVehicleStore
from backend.app.models.vehicle import Vehicle
from backend.domain.vehicle import Plate, PlateChar, VehicleAttributes


def test_the_per_character_read_is_stored(db) -> None:
    plate = Plate(
        text="A8",
        confidence=0.65,
        characters=(
            PlateChar("A", 0.9, (("4", 0.05),)),
            PlateChar("8", 0.4, (("B", 0.3), ("3", 0.2))),
        ),
    )

    with db() as session:
        vehicle_id = SqlAlchemyVehicleStore(session).add(
            VehicleAttributes(vehicle_type="car"), plate, None,
        )
        session.commit()
        row = session.get(Vehicle, uuid.UUID(vehicle_id))

    assert row is not None
    assert row.plate_characters == [
        {
            "char": "A",
            "confidence": 0.9,
            "alternatives": [{"char": "4", "confidence": 0.05}],
        },
        {
            "char": "8",
            "confidence": 0.4,
            "alternatives": [
                {"char": "B", "confidence": 0.3},
                {"char": "3", "confidence": 0.2},
            ],
        },
    ]


def test_a_plate_without_character_detail_stays_null(db) -> None:
    with db() as session:
        vehicle_id = SqlAlchemyVehicleStore(session).add(
            VehicleAttributes(), Plate(text="ABC123", confidence=0.9), None,
        )
        session.commit()
        row = session.get(Vehicle, uuid.UUID(vehicle_id))

    assert row is not None
    assert row.plate_characters is None, "a reader without detail must not invent it"
