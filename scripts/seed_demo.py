"""Seed a realistic day of traffic for demos and screenshots.

This is **demo data**, not a simulation of the detector: it writes vehicles,
sessions, detection telemetry and crops directly, and beats both cameras in
Redis. Use it to rehearse the defence or to fill the UI when the cameras are not
connected.

    DATABASE_URL_SYNC=postgresql+psycopg://... uv run python scripts/seed_demo.py --yes
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.adapters.outbound.db.session import sync_session  # noqa: E402
from backend.app.core.config import settings  # noqa: E402
from backend.app.models.detection_event import DetectionEvent  # noqa: E402
from backend.app.models.parking_session import ParkingSession, SessionStatus  # noqa: E402
from backend.app.models.vehicle import Vehicle  # noqa: E402

TYPES = ["auto", "camioneta", "moto", "bus"]
COLORS = ["plata", "azul", "negro", "blanco", "gris", "rojo"]
PLATE_LETTERS = ["ABC", "XYZ", "DEF", "GHI", "JKL", "MNO"]


def make_crops(source: Path, root: Path, count: int = 8) -> list[str]:
    """Prefer real camera frames; fall back to the supplied image."""
    sources = sorted(Path(".").glob("calle_*.png")) + sorted(Path(".").glob("puerta_*.png"))
    if not sources:
        sources = [source]

    paths: list[str] = []
    for index, candidate in enumerate(sources[:count]):
        image = cv2.imread(str(candidate))
        if image is None:
            continue
        height, width = image.shape[:2]
        target_width = 640
        resized = cv2.resize(
            image, (target_width, int(height * target_width / width)),
            interpolation=cv2.INTER_AREA,
        )
        directory = root / ("gate-1" if index % 2 == 0 else "gate-2")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"demo_{index}.jpg"
        cv2.imwrite(str(path), resized)
        paths.append(str(path))
    return paths


def make_vehicle(crops: list[str], entry: datetime, plate: str | None) -> Vehicle:
    vehicle_type = random.choice(TYPES)
    color = random.choice(COLORS)
    return Vehicle(
        license_plate=plate,
        license_plate_confidence=0.92 if plate else None,
        vehicle_type=vehicle_type,
        vehicle_type_confidence=round(random.uniform(0.72, 0.97), 3),
        color=color,
        color_confidence=round(random.uniform(0.68, 0.96), 3),
        entry_image_path=random.choice(crops) if crops else None,
        created_at=entry,
    )


def telemetry(session_id, camera_id: str, entry: datetime, crop: str | None) -> DetectionEvent:
    return DetectionEvent(
        session_id=session_id,
        camera_id=camera_id,
        frame_timestamp=entry,
        model_name="yolo26n",
        object_class="vehicle",
        confidence=round(random.uniform(0.70, 0.98), 3),
        bbox={"x1": 0.12, "y1": 0.31, "x2": 0.58, "y2": 0.92},
        crop_image_path=crop,
        inference_latency_ms=round(random.uniform(38, 118), 1),
        track_id=random.randint(1, 400),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed demo traffic")
    parser.add_argument("--yes", action="store_true", help="confirm writing demo rows")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--source-image", type=Path, default=Path("test_output.jpg"))
    parser.add_argument("--entries", type=Path, default=Path(settings.ENTRY_IMAGE_DIR))
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit("Refusing to write demo data without --yes")

    random.seed(11)
    now = datetime.now(UTC)
    crops = make_crops(args.source_image, args.entries)
    print(f"crops: {len(crops)} in {args.entries}")

    with sync_session() as session:
        for day in range(args.days):
            for _ in range(random.randint(2, 5)):
                entry = now - timedelta(
                    days=day,
                    hours=random.randint(2, 9),
                    minutes=random.randint(0, 59),
                )
                duration = random.randint(600, 7200)
                plate = (
                    f"{random.randint(1000, 9999)}{random.choice(PLATE_LETTERS)}"
                    if random.random() > 0.25 else None
                )
                vehicle = make_vehicle(crops, entry, plate)
                session.add(vehicle)
                session.flush()

                reason = random.choice(["plate_match", "manual"])
                parking = ParkingSession(
                    vehicle_id=vehicle.id,
                    entry_gate="gate-1",
                    exit_gate="gate-2",
                    entry_time=entry,
                    exit_time=entry + timedelta(seconds=duration),
                    duration_seconds=duration,
                    status=SessionStatus.COMPLETED,
                    close_reason=reason,
                    closed_by="operator" if reason == "manual" else None,
                )
                session.add(parking)
                session.flush()
                session.add(telemetry(parking.id, "gate-1", entry, vehicle.entry_image_path))

        for minutes in (8, 55, 13 * 60):
            entry = now - timedelta(minutes=minutes)
            plate = f"{random.randint(1000, 9999)}{random.choice(PLATE_LETTERS)}"
            vehicle = make_vehicle(crops, entry, plate)
            session.add(vehicle)
            session.flush()
            parking = ParkingSession(
                vehicle_id=vehicle.id,
                entry_gate="gate-1",
                entry_time=entry,
                status=SessionStatus.ACTIVE,
            )
            session.add(parking)
            session.flush()
            session.add(telemetry(parking.id, "gate-1", entry, vehicle.entry_image_path))

        # A couple of sessions closed today, so the daily average has a value.
        for minutes, duration in ((95, 1200), (40, 900)):
            entry = now - timedelta(minutes=minutes)
            vehicle = make_vehicle(crops, entry, None)
            session.add(vehicle)
            session.flush()
            parking = ParkingSession(
                vehicle_id=vehicle.id,
                entry_gate="gate-1",
                exit_gate="gate-2",
                entry_time=entry,
                exit_time=entry + timedelta(seconds=duration),
                duration_seconds=duration,
                status=SessionStatus.COMPLETED,
                close_reason="manual",
                closed_by="operator",
            )
            session.add(parking)
            session.flush()
            session.add(telemetry(parking.id, "gate-1", entry, vehicle.entry_image_path))

    client = redis.Redis.from_url(settings.CELERY_BROKER_URL)
    for camera_id, gate in (("gate-1", "entry"), ("gate-2", "exit")):
        key = f"camera:{camera_id}"
        client.hset(key, mapping={
            "camera_id": camera_id,
            "gate": gate,
            "last_seen": now.isoformat(),
            "dropped_frames": "0",
        })
        client.expire(key, 3600)
    client.close()

    print("demo data written; cameras beaten")


if __name__ == "__main__":
    main()
