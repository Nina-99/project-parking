# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Dump the crossings the system recorded, for the H1.5 comparison.

    uv run python tools/export_system_events.py --output system_2026041511.json

Reads the detection log, not the sessions. A session only exists when an entry
opened it and an exit proved itself with a plate match; an exit that found no
session is still a recorded crossing, and H1.5 must count it or the measurement
would grade the matcher instead of the event decision.

Each row is one confirmed vehicle at one camera:

    {"camera_id": "gate-1", "observed_at": "2026-11-03T03:28:26+00:00"}

The direction is not stored: it comes from which camera recorded the crossing,
and ``compare_events.py`` maps it when comparing.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, text

from backend.app.core.config import settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON file with one row per recorded crossing.")
    parser.add_argument("--since", type=datetime.fromisoformat, default=None,
                        help="Only events at or after this instant.")
    parser.add_argument("--until", type=datetime.fromisoformat, default=None,
                        help="Only events before this instant.")
    parser.add_argument("--camera", action="append", default=[],
                        help="Restrict to these camera ids. Repeatable.")
    parser.add_argument("--database-url", default=settings.DATABASE_URL_SYNC)
    return parser


def _as_utc(value: datetime | None) -> datetime | None:
    """``date -u +%Y-%m-%dT%H:%M:%S`` has no offset; the column does."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.since = _as_utc(args.since)
    args.until = _as_utc(args.until)

    # The OR must be parenthesised: `a or b and c` binds as `a or (b and c)`, and
    # every entry has a session, so a window filter would let them all through.
    query = ("select camera_id, frame_timestamp, direction from detection_events "
             "where (session_id is not null or camera_id like 'gate-%')")
    conditions, params = [], {}
    if args.since:
        conditions.append("frame_timestamp >= :since")
        params["since"] = args.since
    if args.until:
        conditions.append("frame_timestamp < :until")
        params["until"] = args.until
    if args.camera:
        conditions.append("camera_id = any(:cameras)")
        params["cameras"] = args.camera
    if conditions:
        query += " and " + " and ".join(conditions)
    query += " order by frame_timestamp"

    engine = create_engine(args.database_url)
    with engine.connect() as connection:
        rows = connection.execute(text(query), params).mappings().all()

    payload = [
        {"camera_id": row["camera_id"],
         "observed_at": row["frame_timestamp"].isoformat(),
         "direction": row["direction"]}
        for row in rows
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cameras: dict[str, int] = {}
    for row in payload:
        cameras[row["camera_id"]] = cameras.get(row["camera_id"], 0) + 1
    print(f"[events] {len(payload)} crossings -> {args.output}")
    for camera, count in sorted(cameras.items()):
        print(f"  {camera}: {count}")
    if args.since and args.until:
        window = (args.until - args.since).total_seconds()
        print(f"  window {window:.0f}s, {len(payload) / (window / 3600):.1f} per hour")


if __name__ == "__main__":
    main()
