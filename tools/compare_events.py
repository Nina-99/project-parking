# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""How accurately the system decides entry and exit, against a human count (H1.5).

    uv run python tools/compare_events.py \
        --truth ~/Smart_Parking_dataset/truth/ch16_2026033010.csv \
        --system ~/Smart_Parking_dataset/truth/ch16_2026033010_system.json \
        --replay-start 2026-11-02T21:00:00 --tolerance 3

Where the system export comes from: ``GET /api/v1/stats/history`` for the
replayed window, saved as JSON. Each session carries ``entry_time`` and, once
closed, ``exit_time``; those two instants are the detected crossings, because
the system creates one session per entry and closes it on the exit.

The wall clock has to be shifted onto the video timeline: the replay timestamps
events with the machine clock, so ``--replay-start`` is the instant the replay
began and ``--video-start`` the second of video playing at that instant.

What is reported, and why more than one number:

  exactitud   TP / (TP + FP + FN). The criterion of H1.5, and the one that
              matches what an operator feels: a doubled event costs as much as
              a missed one.
  recall      TP / truth, with a Wilson interval. This is the only proportion
              that is genuinely binomial over the ground truth, so it is the
              only one that gets an interval from the labels alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from event_truth import (  # noqa: E402
    DIRECTIONS,
    Comparison,
    Crossing,
    compare,
    read_truth,
    shift_onto_video,
    wilson,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True,
                        help="CSV written by label_events_gui.py.")
    parser.add_argument("--system", type=Path, required=True,
                        help="Crossings the system recorded: either detections from "
                             "export_system_events.py (camera_id, observed_at) or "
                             "sessions from GET /api/v1/stats/history.")
    parser.add_argument("--camera-gate", action="append", default=[],
                        help="NAME=entry|exit for detection rows. Defaults to "
                             "gate-1=entry, gate-2=exit.")
    parser.add_argument("--replay-start", type=datetime.fromisoformat, required=True,
                        help="Instant the replay began, e.g. 2026-11-02T21:00:00.")
    parser.add_argument("--video-start", type=float, default=0.0,
                        help="Second of video playing at --replay-start (default: 0).")
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="Seconds a detection may drift from the truth and still "
                             "count (default: 3). The capture samples every 0.5 s.")
    parser.add_argument("--json", type=Path, default=None)
    return parser


def camera_gates(values: list[str]) -> dict[str, str]:
    """Which camera records entries and which records exits."""
    gates = {"gate-1": "entry", "gate-2": "exit"}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--camera-gate expects NAME=entry|exit, got {value!r}")
        name, gate = value.split("=", 1)
        if gate.strip() not in DIRECTIONS:
            raise SystemExit(f"gate must be entry or exit, got {gate!r}")
        gates[name.strip()] = gate.strip()
    return gates


def parse_moment(value: str) -> datetime:
    """ISO from the API or the database, tolerant of a trailing Z."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _rows_to_crossings(rows: list[dict], gates: dict[str, str]
                       ) -> list[tuple[datetime, str]]:
    """(instant, direction) per recorded crossing, in either input format.

    Detection rows are the honest source: they include exits that found no
    session to close. Session rows are kept because the API already serves them.
    """
    moments: list[tuple[datetime, str]] = []
    for row in rows:
        if "observed_at" in row:
            # The system's own decision wins; the camera's role is only a
            # fallback for rows recorded before the direction was stored.
            gate = row.get("direction") or gates.get(row["camera_id"])
            if gate is None or gate not in DIRECTIONS:
                continue
            moments.append((parse_moment(row["observed_at"]), gate))
            continue
        if row.get("entry_time"):
            moments.append((parse_moment(row["entry_time"]), "entry"))
        if row.get("exit_time"):
            moments.append((parse_moment(row["exit_time"]), "exit"))
    return moments


def detected_crossings(sessions: list[dict], replay_start: datetime,
                       video_start: float, gates: dict[str, str] | None = None
                       ) -> list[Crossing]:
    """Recorded crossings -> crossings on the video timeline."""
    moments = _rows_to_crossings(sessions, gates or camera_gates([]))
    shifted = shift_onto_video([moment for moment, _ in moments],
                               replay_start, video_start)
    return [Crossing(seconds, direction)
            for seconds, (_, direction) in zip(shifted, moments, strict=True)]


def report(title: str, comparison: Comparison, truth: list[Crossing]) -> dict:
    total = len(truth)
    low, high = wilson(len(comparison.matches), total)
    return {
        "subset": title,
        "truth": total,
        "detected": len(comparison.matches) + len(comparison.spurious),
        "matched": len(comparison.matches),
        "missed": len(comparison.missed),
        "spurious": len(comparison.spurious),
        "exactness": round(comparison.exactness, 4),
        "recall": round(comparison.recall, 4),
        "recall_ci": [round(low, 4), round(high, 4)],
        "precision": round(comparison.precision, 4),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if not args.truth.exists():
        raise SystemExit(
            f"no truth file at {args.truth}\n"
            f"Label the footage first:\n"
            f"    uv run python tools/label_events_gui.py --output {args.truth} "
            f"--video <recording>"
        )
    if not args.system.exists():
        raise SystemExit(
            f"no system export at {args.system}\n"
            f"Dump it after the replay:\n"
            f"    uv run python tools/export_system_events.py "
            f"--output {args.system} --since <replay start>"
        )

    truth = read_truth(args.truth)
    sessions = json.loads(args.system.read_text(encoding="utf-8"))
    if isinstance(sessions, dict):
        sessions = sessions.get("sessions", [])
    detected = detected_crossings(sessions, args.replay_start, args.video_start,
                                  camera_gates(args.camera_gate))

    rows = []
    for direction in DIRECTIONS:
        subset_truth = [c for c in truth if c.direction == direction]
        subset_detected = [c for c in detected if c.direction == direction]
        rows.append(report(direction, compare(subset_truth, subset_detected,
                                              args.tolerance), subset_truth))
    rows.append(report("both", compare(truth, detected, args.tolerance), truth))

    print()
    print("| Conjunto | Verdad | Detectado | TP | Perdidos | Espurios | "
          "Exactitud | Recall | IC 95 % | Precisión |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['subset']} | {row['truth']} | {row['detected']} | "
              f"{row['matched']} | {row['missed']} | {row['spurious']} | "
              f"{row['exactness']:.3f} | {row['recall']:.3f} | "
              f"[{row['recall_ci'][0]:.3f}, {row['recall_ci'][1]:.3f}] | "
              f"{row['precision']:.3f} |")
    print()

    for row in rows[:2]:
        subset_truth = [c for c in truth if c.direction == row["subset"]]
        subset_detected = [c for c in detected if c.direction == row["subset"]]
        comparison = compare(subset_truth, subset_detected, args.tolerance)
        for crossing in comparison.missed[:5]:
            print(f"[missed]  {row['subset']} at {crossing.video_seconds:.2f}s")
        for crossing in comparison.spurious[:5]:
            print(f"[spurious] {row['subset']} at {crossing.video_seconds:.2f}s")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[compare] wrote {args.json}")


if __name__ == "__main__":
    main()
