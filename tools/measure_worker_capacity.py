# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""How many gates one inference worker can serve at the configured sample rate.

    uv run python tools/measure_worker_capacity.py --image puerta_1_1.png

The worker runs with ``--concurrency=1`` because track state lives in the
process, so this number is the ceiling on new access points: past it the queue
grows faster than the worker drains it. Measured per frame, in the three stages
that every frame pays for:

  decode     -- the JPEG the capture process puts on the queue
  analyze    -- YOLO plus the two attribute classifiers
  aggregate  -- TrackAggregator.update, pure Python

Persistence is excluded on purpose: it is paid once per confirmed vehicle, not
once per frame. This measures the same block the system records as
``inference_latency_ms``, so the output is comparable with production telemetry.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.domain.events import GateDetection, TrackAggregator  # noqa: E402
from backend.domain.vehicle import VehicleAttributes  # noqa: E402
from backend.worker.engine import DetectionEngine  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", type=Path, default=None,
                        help="Frame to measure. Repeatable; defaults to puerta_1_1.png.")
    parser.add_argument("--frames", type=int, default=60,
                        help="Timed repetitions per image (default: 60).")
    parser.add_argument("--warmup", type=int, default=8,
                        help="Discarded repetitions before timing (default: 8).")
    parser.add_argument("--sample-interval", type=float, default=0.5,
                        help="Capture sampling interval in seconds (default: 0.5, "
                             "which is 2 frames per second per camera).")
    parser.add_argument("--utilisation", type=float, default=0.7,
                        help="Fraction of the worker that may be busy before the "
                             "queue becomes the bottleneck (default: 0.7).")
    parser.add_argument("--yolo", type=Path, default=None,
                        help="Override the detector weights, e.g. yolo26s.pt.")
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write the report as JSON here.")
    return parser


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "mean_ms": round(statistics.mean(samples), 2),
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(p95, 2),
        "max_ms": round(max(samples), 2),
    }


def _timed(fn, times: int) -> tuple[list[float], list]:
    """Per-call wall time, synchronised so GPU work is not timed as a launch."""
    samples, results = [], []
    for _ in range(times):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        results.append(fn())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples, results


def _engine(yolo: Path | None) -> DetectionEngine:
    if yolo is None:
        return DetectionEngine.get_instance()
    from backend.app.core.config import settings

    return DetectionEngine(
        yolo_path=yolo,
        type_checkpoint=Path(settings.TYPE_MODEL_PATH),
        color_checkpoint=Path(settings.COLOR_MODEL_PATH),
        confidence=settings.YOLO_CONFIDENCE,
    )


def _aggregate_cost(detections, args: argparse.Namespace) -> list[float]:
    """One aggregator, four frames: what tracking costs per frame."""
    attributes = VehicleAttributes(
        vehicle_type=None, vehicle_type_confidence=None,
        color=None, color_confidence=None,
    )
    boxes = [d.box for d in detections] or [(0, 0, 1, 1)]

    def run() -> int:
        aggregator = TrackAggregator()
        now = datetime.now(UTC)
        produced = 0
        for step in range(4):
            produced += len(aggregator.update(
                "bench",
                [GateDetection(box=box, attributes=attributes) for box in boxes],
                now + timedelta(seconds=args.sample_interval * step),
            ))
        return produced

    samples, _ = _timed(run, args.frames)
    return [value / 4.0 for value in samples]


def measure_image(engine: DetectionEngine, image: Path, args: argparse.Namespace) -> dict:
    frame = cv2.imread(str(image))
    if frame is None:
        raise SystemExit(f"cannot read {image}")
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise SystemExit(f"cannot encode {image}")
    payload = encoded.tobytes()

    for _ in range(args.warmup):
        engine.process(frame)

    decode, _ = _timed(
        lambda: cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR),
        args.frames,
    )
    analyze, per_frame = _timed(lambda: engine.process(frame), args.frames)
    aggregate = _aggregate_cost(per_frame[-1], args)

    per_frame_ms = statistics.mean(decode) + statistics.mean(analyze) + statistics.mean(aggregate)
    fps = 1000.0 / per_frame_ms
    return {
        "image": image.name,
        "frame_px": [frame.shape[1], frame.shape[0]],
        "jpeg_kib": round(len(payload) / 1024, 1),
        "vehicles_per_frame": round(statistics.mean(len(d) for d in per_frame), 2),
        "decode": _stats(decode),
        "analyze": _stats(analyze),
        "aggregate": _stats(aggregate),
        "per_frame_ms": round(per_frame_ms, 2),
        "frames_per_second": round(fps, 1),
        "cameras_at_full_load": round(fps * args.sample_interval, 1),
        "cameras_at_utilisation": round(fps * args.utilisation * args.sample_interval, 1),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    images = args.image or [Path("puerta_1_1.png")]
    engine = _engine(args.yolo)

    print(f"device={engine.device}")

    reports = []
    for image in images:
        report = measure_image(engine, image, args)
        reports.append(report)
        print(
            f"{report['image']}: per_frame={report['per_frame_ms']}ms "
            f"fps={report['frames_per_second']} "
            f"cameras@{args.sample_interval}s full={report['cameras_at_full_load']} "
            f"at_{int(args.utilisation * 100)}%={report['cameras_at_utilisation']} "
            f"vehicles={report['vehicles_per_frame']}"
        )
        print(
            f"  decode {report['decode']['mean_ms']}ms | "
            f"analyze {report['analyze']['mean_ms']}ms "
            f"(p95 {report['analyze']['p95_ms']}ms) | "
            f"aggregate {report['aggregate']['mean_ms']}ms"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"[measure] wrote {args.json}")


if __name__ == "__main__":
    main()
