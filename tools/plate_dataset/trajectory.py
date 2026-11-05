# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Trajectory viewer used to calibrate the zone polygons.

Runs the tracker over a slice of a video *without* any zone filtering and
draws every track as a polyline over a reference frame. Entering and passing
vehicles show up as two visually distinct families of curves, which is what
lets you place ``street_zone`` / ``inside_zone`` / ``tripwire`` on evidence
instead of guessing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from plate_dataset.config import CameraConfig, GlobalConfig, infer_kwargs
from plate_dataset.geometry import ZoneModel
from plate_dataset.tracking import TrackState

LABEL_COLORS = {
    "entry": (60, 220, 60),      # green
    "exit": (255, 150, 0),       # blue
    "pass": (60, 60, 235),       # red
    "unknown": (170, 170, 170),  # grey
}


def _reset_tracker(model: YOLO) -> None:
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None) if predictor is not None else None
    if not trackers:
        return
    for tracker in trackers:
        try:
            tracker.reset()
        except Exception:
            pass


def collect_tracks(
    model: YOLO,
    video_path: Path,
    zones: ZoneModel,
    camera: CameraConfig,
    glob: GlobalConfig,
    start_frame: int,
    max_frames: int,
) -> tuple[np.ndarray | None, dict[int, TrackState]]:
    """Track a slice of the video and return a reference frame plus the states."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(start_frame + max_frames, total) if max_frames else total
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    _reset_tracker(model)
    states: dict[int, TrackState] = {}
    base: np.ndarray | None = None
    empty = np.empty(0, dtype=np.uint8)

    for frame_idx in range(start_frame, end):
        ok, frame = cap.read()
        if not ok:
            break
        if base is None:
            base = frame.copy()

        results = model.track(
            frame, persist=True, verbose=False, device=glob.device,
            conf=camera.vehicle_conf, classes=list(camera.vehicle_classes),
            tracker=glob.tracker, **infer_kwargs(glob),
        )
        boxes = results[0].boxes if results else None
        if boxes is None or boxes.id is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        for i, track_id in enumerate(ids):
            state = states.get(track_id)
            if state is None:
                state = TrackState(track_id, frame_idx, frame_idx, frame_idx)
                states[track_id] = state
            state.update(frame_idx, xyxy[i], zones, 0, empty, 0.0)

    cap.release()
    return base, states


def _draw_grid(canvas: np.ndarray, width: int, height: int) -> None:
    for step in range(1, 10):
        t = step / 10.0
        x, y = int(t * width), int(t * height)
        cv2.line(canvas, (x, 0), (x, height), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, y), (width, y), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{t:.1f}", (x + 3, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{t:.1f}", (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1, cv2.LINE_AA)


def render(
    base: np.ndarray,
    zones: ZoneModel,
    states: dict[int, TrackState],
    min_hits: int,
    draw_grid: bool = True,
) -> np.ndarray:
    canvas = base.copy()

    overlay = canvas.copy()
    cv2.fillPoly(overlay, [zones.street], (235, 140, 40))
    cv2.fillPoly(overlay, [zones.inside], (40, 150, 240))
    canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)

    cv2.polylines(canvas, [zones.street], True, (235, 140, 40), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [zones.inside], True, (40, 150, 240), 2, cv2.LINE_AA)
    cv2.line(canvas, tuple(zones.tripwire[0]), tuple(zones.tripwire[1]),
             (0, 255, 255), 2, cv2.LINE_AA)

    counts = {"entry": 0, "exit": 0, "pass": 0, "unknown": 0}
    for state in states.values():
        if len(state.points) < 2:
            continue
        label = state.classify(zones, min_hits)
        counts[label] += 1
        pts = np.array([[int(x), int(y)] for _, x, y, _ in state.points])
        color = LABEL_COLORS[label]
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[0]), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[-1]), 6, color, -1, cv2.LINE_AA)

    if draw_grid:
        _draw_grid(canvas, zones.width, zones.height)

    lines = [
        f"tracks: {len(states)}",
        f"entry: {counts['entry']}  exit: {counts['exit']}  pass: {counts['pass']}  unknown: {counts['unknown']}",
        "yellow = tripwire",
    ]
    for i, text in enumerate(lines):
        y = 30 + i * 26
        cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

    return canvas
