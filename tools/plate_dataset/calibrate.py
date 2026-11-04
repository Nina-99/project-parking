# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Interactive, real-time zone calibrator.

Plays a video with the tracker running and draws the trajectories live, so the
entry family (vehicles that turn into the premises) separates from the pass
family (street traffic) as you watch. Zones are edited with the mouse and saved
straight back to ``zones_config.json``.

Controls
--------
    SPACE        pause / resume
    b / f        seek back / forward 5 s (also left/right arrow keys)
    , / .        step back / forward one frame
    1 / 2 / 3    edit street polygon / inside polygon / tripwire
    left drag    move a vertex
    a            add a vertex at the cursor
    d            delete the vertex nearest the cursor
    s / ENTER    save zones to the config file
    r            clear accumulated trajectories
    q / ESC      quit
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from plate_dataset.config import CameraConfig, GlobalConfig, infer_kwargs, resolve_weights

TARGETS = ("street", "inside", "tripwire")
COLORS = {"street": (240, 150, 40), "inside": (40, 160, 240), "tripwire": (0, 255, 255)}
TRAJ_COLORS = {
    "entry": (60, 220, 60),
    "exit": (255, 150, 0),
    "pass": (60, 60, 235),
    "unknown": (170, 170, 170),
}
WINDOW = "Zone calibrator"


def _dist_point_segment(p, a, b) -> float:
    p, a, b = np.array(p, float), np.array(a, float), np.array(b, float)
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float((p - a) @ ab) / denom))
    return float(np.linalg.norm(p - (a + t * ab)))


class ZoneCalibrator:
    def __init__(self, video: Path, camera: CameraConfig, glob: GlobalConfig,
                 config_path: Path) -> None:
        self.video = Path(video)
        self.camera = camera
        self.glob = glob
        self.config_path = Path(config_path)
        self.zones: dict[str, list[list[float]]] = {
            "street": [list(p) for p in camera.street_zone] or [[0.2, 0.1], [0.8, 0.1], [0.8, 0.4], [0.2, 0.4]],
            "inside": [list(p) for p in camera.inside_zone] or [[0.2, 0.5], [0.8, 0.5], [0.8, 0.9], [0.2, 0.9]],
            "tripwire": [list(p) for p in camera.tripwire] or [[0.2, 0.45], [0.8, 0.45]],
        }
        self.target = "street"
        self.mouse = (0, 0)
        self.dragging: int | None = None
        self.tracks: dict[int, list[tuple[float, float]]] = {}
        self.paused = False
        self.width = 0
        self.height = 0
        self.total = 0
        self.fps = 15.0
        self.model: YOLO | None = None
        # Monotonic instant until which the "saved" badge stays on screen. Saving
        # does not close the window, and without this the only feedback went to a
        # terminal the calibrator covers.
        self.save_flash = 0.0
        self.save_count = 0

    # -- coordinate helpers ---------------------------------------------------
    def _to_px(self, point) -> tuple[int, int]:
        return int(point[0] * self.width), int(point[1] * self.height)

    def _to_norm(self, x: int, y: int) -> list[float]:
        return [min(1.0, max(0.0, x / self.width)), min(1.0, max(0.0, y / self.height))]

    def _zone_model(self):
        from plate_dataset.geometry import ZoneModel

        return ZoneModel(self.zones["street"], self.zones["inside"],
                         self.zones["tripwire"], self.width, self.height)

    # -- tracking -------------------------------------------------------------
    def _reset_tracker(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        for tracker in trackers or []:
            try:
                tracker.reset()
            except Exception:
                pass

    def _process(self, frame: np.ndarray, model: YOLO) -> None:
        results = model.track(
            frame, persist=True, verbose=False, device=self.glob.device,
            conf=self.camera.vehicle_conf, classes=list(self.camera.vehicle_classes),
            tracker=self.glob.tracker, **infer_kwargs(self.glob),
        )
        boxes = results[0].boxes if results else None
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        for i, track_id in enumerate(ids):
            x1, y1, x2, y2 = xyxy[i]
            point = (float((x1 + x2) / 2.0), float(y2))
            self.tracks.setdefault(int(track_id), []).append(point)

    # -- classification (live feedback, mirrors the extractor's rule) ---------
    def _classify(self, points: list[tuple[float, float]], zm) -> str:
        if len(points) < 3:
            return "unknown"
        state = 0  # 0 unknown, 1 last in street, 2 last inside
        for point in points:
            if zm.in_street(point):
                if state == 2:
                    return "exit"
                state = 1
            elif zm.in_inside(point):
                if state == 1:
                    return "entry"
                state = 2
        return "unknown" if state == 2 else "pass"

    # -- mouse / keyboard -----------------------------------------------------
    def _nearest_vertex(self, x: int, y: int) -> int | None:
        best, best_d = None, 14.0
        for i, point in enumerate(self.zones[self.target]):
            px, py = self._to_px(point)
            d = float(np.hypot(px - x, py - y))
            if d < best_d:
                best, best_d = i, d
        return best

    def _on_mouse(self, event, x, y, flags, param) -> None:  # noqa: ANN001
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = self._nearest_vertex(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging is not None:
            self.zones[self.target][self.dragging] = self._to_norm(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    def _add_vertex(self) -> None:
        x, y = self.mouse
        pts = self.zones[self.target]
        if self.target == "tripwire":
            if len(pts) < 2:
                pts.append(self._to_norm(x, y))
            return
        if len(pts) < 3:
            pts.append(self._to_norm(x, y))
            return
        new = self._to_norm(x, y)
        best = min(
            ((_dist_point_segment(new, pts[i], pts[(i + 1) % len(pts)]), i) for i in range(len(pts))),
            key=lambda item: item[0],
        )
        pts.insert(best[1] + 1, new)

    def _delete_vertex(self) -> None:
        pts = self.zones[self.target]
        index = self._nearest_vertex(*self.mouse)
        if index is None:
            return
        minimum = 2 if self.target == "tripwire" else 3
        if len(pts) > minimum:
            pts.pop(index)

    def _save(self) -> None:
        data = json.loads(self.config_path.read_text(encoding="utf-8")) if self.config_path.exists() else {}
        cam = data.setdefault("cameras", {}).setdefault(self.camera.name, {})
        cam["name"] = self.camera.name
        cam["street_zone"] = [[round(p[0], 4), round(p[1], 4)] for p in self.zones["street"]]
        cam["inside_zone"] = [[round(p[0], 4), round(p[1], 4)] for p in self.zones["inside"]]
        cam["tripwire"] = [[round(p[0], 4), round(p[1], 4)] for p in self.zones["tripwire"]]
        cam.setdefault("vehicle_conf", self.camera.vehicle_conf)
        cam.setdefault("min_area_pixels", self.camera.min_area_pixels)
        cam.setdefault("vehicle_classes", list(self.camera.vehicle_classes))
        if self.camera.osd_zones:
            cam.setdefault("osd_zones", [list(b) for b in self.camera.osd_zones])
        self.config_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
        self.save_flash = time.monotonic() + 1.8
        self.save_count += 1
        print(f"[saved] {self.config_path} -> camera '{self.camera.name}'")

    def _seek(self, cap: cv2.VideoCapture, delta: int, current: np.ndarray) -> np.ndarray:
        """Move the playhead and reset the tracker so IDs stay coherent."""
        position = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + delta
        position = max(0, position)
        if self.total:
            position = min(position, self.total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        self._reset_tracker()
        if self.paused:
            ok, frame = cap.read()
            if ok:
                return frame
        return current

    # -- rendering ------------------------------------------------------------
    def _render(self, frame: np.ndarray) -> np.ndarray:
        canvas = frame.copy()
        zm = self._zone_model()
        overlay = canvas.copy()

        for name in ("street", "inside"):
            pts = np.array([self._to_px(p) for p in self.zones[name]], np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], COLORS[name])
        canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)

        counts = {"entry": 0, "exit": 0, "pass": 0, "unknown": 0}
        for points in self.tracks.values():
            label = self._classify(points, zm)
            counts[label] += 1
            pts = np.array([(int(x), int(y)) for x, y in points], np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, TRAJ_COLORS[label], 2, cv2.LINE_AA)
                cv2.circle(canvas, tuple(pts[-1]), 5, TRAJ_COLORS[label], -1, cv2.LINE_AA)

        for name in TARGETS:
            pts = [self._to_px(p) for p in self.zones[name]]
            if name == "tripwire" and len(pts) == 2:
                cv2.line(canvas, pts[0], pts[1], COLORS[name], 2, cv2.LINE_AA)
            elif len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts, np.int32)], True, COLORS[name], 2, cv2.LINE_AA)
            for i, (px, py) in enumerate(pts):
                active = name == self.target
                cv2.circle(canvas, (px, py), 6 if active else 4,
                           COLORS[name] if active else (150, 150, 150), -1, cv2.LINE_AA)

        help_lines = [
            f"target: {self.target}  [1]street [2]inside [3]tripwire",
            f"tracks {len(self.tracks)}  entry {counts['entry']}  exit {counts['exit']}  pass {counts['pass']}  unknown {counts['unknown']}",
            "[a]dd [d]elete vertex | drag to move | [r]eset | SPACE pause",
            "[s]/ENTER save (window stays open) | [q] quit",
            "[b]/[f] seek 5s | [,]/[.] 1 frame | arrows seek",
            "PAUSED" if self.paused else "playing",
        ]
        for i, text in enumerate(help_lines):
            y = 26 + i * 24
            cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if time.monotonic() < self.save_flash:
            text = f"SAVED  {self.camera.name}  ({self.save_count})"
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x = max(12, canvas.shape[1] - text_w - 36)
            y = 34
            cv2.rectangle(canvas, (x - 12, y - text_h - 10),
                          (x + text_w + 12, y + 10), (40, 140, 40), -1)
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    # -- main loop ------------------------------------------------------------
    def run(self) -> None:
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            print(f"[ERROR] cannot open {self.video}")
            return
        ok, frame = cap.read()
        if not ok:
            print(f"[ERROR] cannot read {self.video}")
            return
        self.height, self.width = frame.shape[:2]
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(self.width, 1280), min(self.height, 720))
        cv2.setMouseCallback(WINDOW, self._on_mouse)

        model = YOLO(resolve_weights(self.glob.model_name))
        self.model = model
        self._reset_tracker()
        print("Calibrating. Adjust zones with the mouse, press [s] to save, [q] to quit.")

        while True:
            if not self.paused:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.tracks.clear()
                    self._reset_tracker()
                    continue
                self._process(frame, model)

            cv2.imshow(WINDOW, self._render(frame))
            raw = cv2.waitKeyEx(1)
            key = raw & 0xFF
            left_arrow = (65361, 63234, 2424832)
            right_arrow = (65363, 63235, 2555904)

            if key in (ord("q"), 27):
                break
            if key == 32:
                self.paused = not self.paused
            elif key == ord("1"):
                self.target = "street"
            elif key == ord("2"):
                self.target = "inside"
            elif key == ord("3"):
                self.target = "tripwire"
            elif key == ord("a"):
                self._add_vertex()
            elif key == ord("d"):
                self._delete_vertex()
            elif key in (ord("s"), 13):
                self._save()
            elif key == ord("r"):
                self.tracks.clear()
            elif key == ord("b") or raw in left_arrow:
                frame = self._seek(cap, -max(1, int(self.fps * 5)), frame)
            elif key == ord("f") or raw in right_arrow:
                frame = self._seek(cap, max(1, int(self.fps * 5)), frame)
            elif key == ord(","):
                frame = self._seek(cap, -1, frame)
            elif key == ord("."):
                frame = self._seek(cap, 1, frame)

        cap.release()
        cv2.destroyAllWindows()
