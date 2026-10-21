"""Two-pass extraction pipeline.

Pass 1 scans the video sparsely and keeps only the frame windows where
something moves. Pass 2 re-decodes those windows densely and runs the tracker,
which needs frame-to-frame continuity. Entry/pass is decided per track, and
PaddleOCR pre-labels the plate only for tracks classified as entries.

Every classified track also queues one vehicle crop for human body-type and
colour labelling. Those two attributes are never pseudo-labelled: an automatic
labeller would teach the classifier the rule it is later compared against.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from plate_dataset.config import CameraConfig, GlobalConfig, infer_kwargs
from plate_dataset.geometry import ZoneModel, clamp_box
from plate_dataset.motion import MotionGate, find_active_segments
from plate_dataset.plate_ocr import PlateReader
from plate_dataset.tracking import TrackState
from plate_dataset.writer import DatasetWriter

Window = tuple[int, int]


def vehicle_pre_score(vehicle_crop: np.ndarray, frame_area: int | None = None) -> float:
    """Cheap plate-legibility proxy used to keep the top-K frames per track.

    The plate sits in the lower part of a vehicle, so sharpness is measured
    there instead of over the whole crop, scaled by the vehicle size. A crop
    that fills the frame is penalised: the vehicle is so close that its plate is
    usually out of view (or clipped), which is the opposite of what we want.
    """
    h, w = vehicle_crop.shape[:2]
    if h < 10 or w < 10:
        return 0.0
    lower = vehicle_crop[int(h * 0.45):h, :]
    if lower.size == 0:
        lower = vehicle_crop
    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    size = float(w * h)
    if frame_area:
        fraction = size / float(frame_area)
        if fraction > 0.55:
            size *= max(0.05, 1.0 - (fraction - 0.55) / 0.45)
    return sharpness * (size / 10000.0)


class Extractor:
    def __init__(self, glob: GlobalConfig, model: YOLO, reader: PlateReader,
                 writer: DatasetWriter) -> None:
        self.glob = glob
        self.model = model
        self.reader = reader
        self.writer = writer

    # -- Pass 1 ---------------------------------------------------------------
    def _find_windows(self, cap: cv2.VideoCapture, glob: GlobalConfig,
                      max_frames: int | None = None) -> list[Window]:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames) if total else max_frames
        gate = MotionGate(glob.motion_downscale, glob.motion_pixel_threshold)
        energies: list[float] = []
        idx = 0

        while True:
            if total and idx >= total:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            energies.append(gate.energy(frame))
            idx += glob.scan_stride

        return find_active_segments(
            energies,
            stride=glob.scan_stride,
            threshold=glob.motion_threshold,
            pad_frames=glob.segment_pad_frames,
            merge_gap=glob.segment_merge_gap,
        )

    # -- Pass 2 ----------------------------------------------------------------
    def _reset_tracker(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        if not trackers:
            return
        for tracker in trackers:
            try:
                tracker.reset()
            except Exception:
                pass

    def _process_window(self, cap: cv2.VideoCapture, camera: CameraConfig,
                        zones: ZoneModel, window: Window, video: str) -> None:
        start, end = window
        self._reset_tracker()
        tracks: dict[int, TrackState] = {}
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for frame_idx in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            if self.glob.dense_stride > 1 and (frame_idx % self.glob.dense_stride) != 0:
                continue
            if frame_idx % 2000 == 0:
                pct = 100.0 * (frame_idx - start) / max(1, end - start)
                print(f"    {pct:5.1f}%  frame {frame_idx}/{end}  tracks {len(tracks)}",
                      flush=True)

            results = self.model.track(
                frame, persist=True, verbose=False, device=self.glob.device,
                conf=camera.vehicle_conf, classes=list(camera.vehicle_classes),
                tracker=self.glob.tracker, **infer_kwargs(self.glob),
            )

            boxes = results[0].boxes if results else None
            if boxes is not None and boxes.id is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                cls = boxes.cls
                classes = cls.cpu().numpy().astype(int) if cls is not None else None
                for i, track_id in enumerate(ids):
                    track_id = int(track_id)
                    x1, y1, x2, y2 = clamp_box(xyxy[i], frame.shape[1], frame.shape[0])
                    if (x2 - x1) * (y2 - y1) < camera.min_area_pixels:
                        continue
                    vehicle_crop = frame[y1:y2, x1:x2]
                    frame_h, frame_w = frame.shape[:2]
                    score = vehicle_pre_score(vehicle_crop, frame_h * frame_w)
                    origin = (x1, y1, frame_w, frame_h)
                    state = tracks.get(track_id)
                    if state is None:
                        state = TrackState(track_id, frame_idx, frame_idx, frame_idx)
                        tracks[track_id] = state
                    state.update(frame_idx, np.array([x1, y1, x2, y2], dtype=float),
                                 zones, self.glob.top_k_frames, vehicle_crop, score,
                                 origin,
                                 None if classes is None else int(classes[i]))

            self._age_tracks(tracks, frame_idx, zones, video, camera)

        for track_id in list(tracks):
            self._finalize(tracks, track_id, zones, video, camera)

    def _age_tracks(self, tracks: dict[int, TrackState], frame_idx: int,
                    zones: ZoneModel, video: str, camera: CameraConfig) -> None:
        for track_id, state in list(tracks.items()):
            if state.last_seen == frame_idx:
                continue
            state.frames_since_seen += 1
            if state.frames_since_seen > self.glob.grace_frames:
                self._finalize(tracks, track_id, zones, video, camera)

    # -- Emission --------------------------------------------------------------
    def _finalize(self, tracks: dict[int, TrackState], track_id: int, zones: ZoneModel,
                  video: str, camera: CameraConfig) -> None:
        state = tracks.pop(track_id, None)
        if state is None:
            return
        label = state.classify(zones, self.glob.min_track_hits)

        if label in ("entry", "exit"):
            self._emit_gate_event(state, video, camera, label)
        elif label == "pass" and state.best_frames:
            best = state.best_frames[0]
            self.writer.write_negative(video, camera.name, state.track_id,
                                       best.frame_idx, best.crop, best.score)

        if label != "unknown":
            self._emit_attribute_sample(state, video, camera, label)

    def _emit_attribute_sample(self, state: TrackState, video: str,
                               camera: CameraConfig, kind: str) -> None:
        """Queue the widest view of the track for type and colour labelling.

        Emitted for every classified track, entries and passers alike. The
        attribute vocabulary is independent of the plate dataset, so a vehicle
        that simply drove past is still a valid body-type and colour sample.
        """
        candidate = state.largest_candidate()
        if candidate is None:
            return
        self.writer.write_attribute_sample(
            video, camera.name, state.track_id, candidate.frame_idx,
            candidate.crop, candidate.score, kind,
            vehicle_class=candidate.vehicle_class,
            min_px=self.glob.attribute_min_px,
        )

    def _osd_boxes_in_crop(self, candidate, camera: CameraConfig) -> list[tuple[int, int, int, int]]:
        """Map the frame-fixed OSD regions into the candidate crop's pixels."""
        x0, y0, frame_w, frame_h = candidate.origin
        boxes = []
        for nx1, ny1, nx2, ny2 in camera.osd_zones:
            boxes.append((
                int(nx1 * frame_w) - x0,
                int(ny1 * frame_h) - y0,
                int(nx2 * frame_w) - x0,
                int(ny2 * frame_h) - y0,
            ))
        return boxes

    def _emit_gate_event(self, state: TrackState, video: str, camera: CameraConfig,
                         kind: str) -> None:
        best: tuple[tuple[float, float], object] | None = None
        for candidate in state.best_frames:
            plate = self.reader.read(candidate.crop,
                                     exclude_boxes=self._osd_boxes_in_crop(candidate, camera))
            if plate is None:
                continue
            key = (plate.confidence, plate.sharpness)
            if best is None or key > best[0]:
                best = (key, (plate, candidate))

        if best is None:
            # No readable plate: do not lose the event, park it for review.
            if state.best_frames:
                fallback = state.best_frames[0]
                self.writer.write_unlabeled_entry(video, camera.name, state.track_id,
                                                  fallback.frame_idx, fallback.crop,
                                                  fallback.score, kind=kind)
                print(f"    [{kind}] track {state.track_id} frame {fallback.frame_idx} "
                      f"— sin placa (review)", flush=True)
            return

        plate, candidate = best[1]
        review = (plate.confidence < self.glob.ocr_min_conf
                  or plate.sharpness < self.glob.min_plate_sharpness)
        self.writer.write_entry(video, camera.name, state.track_id, candidate.frame_idx,
                                candidate.crop, plate, candidate.score, review, kind=kind)
        tag = " review" if review else ""
        print(f"    [{kind}] track {state.track_id} frame {candidate.frame_idx} "
              f"placa={plate.text} conf={plate.confidence:.2f}{tag}", flush=True)

    # -- Driver ----------------------------------------------------------------
    def run_video(self, video_path: Path, camera: CameraConfig,
                  max_frames: int | None = None) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[{video_path.name}] ERROR: cannot open video")
            return

        if not (camera.street_zone and camera.inside_zone and camera.tripwire):
            print(f"[{video_path.name}] ERROR: camera '{camera.name}' has incomplete zones")
            cap.release()
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        zones = ZoneModel(camera.street_zone, camera.inside_zone, camera.tripwire,
                          width, height)

        windows = self._find_windows(cap, self.glob, max_frames)
        print(f"[{video_path.name}] camera={camera.name} — {len(windows)} activity window(s)",
              flush=True)

        for wi, (start, end) in enumerate(windows, 1):
            if max_frames:
                end = min(end, max_frames)
            if end <= start:
                continue
            print(f"  window {wi}/{len(windows)}: frames {start}-{end}", flush=True)
            self._process_window(cap, camera, zones, (start, end), video_path.stem)

        cap.release()
