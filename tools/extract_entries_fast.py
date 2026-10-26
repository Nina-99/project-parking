# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Fast entry-focused vehicle extractor (no plate OCR).

    uv run python tools/extract_entries_fast.py \
        --input /mnt/nina/PARKING/ver \
        --output /mnt/nina/SmartParking/entries_fast \
        --model yolo26m.pt --stride 2 --pass-per-class 2 --resume

Why a second extractor: the full one (``extract_dataset.py``) runs PaddleOCR and
keeps top-k frames per track, which is right for plates but far too slow when the
goal is body type and colour. This tool drops OCR and keeps, per track, a single
crop — the whole point is speed over 1000+ hours of video.

What it emits, tagged by ``kind``:

* ``entry`` — one crop per vehicle that crosses the tripwire into the premises
  (gate view, mostly the rear).
* ``exit`` — the same for vehicles leaving (gate view, mostly the front). Both
  gate directions are the view the app actually sees, so both are kept.
* ``pass`` — a *capped* number of street passers, and only for the detector
  classes that do NOT enter (motorcycle / bus / truck). These feed the classes
  that barely appear at the gate: moto, minibus and ``otro`` (bus/camión).
  Passing cars are deliberately skipped — entries already cover cars, and
  hoarding passers would only unbalance the dataset.

Balance is a *dataset-build* concern; extraction only has to cover the gaps.

Image names keep the ``<video>_track<id>_f<frame>`` convention (traceable and
track-groupable for leak-free splits). ``kind`` lives in the folder and the
manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    infer_camera,
    infer_kwargs,
    list_videos,
    load_config,
    resolve_weights,
)
from plate_dataset.geometry import ZoneModel, clamp_box  # noqa: E402
from plate_dataset.pipeline import vehicle_pre_score  # noqa: E402
from plate_dataset.tracking import TrackState  # noqa: E402

# COCO classes that are vehicles, and which ones are the *gap* classes the gate
# rarely sees. Passing cars are not stored on purpose (see the module docstring).
VEHICLE_CLASSES = (2, 3, 5, 7)
GAP_CLASSES = {3, 5, 7}  # motorcycle, bus, truck
COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
KIND_FOLDERS = {"entry": "entries", "exit": "exits", "pass": "passers"}


def choose_representative(candidates) -> object | None:
    """Best crop for body type and colour.

    Ranked by ``vehicle_pre_score`` (sharp, sizable, penalising frame-filling
    crops) — the frame the original pipeline used for gate events, which yields
    clean rear/front views. A crop occupying almost the whole frame is skipped
    when any smaller view exists: that is the useless roof close-up a car shows
    when it hugs the barrier camera.
    """
    if not candidates:
        return None

    def height_fraction(c) -> float:
        frame_h = c.origin[3] if c.origin and c.origin[3] else c.crop.shape[0]
        return c.crop.shape[0] / frame_h if frame_h else 1.0

    usable = [c for c in candidates if height_fraction(c) <= 0.8]
    return max(usable or candidates, key=lambda c: c.score)


class FastExtractor:
    def __init__(self, glob, model: YOLO, output: Path, args) -> None:
        self.glob = glob
        self.model = model
        self.output = output
        self.args = args
        self.manifest = output / "manifest.jsonl"
        self.counts: dict[str, int] = {}

    def _save(self, stem: str, kind: str, crop: np.ndarray, row: dict) -> None:
        folder = self.output / KIND_FOLDERS[kind]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stem}.jpg"
        if crop.size == 0 or not cv2.imwrite(str(path), crop):
            return
        row["image"] = str(path.relative_to(self.output))
        with self.manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def _reset_tracker(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        for tracker in trackers or []:
            try:
                tracker.reset()
            except Exception:
                pass

    def _finalize(self, tracks, track_id, zones, video, camera, pass_counts) -> None:
        state = tracks.pop(track_id, None)
        if state is None:
            return
        label = state.classify(zones, self.glob.min_track_hits)
        candidate = choose_representative(state.best_frames)
        if candidate is None:
            return

        if label in ("entry", "exit"):
            stem = f"{video}_track{state.track_id:04d}_f{candidate.frame_idx:07d}"
            self._save(stem, label, candidate.crop, {
                "stem": stem, "kind": label, "camera": camera.name, "video": video,
                "track_id": state.track_id, "frame": candidate.frame_idx,
                "crop_hw": list(candidate.crop.shape[:2]),
                "vehicle_class": candidate.vehicle_class,
            })
        elif label == "pass" and candidate.vehicle_class in GAP_CLASSES:
            cls = int(candidate.vehicle_class)
            if pass_counts.get(cls, 0) >= self.args.pass_per_class:
                return
            pass_counts[cls] = pass_counts.get(cls, 0) + 1
            stem = f"{video}_track{state.track_id:04d}_f{candidate.frame_idx:07d}"
            self._save(stem, "pass", candidate.crop, {
                "stem": stem, "kind": "pass", "camera": camera.name, "video": video,
                "track_id": state.track_id, "frame": candidate.frame_idx,
                "crop_hw": list(candidate.crop.shape[:2]),
                "vehicle_class": cls,
            })

    def run_video(self, video_path: Path, camera) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[{video_path.name}] ERROR: cannot open")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        zones = ZoneModel(camera.street_zone, camera.inside_zone, camera.tripwire, width, height)

        self._reset_tracker()
        tracks: dict[int, TrackState] = {}
        pass_counts: dict[int, int] = {}
        video = video_path.stem
        prev_small = None
        idle = 0
        idx = -1
        processed = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1

            # Motion gate. Skips only *sustained* idle: a brief dip in the
            # frame difference (a slow moment mid-track) still gets processed,
            # so ByteTrack keeps frame-to-frame continuity and a vehicle is not
            # split into several tracks. Skipping every low-diff frame instead
            # made the entry count swing wildly (6..11).
            if self.args.motion_threshold > 0:
                small = cv2.resize(frame, None, fx=0.25, fy=0.25,
                                   interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                diff = 999.0 if prev_small is None else float(cv2.absdiff(gray, prev_small).mean())
                prev_small = gray
                if diff >= self.args.motion_threshold:
                    idle = 0
                else:
                    idle += 1
                if idle > self.args.idle_frames:
                    continue

            if self.args.stride > 1 and idx % self.args.stride != 0:
                continue

            processed += 1
            results = self.model.track(
                frame, persist=True, verbose=False, device=self.glob.device,
                conf=camera.vehicle_conf, classes=list(VEHICLE_CLASSES),
                tracker=self.glob.tracker, imgsz=self.args.imgsz, **infer_kwargs(self.glob),
            )
            boxes = results[0].boxes if results else None
            if boxes is not None and boxes.id is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                classes = boxes.cls.cpu().numpy().astype(int)
                for i, track_id in enumerate(ids):
                    x1, y1, x2, y2 = clamp_box(xyxy[i], width, height)
                    if (x2 - x1) * (y2 - y1) < camera.min_area_pixels:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    score = vehicle_pre_score(crop, width * height)
                    state = tracks.get(int(track_id))
                    if state is None:
                        state = TrackState(int(track_id), idx, idx, idx, hits=0)
                        tracks[int(track_id)] = state
                    state.update(idx, np.array([x1, y1, x2, y2], dtype=float), zones,
                                 self.glob.top_k_frames, crop, score,
                                 (x1, y1, width, height), int(classes[i]))

            for tid, st in list(tracks.items()):
                if st.last_seen != idx:
                    st.frames_since_seen += 1
                    if st.frames_since_seen > self.glob.grace_frames:
                        self._finalize(tracks, tid, zones, video, camera, pass_counts)

        for tid in list(tracks):
            self._finalize(tracks, tid, zones, video, camera, pass_counts)
        cap.release()
        print(f"  [{video_path.name}] frames analizados {processed} | "
              f"entries acumulados {self.counts.get('entry', 0)} | "
              f"passers {self.counts.get('pass', 0)}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast entry-focused vehicle extractor.")
    parser.add_argument("--input", type=Path, help="Directory with videos.")
    parser.add_argument("--video", type=Path, help="Single video file.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", default="yolo26m.pt", help="YOLO detection weights.")
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference size (lower = faster, less detail).")
    parser.add_argument("--motion-threshold", type=float, default=0.0,
                        help="Skip frames with mean abs-diff below this (0 = off).")
    parser.add_argument("--idle-frames", type=int, default=15,
                        help="Consecutive low-motion frames before the gate sleeps.")
    parser.add_argument("--pass-per-class", type=int, default=2,
                        help="Max street passers per gap class (moto/bus/truck) per video.")
    parser.add_argument("--limit", type=int, help="Process at most N videos.")
    parser.add_argument("--index", type=int, help="1-based index of one video to process.")
    parser.add_argument("--resume", action="store_true", help="Skip videos already processed.")
    parser.add_argument("--device", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--camera", help="Force a camera key from the config.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cameras, glob = load_config(args.config)
    if args.device is not None:
        glob.device = args.device

    if args.video:
        videos = [args.video]
    else:
        if args.input is None:
            print("[ERROR] provide --input <dir> or --video <file>")
            return
        videos = list_videos(args.input)
    if args.index is not None:
        videos = [videos[args.index - 1]]
    if args.limit:
        videos = videos[:args.limit]

    weights = resolve_weights(args.model)
    print(f"Videos: {len(videos)} | output: {args.output} | model: {weights} | "
          f"stride: {args.stride} | pass/class: {args.pass_per_class}")
    model = YOLO(weights)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / ".processed.jsonl"
    done: set[str] = set()
    if args.resume and state_path.exists():
        done = {ln.strip() for ln in state_path.read_text(encoding="utf-8").splitlines() if ln.strip()}
        print(f"Resume: {len(done)} video(s) already done")

    extractor = FastExtractor(glob, model, output, args)
    for i, video in enumerate(videos, 1):
        if args.resume and video.stem in done:
            continue
        camera = cameras[args.camera] if args.camera else infer_camera(video.name, cameras)
        print(f"\n[{i}/{len(videos)}] {video.name}", flush=True)
        extractor.run_video(video, camera)
        with state_path.open("a", encoding="utf-8") as fh:
            fh.write(video.stem + "\n")

    print(f"\nDone. Totales: {extractor.counts}")


if __name__ == "__main__":
    main()
