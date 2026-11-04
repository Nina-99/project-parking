# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Export full frames, one per vehicle pass, for measuring detection (H1.1).

    uv run python tools/export_detection_frames.py \
        --videos /mnt/nina/PARKING --hour 2026033010 \
        --output /home/nina/Smart_Parking_dataset/detection

Why one frame per pass and not one frame per N seconds: a vehicle at a gate is
in view for several samples, and consecutive frames of the same car are almost
identical. Sampling time would give hundreds of near-duplicates of a few cars
and an interval that pretends to be narrower than it is. The independent unit is
the pass, so the pass is what gets exported.

Why the frame is saved at the third sighting: that is the moment the tracker
confirms the vehicle, and deferring the save to pick the largest view would need
a second pass over the footage. Third sighting means the vehicle is still
approaching, so the frame is on the small side -- the measurement comes out
slightly conservative, never generous.

Which vehicles count: only those whose ground-contact point reaches the camera's
``inside_zone``, the same rule the extractor uses to decide that a vehicle
entered. Without it a street camera contributes every car that drove past, and
the sample stops being "vehicles at the access". ``--all-vehicles`` turns the
rule off for a deliberately different question.

The output is images plus ``proposals.jsonl``: boxes the model suggests, at a
low threshold. They are a starting point for a person, not ground truth. The
labels that count are the ones ``label_boxes_gui.py`` writes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.config import infer_camera, load_config  # noqa: E402
from plate_dataset.geometry import ZoneModel, bottom_center  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov"}


@dataclass
class _Track:
    hits: int = 0
    saved: bool = False
    inside_hits: int = 0


@dataclass
class _Stats:
    videos: int = 0
    frames_read: int = 0
    tracks_confirmed: int = 0
    images_written: int = 0
    skipped_existing: int = 0
    boxes_proposed: int = 0
    started: float = field(default_factory=time.perf_counter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=None,
                        help="Video file or directory of videos.")
    parser.add_argument("--list", type=Path, default=None, dest="video_list",
                        help="Text file with one video path per line. Takes "
                             "precedence over --videos.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Dataset root. Writes images/, proposals.jsonl, classes.txt.")
    parser.add_argument("--camera", default=None,
                        help="Camera name for the output folder. Defaults to the "
                             "camera inferred from the filename, else the video stem.")
    parser.add_argument("--model", type=Path, default=Path("yolo26m.pt"),
                        help="Detector used only to propose boxes (default: yolo26m.pt, "
                             "which finds more candidates than the n variant).")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Proposal threshold, deliberately low: a missed vehicle "
                             "is a hole in the sample, a false box costs one click.")
    parser.add_argument("--sample-interval", type=float, default=0.5,
                        help="Seconds between sampled frames (default: 0.5).")
    parser.add_argument("--confirm-after", type=int, default=3,
                        help="Sightings before a pass is exported (default: 3).")
    parser.add_argument("--zones", type=Path, default=Path("zones_config.json"),
                        help="Zone configuration used to decide whether a vehicle "
                             "reached the access (default: zones_config.json).")
    parser.add_argument("--all-vehicles", action="store_true",
                        help="Export every confirmed track, including vehicles that "
                             "never reached the premises. Off by default: on a street "
                             "camera that would fill the sample with passing traffic.")
    parser.add_argument("--inside-hits", type=int, default=2,
                        help="Samples inside the access zone before a pass counts "
                             "(default: 2). A vehicle driving past the threshold can "
                             "graze it once; one that enters stays inside.")
    parser.add_argument("--hour", default=None,
                        help="Only videos whose name contains this YYYYMMDDHH.")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Stop after this many images (0 = no limit).")
    parser.add_argument("--device", default=None, help="cuda, cpu, ... (default: auto).")
    parser.add_argument("--force", action="store_true",
                        help="Re-export images that already exist.")
    return parser


def videos_for(path: Path | None, hour: str | None,
               listing: Path | None = None) -> list[Path]:
    if listing is not None:
        lines = (line.strip() for line in
                 listing.read_text(encoding="utf-8").splitlines())
        return [Path(line) for line in lines if line and not line.startswith("#")]
    if path is None:
        raise SystemExit("give --videos or --list")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"{path} is neither a video nor a directory")
    found = [p for p in sorted(path.rglob("*"))
             if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    if hour:
        found = [p for p in found if hour in p.name]
    return found


def camera_name(video: Path, override: str | None) -> str:
    if override:
        return override
    lowered = video.name.lower()
    for marker, name in (("ch16", "channel16"), ("ch6", "channel6")):
        if marker in lowered:
            return name
    return video.stem


def _zone_for(video: Path, cameras: dict, width: int, height: int,
              all_vehicles: bool):
    """The access polygon for this video, or None when filtering is off.

    A camera with no calibrated zones is not filtered: guessing a polygon would
    silently change which vehicles count as events.
    """
    if all_vehicles or not cameras:
        return None
    camera = infer_camera(video.name, cameras)
    if not (camera.inside_zone and camera.tripwire):
        print(f"[export] {camera.name} has no inside_zone; exporting every track",
              flush=True)
        return None
    return ZoneModel(camera.street_zone, camera.inside_zone, camera.tripwire,
                     width, height)


def export_video(model, video: Path, output: Path, camera: str,
                 args: argparse.Namespace, stats: _Stats, proposals,
                 cameras: dict) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"[export] cannot open {video}", flush=True)
        return
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps * args.sample_interval))

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zone = _zone_for(video, cameras, width, height, args.all_vehicles)

    tracks: dict[int, _Track] = {}
    sample = 0
    frame_index = 0
    stats.videos += 1
    print(f"[export] {video.name} -> {camera}, 1 every {step} frames, "
          f"inside_zone={'off' if zone is None else 'on'}", flush=True)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step:
            frame_index += 1
            continue

        result = model.track(frame, persist=True, verbose=False, conf=args.conf,
                             classes=[2, 3, 5, 7], imgsz=640, device=args.device)[0]
        stats.frames_read += 1
        ids = ([] if result.boxes is None or result.boxes.id is None
               else [int(v) for v in result.boxes.id])
        boxes = ([] if result.boxes is None
                 else [[int(v) for v in b] for b in result.boxes.xyxy.cpu().numpy()])
        scores = ([] if result.boxes is None
                  else [float(v) for v in result.boxes.conf.cpu().numpy()])

        confirmed = []
        for track_id, box in zip(ids, boxes, strict=False):
            state = tracks.setdefault(track_id, _Track())
            state.hits += 1
            if zone is not None and zone.in_inside(bottom_center(box)):
                state.inside_hits += 1
            reached = zone is None or state.inside_hits >= args.inside_hits
            if state.hits >= args.confirm_after and reached and not state.saved:
                confirmed.append(track_id)

        if confirmed:
            name = f"{video.stem}_f{frame_index:07d}.jpg"
            destination = output / "images" / camera / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not args.force:
                stats.skipped_existing += 1
            else:
                cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                stats.images_written += 1
            for track_id in confirmed:
                tracks[track_id].saved = True
                stats.tracks_confirmed += 1
            proposals.write(json.dumps({
                "image": f"{camera}/{name}",
                "camera": camera,
                "video": video.name,
                "frame_index": frame_index,
                "video_seconds": round(sample * args.sample_interval, 3),
                "track_ids": confirmed,
                "boxes": boxes,
                "confidences": scores,
            }) + "\n")
            proposals.flush()
            stats.boxes_proposed += len(boxes)

        sample += 1
        frame_index += 1

        if args.max_images and stats.images_written >= args.max_images:
            print(f"[export] reached --max-images {args.max_images}", flush=True)
            break

    capture.release()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    videos = videos_for(args.videos, args.hour, args.video_list)
    if not videos:
        raise SystemExit(f"no videos found under {args.videos}"
                         + (f" matching {args.hour}" if args.hour else ""))

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "classes.txt").write_text("vehiculo\n", encoding="utf-8")

    cameras: dict = {}
    if not args.all_vehicles and args.zones.exists():
        cameras, _ = load_config(args.zones)
    elif not args.all_vehicles:
        print(f"[export] {args.zones} not found; exporting every confirmed track",
              flush=True)

    stats = _Stats()
    # Append, never truncate: a second invocation into the same output would
    # otherwise erase the proposals of the first. Duplicates collapse on load,
    # because the GUI keys proposals by image path.
    with (args.output / "proposals.jsonl").open("a", encoding="utf-8") as proposals:
        for video in videos:
            export_video(model, video, args.output,
                         camera_name(video, args.camera), args, stats, proposals,
                         cameras)

    elapsed = time.perf_counter() - stats.started
    print(
        f"[export] videos={stats.videos} sampled_frames={stats.frames_read} "
        f"passes={stats.tracks_confirmed} images={stats.images_written} "
        f"skipped={stats.skipped_existing} boxes={stats.boxes_proposed} "
        f"in {elapsed:.0f}s"
    )
    print(f"[export] next: uv run python tools/label_boxes_gui.py "
          f"--root {args.output}")


if __name__ == "__main__":
    main()
