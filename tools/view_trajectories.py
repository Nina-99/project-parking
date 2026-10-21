"""Trajectory viewer CLI.

Draws tracked vehicle trajectories over a reference frame so the zone polygons
can be calibrated against real footage:

    python tools/view_trajectories.py --video /mnt/PARKING/ver/<file>.mp4 \
        --max-frames 3000 --out /tmp/opencode/traj.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    infer_camera,
    load_config,
    resolve_weights,
)
from plate_dataset.geometry import ZoneModel  # noqa: E402
from plate_dataset.trajectory import collect_tracks, render  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualise vehicle trajectories for zone calibration.")
    parser.add_argument("--video", type=Path, help="Specific video file.")
    parser.add_argument("--input", type=Path, help="Directory with videos (first one is used).")
    parser.add_argument("--camera", help="Camera key from the config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", help="YOLO weights (overrides config).")
    parser.add_argument("--device", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--start", type=int, default=0, help="First frame to process.")
    parser.add_argument("--max-frames", type=int, default=3000, help="Frames to process.")
    parser.add_argument("--out", type=Path, help="Output PNG path.")
    return parser


def resolve_device(device: str | int) -> str | int:
    if str(device) == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return device
    except Exception:
        pass
    return "cpu"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cameras, glob = load_config(args.config)
    if args.model:
        glob.model_name = args.model
    glob.device = resolve_device(args.device if args.device is not None else glob.device)

    if args.camera and args.camera not in cameras:
        print(f"[ERROR] unknown camera '{args.camera}'; available: {', '.join(cameras)}")
        return

    video = args.video
    if video is None:
        input_dir = args.input or Path(glob.input_dir)
        videos = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ) if input_dir.exists() else []
        if not videos:
            print(f"[ERROR] no videos found in {input_dir}")
            return
        video = videos[0]

    camera = cameras[args.camera] if args.camera else infer_camera(video.name, cameras)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] cannot open {video}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    zones = ZoneModel(camera.street_zone, camera.inside_zone, camera.tripwire, width, height)
    weights = resolve_weights(glob.model_name)
    if str(weights).endswith(".engine"):
        glob.half = False
    model = YOLO(weights)

    print(f"video={video.name} camera={camera.name} device={glob.device} "
          f"frames={args.start}..{args.start + args.max_frames}")

    base, states = collect_tracks(model, video, zones, camera, glob, args.start, args.max_frames)
    if base is None:
        print("[ERROR] no frames decoded")
        return

    canvas = render(base, zones, states, min_hits=glob.min_track_hits)
    out = args.out or Path("/tmp/opencode") / f"traj_{video.stem}_{args.start}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
