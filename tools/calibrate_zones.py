"""Interactive real-time zone calibrator.

    uv run python tools/calibrate_zones.py \
        --video /mnt/PARKING/ver/C.OCCIDENTE_ch16_main_20260415190012_20260415200012.mp4 \
        --camera channel16

Plays the video with tracking on and draws trajectories live: green curves are
vehicles entering the premises, red curves are street traffic. Adjust the
polygons with the mouse and press [s] to save them into zones_config.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.calibrate import ZoneCalibrator  # noqa: E402
from plate_dataset.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    infer_camera,
    list_videos,
    load_config,
    resolve_weights,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactively calibrate camera zones.")
    parser.add_argument("--video", type=Path, help="Video file to calibrate against.")
    parser.add_argument("--input", type=Path, help="Directory with videos.")
    parser.add_argument("--index", type=int,
                        help="1-based index of the video to use from --input.")
    parser.add_argument("--list", action="store_true",
                        help="List the videos in --input with their index and exit.")
    parser.add_argument("--camera", help="Camera key from the config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", help="YOLO weights (overrides config).")
    parser.add_argument("--device", help="Inference device, e.g. 0 or cpu.")
    return parser


def _resolve_device(device: str | int) -> str | int:
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
    glob.device = _resolve_device(args.device if args.device is not None else glob.device)

    if args.camera and args.camera not in cameras:
        print(f"[ERROR] unknown camera '{args.camera}'; available: {', '.join(cameras)}")
        return

    video = args.video
    if video is None:
        input_dir = args.input or Path(glob.input_dir)
        available = list_videos(input_dir)
        if not available:
            print(f"[ERROR] no videos found in {input_dir}")
            return
        if args.list:
            for i, path in enumerate(available, 1):
                print(f"{i:>4}  {path.name}")
            return
        if args.index is not None:
            if not 1 <= args.index <= len(available):
                print(f"[ERROR] --index out of range (1..{len(available)})")
                return
            video = available[args.index - 1]
        else:
            video = available[0]
    if not Path(video).exists():
        print(f"[ERROR] video not found: {video}")
        return

    camera = cameras[args.camera] if args.camera else infer_camera(Path(video).name, cameras)
    weights = resolve_weights(glob.model_name)
    if str(weights).endswith(".engine"):
        glob.half = False
    print(f"video={Path(video).name} camera={camera.name} "
          f"model={weights} device={glob.device}")
    ZoneCalibrator(Path(video), camera, glob, args.config).run()


if __name__ == "__main__":
    main()
