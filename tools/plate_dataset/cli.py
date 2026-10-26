# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Command-line entry point for the plate-dataset extractor."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from plate_dataset.config import (
    DEFAULT_CONFIG_PATH,
    infer_camera,
    list_videos,
    load_config,
    resolve_weights,
)
from plate_dataset.pipeline import Extractor
from plate_dataset.plate_ocr import PlateReader
from plate_dataset.writer import DatasetWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a license-plate dataset from recorded videos."
    )
    parser.add_argument("--input", type=Path, help="Directory with input videos.")
    parser.add_argument("--video", type=Path, help="Process a single video file.")
    parser.add_argument("--output", type=Path, help="Dataset output directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to zones_config.json.")
    parser.add_argument("--model", help="YOLO weights (overrides config).")
    parser.add_argument("--device", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--camera", help="Force a camera key from the config.")
    parser.add_argument("--no-negatives", action="store_true",
                        help="Do not store passing vehicles.")
    parser.add_argument("--max-frames", type=int,
                        help="Analyse only the first N frames of each video (for testing).")
    parser.add_argument("--index", type=int,
                        help="1-based index of the video to process from --input.")
    parser.add_argument("--list", action="store_true",
                        help="List the videos in --input with their index and exit.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip videos already recorded in <output>/.processed.jsonl.")
    parser.add_argument("--no-half", action="store_true",
                        help="Disable FP16 inference (slower; rarely needed).")
    parser.add_argument("--limit", type=int, help="Process at most N videos.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cameras, glob = load_config(args.config)

    if args.input:
        glob.input_dir = str(args.input)
    if args.output:
        glob.output_dir = str(args.output)
    if args.model:
        glob.model_name = args.model
    if args.device is not None:
        glob.device = args.device
    if args.no_negatives:
        glob.save_negatives = False
    if args.no_half:
        glob.half = False

    if args.camera and args.camera not in cameras:
        print(f"[ERROR] unknown camera '{args.camera}'; available: {', '.join(cameras)}")
        return

    if args.video:
        if not args.video.exists():
            print(f"[ERROR] video not found: {args.video}")
            return
        videos = [args.video]
    else:
        input_dir = Path(glob.input_dir)
        if not input_dir.exists():
            print(f"[ERROR] input directory not found: {input_dir}")
            return
        available = list_videos(input_dir)
        if not available:
            print(f"[WARN] no videos found in {input_dir}")
            return
        if args.list:
            for i, path in enumerate(available, 1):
                print(f"{i:>4}  {path.name}")
            return
        if args.index is not None:
            if not 1 <= args.index <= len(available):
                print(f"[ERROR] --index out of range (1..{len(available)})")
                return
            videos = [available[args.index - 1]]
        else:
            videos = available
            if args.limit:
                videos = videos[:args.limit]

    print(f"Videos: {len(videos)} | output: {glob.output_dir} | config: {args.config}")
    weights = resolve_weights(glob.model_name)
    if str(weights).endswith(".engine"):
        glob.half = False  # TensorRT engines are already FP16
    print(f"Model: {weights} | device: {glob.device} | fp16: {glob.half} | tracker: {glob.tracker}")

    model = YOLO(weights)
    reader = PlateReader(
        min_conf=glob.ocr_min_conf,
        min_plate_px=glob.min_plate_px,
        plate_y_min=glob.plate_y_min,
        plate_y_max=glob.plate_y_max,
    )
    output_dir = Path(glob.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / ".processed.jsonl"
    done: set[str] = set()
    if args.resume and state_path.exists():
        done = {ln.strip() for ln in state_path.read_text(encoding="utf-8").splitlines() if ln.strip()}
        print(f"Resume: {len(done)} video(s) already done")

    writer = DatasetWriter(output_dir, save_negatives=glob.save_negatives)
    extractor = Extractor(glob, model, reader, writer)

    for index, video in enumerate(videos, 1):
        if args.resume and video.stem in done:
            print(f"[{index}/{len(videos)}] skip (done): {video.name}")
            continue
        camera = cameras[args.camera] if args.camera else infer_camera(video.name, cameras)
        print(f"\n[{index}/{len(videos)}] {video.name}", flush=True)
        skips_before = writer.skips
        extractor.run_video(video, camera, max_frames=args.max_frames)
        if writer.skips > skips_before:
            print(f"[WARN] {writer.skips - skips_before} sample(s) lost to write errors — "
                  f"NOT marking '{video.name}' as done (a --resume run will retry it).", flush=True)
            continue
        with state_path.open("a", encoding="utf-8") as fh:
            fh.write(video.stem + "\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
