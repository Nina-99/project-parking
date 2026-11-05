# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Train the single-class plate detector produced by the review GUI.

    uv run python tools/train_plate_detector.py \
        --data /home/nina/FNI2025/ProjectParking/plates_det/dataset/data.yaml \
        --epochs 100 --imgsz 640 --batch 8 \
        --save-dir /home/nina/FNI2025/ProjectParking/plates_det/runs

The task is deliberately narrow: one class, high-contrast rectangular target, a
fixed set of cameras. That is why a nano backbone is enough and why a few
hundred verified boxes can work where the body-type classifier needed thousands.

``--batch 8`` is not conservative, it is the budget: the training box is a 4 GB
laptop GPU. Raise it only after watching nvidia-smi.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="data.yaml written by build_plate_detection_dataset.py")
    parser.add_argument("--model", default="yolo26n.pt",
                        help="Starting weights. COCO-pretrained, fine-tuned to 1 class.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stop after N epochs without mAP improvement.")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--name", default="plate_det")
    parser.add_argument("--device", default="",
                        help="'', '0' for the first GPU, 'cpu'. Empty = auto.")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs at imgsz 320: proves the wiring, trains "
                             "nothing useful. Run this before a real run.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.data.exists():
        raise SystemExit(f"missing {args.data}")

    # Ultralytics resolves a relative ``project`` against its own runs_dir, so
    # "../plates_det/runs" silently landed inside software/runs/. Resolve once.
    save_dir = args.save_dir.expanduser().resolve()
    data_path = args.data.resolve()

    from ultralytics import YOLO

    epochs = 2 if args.smoke else args.epochs
    imgsz = 320 if args.smoke else args.imgsz

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(save_dir),
        name=args.name,
        exist_ok=True,
        device=args.device or None,
        # A plate is a small target: mosaic helps, but vertical flips never
        # happen on a road and teach a distribution that does not exist.
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        verbose=True,
    )

    metrics = getattr(results, "results_dict", None) or {}
    best = Path(results.save_dir) / "weights" / "best.pt"

    evaluation = {}
    if best.exists():
        validation = YOLO(str(best)).val(data=str(data_path), split="test",
                                         imgsz=imgsz, batch=args.batch,
                                         device=args.device or None,
                                         project=str(save_dir),
                                         name=f"{args.name}_test", exist_ok=True)
        evaluation = dict(getattr(validation, "results_dict", None) or {})

    report = {
        "data": str(data_path),
        "model": args.model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": args.batch,
        "smoke": args.smoke,
        "train": {k: float(v) for k, v in metrics.items()
                  if isinstance(v, (int, float))},
        "test": {k: float(v) for k, v in evaluation.items()
                 if isinstance(v, (int, float))},
        "best_weights": str(best) if best.exists() else None,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / f"{args.name}_metrics.json").write_text(
        json.dumps(report, indent=4) + "\n", encoding="utf-8"
    )

    print(f"[plate-det] best weights: {best}")
    print(f"[plate-det] metrics: {save_dir / f'{args.name}_metrics.json'}")
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                "metrics/precision(B)", "metrics/recall(B)"):
        if key in evaluation:
            print(f"  test {key}: {evaluation[key]:.4f}")


if __name__ == "__main__":
    main()
