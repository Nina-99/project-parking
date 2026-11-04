# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Score vehicle detection against human boxes (H1.1, mAP@50 >= 0.80).

    uv run python tools/eval_vehicle_detection.py \
        --root /home/nina/Smart_Parking_dataset/detection \
        --model yolo26n.pt --model yolo26s.pt

Every image under ``root/images`` must have a label file under ``root/labels``;
a missing one aborts instead of being scored as "no vehicles". An empty label
file is a deliberate negative and counts as one.

Two numbers, because they answer different questions:

  AP@50                 the criterion H1.1 states, over the whole confidence
                        range. This is the headline.
  precision / recall    at one operating threshold, which is what the deployed
  at --operating-conf   system actually does.

The interval on AP resamples whole images: two boxes in the same frame are not
independent, and resampling boxes would report an interval narrower than the
evidence supports. The interval on recall is Wilson over ground-truth boxes and
is labelled as such, since it assumes independence the frames cannot offer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection_dataset import (  # noqa: E402
    ImageResult,
    Prediction,
    average_precision,
    bootstrap_ap,
    image_paths,
    label_path_for,
    match_per_prediction,
    read_labels,
    wilson,
)

VEHICLE_CLASSES = [2, 3, 5, 7]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", action="append", type=Path, required=True,
                        help="Detector to score. Repeat to compare, e.g. n and s.")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="Match threshold (default: 0.5, the @50 in mAP@50).")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Floor for the precision-recall curve (default: 0.001).")
    parser.add_argument("--operating-conf", type=float, default=0.5,
                        help="Threshold for the precision/recall row (default: 0.5, "
                             "the value the system serves with).")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--subset", choices=("all", "blind", "proposed"), default="all",
                        help="Score only the images reviewed without proposals, or "
                             "only the ones reviewed with them. Requires "
                             "review_journal.jsonl. The blind subset is the only one "
                             "where a miss could be seen at all.")
    parser.add_argument("--rounds", type=int, default=2000,
                        help="Bootstrap rounds for the AP interval (default: 2000).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", type=Path, default=None)
    return parser


def predict(model, frame, args: argparse.Namespace) -> list[Prediction]:
    result = model.predict(frame, verbose=False, conf=args.conf,
                           classes=VEHICLE_CLASSES, imgsz=args.imgsz,
                           device=args.device)[0]
    if result.boxes is None:
        return []
    return [
        Prediction(box=tuple(int(v) for v in box), confidence=float(score))
        for box, score in zip(result.boxes.xyxy.cpu().numpy(),
                              result.boxes.conf.cpu().numpy(), strict=True)
    ]


def score(model, images: list[Path], images_root: Path, labels_root: Path,
          args: argparse.Namespace) -> dict[str, object]:
    results: list[ImageResult] = []
    ground_truths = 0
    flagged = 0
    detected = 0

    for image in images:
        frame = cv2.imread(str(image))
        if frame is None:
            raise SystemExit(f"cannot read {image}")
        height, width = frame.shape[:2]
        boxes = read_labels(label_path_for(image, images_root, labels_root), width, height)
        ground_truths += len(boxes)

        predictions = sorted(predict(model, frame, args),
                             key=lambda item: item.confidence, reverse=True)
        outcome = match_per_prediction(predictions, boxes, args.iou)
        results.append(ImageResult(
            scored=[(prediction.confidence, hit)
                    for prediction, hit in zip(predictions, outcome, strict=True)],
            positives=len(boxes),
        ))
        for prediction, hit in zip(predictions, outcome, strict=True):
            if prediction.confidence >= args.operating_conf:
                flagged += 1
                detected += 1 if hit else 0

    ap = average_precision(results)
    low, high = bootstrap_ap(results, rounds=args.rounds)
    recall_low, recall_high = wilson(detected, ground_truths)
    return {
        "images": len(images),
        "ground_truth_boxes": ground_truths,
        "ap50": round(ap, 4),
        "ap50_ci": [round(low, 4), round(high, 4)],
        "operating_conf": args.operating_conf,
        "detected_at_operating": detected,
        "false_positives_at_operating": flagged - detected,
        "precision_at_operating": round(detected / flagged, 4) if flagged else 0.0,
        "recall_at_operating": round(detected / ground_truths, 4) if ground_truths else 0.0,
        "recall_ci_wilson": [round(recall_low, 4), round(recall_high, 4)],
    }


def select_subset(images: list[Path], images_root: Path, root: Path,
                  subset: str) -> list[Path]:
    """Keep only the images the reviewer saw blind, or saw with proposals."""
    if subset == "all":
        return images
    journal = root / "review_journal.jsonl"
    if not journal.exists():
        raise SystemExit(f"--subset {subset} needs {journal}, which does not exist")
    blind = {}
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        blind[row["image"]] = bool(row.get("blind"))
    wanted = subset == "blind"
    selected = [image for image in images
                if blind.get(str(image.relative_to(images_root))) is wanted]
    if not selected:
        raise SystemExit(f"no images in the {subset} subset")
    return selected


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    images_root = args.root / "images"
    labels_root = args.root / "labels"
    images = image_paths(images_root)
    if not images:
        raise SystemExit(f"no images under {images_root}")
    images = select_subset(images, images_root, args.root, args.subset)

    missing = [p for p in images
               if not label_path_for(p, images_root, labels_root).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(images)} images have no label file.\n"
            f"Scoring them as 'no vehicles' would reward the detector for unlabelled "
            f"work, so nothing is evaluated until the set is complete.\n"
            f"Label them first:\n"
            f"    uv run python tools/label_boxes_gui.py --root {args.root} "
            f"--blind-every 4\n"
            f"first missing: {missing[0]}"
        )

    from ultralytics import YOLO

    reports = []
    for model_path in args.model:
        model = YOLO(str(model_path))
        report = score(model, images, images_root, labels_root, args)
        report["model"] = Path(model_path).name
        reports.append(report)
        print(f"[eval] {report['model']}: AP@50={report['ap50']} "
              f"CI={report['ap50_ci']} recall@{args.operating_conf}="
              f"{report['recall_at_operating']}", flush=True)
        del model

    print()
    print("| Modelo | AP@50 | IC 95 % (bootstrap) | Recall @ conf | P | IC 95 % (Wilson) |")
    print("|---|---|---|---|---|---|")
    for report in reports:
        print(
            f"| {report['model']} | {report['ap50']:.3f} | "
            f"[{report['ap50_ci'][0]:.3f}, {report['ap50_ci'][1]:.3f}] | "
            f"{report['recall_at_operating']:.3f} | "
            f"{report['precision_at_operating']:.3f} | "
            f"[{report['recall_ci_wilson'][0]:.3f}, "
            f"{report['recall_ci_wilson'][1]:.3f}] |"
        )
    print()
    print(f"images={reports[0]['images']} "
          f"ground_truth_boxes={reports[0]['ground_truth_boxes']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"[eval] wrote {args.json}")


if __name__ == "__main__":
    main()
