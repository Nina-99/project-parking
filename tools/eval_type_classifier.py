# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Evaluate a type (or colour) classifier checkpoint on a dataset's test split.

    uv run python tools/eval_type_classifier.py \
        --dataset "/mnt/nina/SmartParking/type dataset" \
        --checkpoint "/mnt/nina/SmartParking/checkpoints/resnet18_type_best.pt"

To compare two models fairly, run it twice on the SAME ``--dataset``: same test
split, same preprocessing, same metrics. Training-time numbers are not comparable
across runs because each run's split differs.

Class order is taken from the checkpoint's own ``*_metrics.json`` when present,
so a model trained with a different class order is still reported correctly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _train_common import (  # noqa: E402
    MIN_TEST_SUPPORT,
    VehicleCropDataset,
    build_model,
    build_transforms,
    confusion,
    per_class_metrics,
    print_summary,
    summarise,
)


def metrics_classes(checkpoint: Path) -> list[str] | None:
    stem = checkpoint.stem
    prefix = stem[:-5] if stem.endswith("_best") else stem
    for candidate in [checkpoint.parent / f"{prefix}_metrics.json"] + \
            sorted(checkpoint.parent.glob("*_metrics.json")):
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("classes"):
                return list(data["classes"])
    return None


def _print_by_type(
    types_path: Path,
    ds: VehicleCropDataset,
    y_true: list[int],
    y_pred: list[int],
    classes: list[str],
) -> None:
    """Per-type breakdown, as a diagnostic rather than a reported number.

    Splitting the test set by type leaves each group below the support gate, so
    no macro average is claimed here: the table shows where the errors live, not
    a headline score.
    """
    type_of: dict[str, str] = {}
    for line in types_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            type_of[row["stem"]] = row["type"]

    groups: dict[str, list[int]] = defaultdict(list)
    for i, (path, _) in enumerate(ds.items):
        vehicle_type = type_of.get(path.stem)
        if vehicle_type:
            groups[vehicle_type].append(i)
    if not groups:
        print(f"\n[eval] {types_path.name} matched no test crop")
        return

    print("\n=== per vehicle type (diagnostic; groups are too small for a macro) ===")
    for vehicle_type in sorted(groups):
        indices = groups[vehicle_type]
        sub = confusion([y_true[i] for i in indices], [y_pred[i] for i in indices], len(classes))
        _, _, f1s = per_class_metrics(sub)
        parts = []
        for index, name in enumerate(classes):
            support = int(sub[index, :].sum())
            if support:
                parts.append(f"{name} {f1s[index]:.2f} (n={support})")
        print(f"  {vehicle_type} (n={len(indices)}): " + "  ".join(parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a classifier checkpoint.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--types", type=Path,
                        help="types.jsonl written by build_color_dataset.py. When given, "
                             "metrics are also broken down by vehicle type.")
    parser.add_argument("--min-test-support", type=int, default=MIN_TEST_SUPPORT,
                        help="Classes with fewer samples are shown but not reported.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data = yaml.safe_load((args.dataset / "data.yaml").read_text(encoding="utf-8"))
    dataset_names = list(data["names"])
    classes = metrics_classes(args.checkpoint) or dataset_names
    if classes != dataset_names:
        print(f"[eval] WARNING: checkpoint classes {classes} != dataset {dataset_names}")

    _, eval_tf = build_transforms(args.img_size)
    ds = VehicleCropDataset(args.dataset, args.split, eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"[eval] {args.checkpoint.name} on {args.dataset.name}/{args.split} "
          f"({len(ds)} images, {len(classes)} classes)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, len(classes)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, targets in loader:
            preds = model(images.to(device)).argmax(1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(targets.tolist())

    matrix = confusion(y_true, y_pred, len(classes))
    summary = summarise(matrix, classes, args.min_test_support)
    print_summary(summary, classes)
    if args.types:
        _print_by_type(args.types, ds, y_true, y_pred, classes)

    print("confusion (rows=true):")
    print("        " + " ".join(f"{n[:6]:>7}" for n in classes))
    for i, name in enumerate(classes):
        print(f"{name[:6]:>6} " + " ".join(f"{v:7d}" for v in matrix[i]))


if __name__ == "__main__":
    main()
