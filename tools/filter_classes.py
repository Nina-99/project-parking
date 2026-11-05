# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Drop classes from a YOLO dataset and remap the remaining indices.

    uv run python tools/filter_classes.py \
        --dataset "/mnt/nina/SmartParking/type car grouped" \
        --output  "/mnt/nina/SmartParking/type car 6cls" \
        --drop otro

Why: the classifier must model only the vehicles that can actually pass the gate.
`otro` (bus/camión) never enters the premises, so keeping it adds an untrained
class at inference. Dropping it here also removes its labels from multi-box
images and compacts the class indices, so the resulting dataset trains a clean
6-class model without touching the source.

Images left with zero boxes after the drop are skipped (they were `otro`-only).
Splits are preserved, so a leak-free split stays leak-free.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drop YOLO classes and remap indices.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop", nargs="+", required=True,
                        help="Class names to remove, e.g. --drop otro")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data = yaml.safe_load((args.dataset / "data.yaml").read_text(encoding="utf-8"))
    names = list(data["names"])
    drop = set(args.drop)
    missing = drop - set(names)
    if missing:
        print(f"[ERROR] classes not in dataset: {sorted(missing)}; have {names}")
        return

    keep = [i for i, name in enumerate(names) if name not in drop]
    remap = {old: new for new, old in enumerate(keep)}
    new_names = [names[i] for i in keep]
    print(f"[filter] {names} -> {new_names}")

    counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    written = {s: 0 for s in SPLITS}
    for split in SPLITS:
        img_dir = args.dataset / split / "images"
        lbl_dir = args.dataset / split / "labels"
        if not img_dir.exists():
            continue
        out_img = args.output / split / "images"
        out_lbl = args.output / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img in sorted(img_dir.glob("*.jpg")):
            label = lbl_dir / f"{img.stem}.txt"
            if not label.exists():
                continue
            rows: list[str] = []
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                old = int(float(parts[0]))
                if old not in remap:
                    continue
                rows.append(f"{remap[old]} " + " ".join(parts[1:5]))
            if not rows:
                continue  # image was only the dropped class
            shutil.copy2(img, out_img / img.name)
            (out_lbl / f"{img.stem}.txt").write_text("\n".join(rows) + "\n",
                                                    encoding="utf-8")
            written[split] += 1
            for row in rows:
                counts[split][new_names[int(row.split()[0])]] += 1

    yaml_text = ("train: train/images\nval: valid/images\ntest: test/images\n\n"
                 f"nc: {len(new_names)}\nnames: {new_names}\n")
    (args.output / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("[filter] images kept:", {s: written[s] for s in SPLITS})
    for split in SPLITS:
        print(f"  {split:5s}: " + "  ".join(f"{n}={counts[split].get(n, 0)}"
                                            for n in new_names))
    print(f"[filter] written to {args.output}")


if __name__ == "__main__":
    main()
