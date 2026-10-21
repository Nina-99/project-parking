"""Extract balanced vehicle crops from the attribute queue into a folder.

    uv run python tools/extract_balanced_crops.py \
        --dataset /home/nina/FNI2025/ProjectParking/dataset \
        --per-class 300

Reads ``attributes/manifest.jsonl`` and copies, for every detector class
(car / moto / bus / truck), up to ``--per-class`` deduplicated crops into
``attributes/balanced/<class>/``. One crop per vehicle track (the most typical
one), so a single car is never counted dozens of times.

Why this exists: the type classifier's dataset has zero motorcycles as the main
vehicle, yet the attribute queue holds thousands of them. This tool surfaces
them, balanced per class, so they can be labelled and folded into training.

Caveat: the queue carries only the COCO detector hint, not your body types. A
car is a car for the detector, so automovil/vagoneta/camioneta cannot be split
from here — that still needs the human labeller. What this *does* solve is the
missing `motocicleta` (and gives candidates for `otro`).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.attributes import image_path, iter_manifest, pick_representatives  # noqa: E402
from plate_dataset.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

COCO_NAMES = {2: "car", 3: "moto", 5: "bus", 7: "truck"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy balanced, deduplicated crops out of the attribute queue.")
    parser.add_argument("--dataset", type=Path, help="Dataset root (defaults to config).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--per-class", type=int, default=300,
                        help="Max crops to keep per detector class.")
    parser.add_argument("--kinds", type=str, default="",
                        help="Comma-separated kinds to keep (e.g. entry,exit). Empty = all.")
    parser.add_argument("--classes", type=str, default="",
                        help="Comma-separated detector classes to keep (car,moto,bus,truck). Empty = all.")
    parser.add_argument("--output", type=Path, default=Path("attributes") / "balanced",
                        help="Output folder, relative to the dataset root.")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, glob = load_config(args.config)
    root = args.dataset or Path(glob.output_dir)

    manifest_path = root / "attributes" / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"[ERROR] no attribute queue at {manifest_path}")
        return

    wanted_kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    wanted_classes = {c.strip().lower() for c in args.classes.split(",") if c.strip()}
    rows = list(iter_manifest(manifest_path))
    if wanted_kinds:
        rows = [r for r in rows if r.get("kind") in wanted_kinds]
    if wanted_classes:
        wanted_ids = {i for i, n in COCO_NAMES.items() if n in wanted_classes}
        rows = [r for r in rows if r.get("vehicle_class") in wanted_ids]
    print(f"[balance] candidate rows  : {len(rows)}")

    present = [r for r in rows if image_path(root, r).exists()]
    by_class: dict[int, list[dict]] = defaultdict(list)
    for row in present:
        by_class[row.get("vehicle_class")].append(row)

    rng = random.Random(args.seed)
    out_root = root / args.output
    for vehicle_class in sorted(k for k in by_class if k is not None):
        reps = pick_representatives(by_class[vehicle_class])
        reps = sorted(reps, key=lambda r: str(r.get("stem")))
        rng.shuffle(reps)
        chosen = reps[:args.per_class]
        name = COCO_NAMES.get(vehicle_class, f"class{vehicle_class}")
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for row in chosen:
            src = image_path(root, row)
            dst = out_dir / f"{row['stem']}.jpg"
            try:
                shutil.copy2(src, dst)
                copied += 1
            except OSError as exc:
                print(f"[warn] skip {src.name}: {exc}")
        print(f"[balance] {name:6s} -> {copied:4d} crops  ({out_dir})")

    print(f"[balance] done. Root: {out_root}")


if __name__ == "__main__":
    main()
