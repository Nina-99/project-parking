"""Build a colour dataset from the flat labelling pool.

    uv run python tools/build_color_dataset.py \
        --pool "/mnt/nina/SmartParking/car type" \
        --output "/mnt/nina/SmartParking/color dataset v2" \
        --exclude-types minibus --min-count 10

The label GUI writes a flat pool (images/ + labels_color/ + data_color.yaml) and
does not split it. This tool turns that pool into the ``train/valid/test`` layout
``train_color_classifier.py`` expects, using the COLOUR labels.

Three things it does that a plain copy would not:

* Drops colour classes with fewer than ``--min-count`` samples (a class with 1
  example cannot be learned and only skews the metrics).
* Drops crops whose vehicle type is in ``--exclude-types``. Minibuses are left
  out by project policy: their body is covered by livery whose dominant colour
  depends on the viewing angle, so the same vehicle is "blue" in profile and
  "white" from the rear -- the view the gate camera actually gets. That makes
  their colour label incomparable with a private car's base paint. The filter
  runs at build time, never by deleting crops from the pool: the pool stays
  intact and auditable.
* Splits by *frame*: crops exported from the same frame share the vehicle, so a
  random per-crop split would leak near-duplicates into validation.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")
FRAME_RE = re.compile(r"^(?P<frame>.*)_b\d+$")


def frame_of(stem: str) -> str:
    match = FRAME_RE.match(stem)
    return match.group("frame") if match else stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a colour dataset from the pool.")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels-dir", default="labels_color",
                        help="Label folder inside the pool (labels_color or labels).")
    parser.add_argument("--data-yaml", default="data_color.yaml",
                        help="Which yaml holds the names (data_color.yaml or data.yaml).")
    parser.add_argument("--type-labels-dir", default="labels",
                        help="Where the vehicle-type labels live; needed by --exclude-types.")
    parser.add_argument("--type-yaml", default="data.yaml",
                        help="Yaml holding the vehicle-type names.")
    parser.add_argument("--exclude-types", default="",
                        help="Comma-separated vehicle types to drop, e.g. 'minibus'. "
                             "Empty keeps everything.")
    parser.add_argument("--min-count", type=int, default=10,
                        help="Drop classes with fewer samples than this.")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pool = args.pool
    data = yaml.safe_load((pool / args.data_yaml).read_text(encoding="utf-8"))
    old_names = list(data["names"])

    items: list[tuple[Path, Path, str, int]] = []
    excluded = {name.strip() for name in args.exclude_types.split(",") if name.strip()}
    excluded_idx: set[int] = set()
    if excluded:
        type_data = yaml.safe_load((pool / args.type_yaml).read_text(encoding="utf-8"))
        type_names = list(type_data["names"])
        unknown = excluded - set(type_names)
        if unknown:
            raise SystemExit(
                f"[color] unknown --exclude-types {sorted(unknown)}; known: {type_names}")
        excluded_idx = {i for i, name in enumerate(type_names) if name in excluded}

    skipped_by_type = 0
    missing_type = 0
    for image in sorted((pool / "images").glob("*.jpg")):
        label = pool / args.labels_dir / f"{image.stem}.txt"
        if not label.exists():
            continue
        parts = label.read_text(encoding="utf-8").split()
        if not parts or not parts[0].isdigit():
            continue
        if excluded_idx:
            type_label = pool / args.type_labels_dir / f"{image.stem}.txt"
            if type_label.exists():
                type_parts = type_label.read_text(encoding="utf-8").split()
                if type_parts and type_parts[0].isdigit() and int(type_parts[0]) in excluded_idx:
                    skipped_by_type += 1
                    continue
            else:
                missing_type += 1
        items.append((image, label, frame_of(image.stem), int(parts[0])))

    if excluded:
        note = f"; {missing_type} crops without a type label were kept" if missing_type else ""
        print(f"[color] excluded vehicle types {sorted(excluded)}: "
              f"{skipped_by_type} crops dropped{note}")
    else:
        print("[color] excluded vehicle types: none")

    counts = Counter(cls for _, _, _, cls in items)
    keep = [i for i in range(len(old_names)) if counts.get(i, 0) >= args.min_count]
    remap = {old: new for new, old in enumerate(keep)}
    names = [old_names[i] for i in keep]
    print(f"[color] kept {len(names)} classes: "
          f"{ {names[n]: counts[i] for n, i in enumerate(keep)} }")
    dropped = [old_names[i] for i in range(len(old_names)) if i not in remap]
    print(f"[color] dropped (<{args.min_count}): {dropped}")

    kept = [(img, frame, remap[cls]) for img, _, frame, cls in items if cls in remap]
    if not kept:
        print("[color] no samples left")
        return

    by_frame: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for img, frame, cls in kept:
        by_frame[frame].append((img, cls))

    # Stratify frames by their dominant colour so every split sees every class.
    strata: dict[int, list[str]] = defaultdict(list)
    for frame, crops in by_frame.items():
        dominant = Counter(cls for _, cls in crops).most_common(1)[0][0]
        strata[dominant].append(frame)

    rng = random.Random(args.seed)
    assign: dict[str, str] = {}
    for cls in sorted(strata):
        frames = sorted(strata[cls])
        rng.shuffle(frames)
        n = len(frames)
        n_test = round(n * args.test_frac)
        n_val = round(n * args.val_frac)
        for f in frames[:n_test]:
            assign[f] = "test"
        for f in frames[n_test:n_test + n_val]:
            assign[f] = "valid"
        for f in frames[n_test + n_val:]:
            assign[f] = "train"

    for split in SPLITS:
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)

    written: dict[str, Counter] = {s: Counter() for s in SPLITS}
    for frame, crops in by_frame.items():
        split = assign[frame]
        for img, cls in crops:
            shutil.copy2(img, args.output / split / "images" / img.name)
            (args.output / split / "labels" / f"{img.stem}.txt").write_text(
                f"{cls} 0.5 0.5 1 1\n", encoding="utf-8")
            written[split][names[cls]] += 1

    (args.output / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(names)}\nnames: {names}\n", encoding="utf-8")

    total = sum(sum(c.values()) for c in written.values())
    print(f"[color] {total} crops from {len(by_frame)} frames")
    for split in SPLITS:
        print(f"  {split:5s}: " + "  ".join(f"{n}={written[split].get(n, 0)}" for n in names))
    print(f"[color] written to {args.output}")


if __name__ == "__main__":
    main()
