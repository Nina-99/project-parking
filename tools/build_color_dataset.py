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

The split is a *stable* function of the frame name, not of a seeded shuffle over
whatever the pool contains today. That matters more than it sounds: the pool is
still being labelled, and a global shuffle means every new crop moves hundreds of
existing frames between splits, silently invalidating every number already
published from the previous split. Hashing the frame name keeps each crop in the
split it was assigned the first time.

Rebuilding over an existing output directory is refused, because the moved crops
would stay behind in their old split and the dataset would report one thing while
containing another. Pass ``--force`` to replace it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing dataset in --output. Without it a "
                             "non-empty output is refused instead of mixed into.")
    return parser


def split_for(frame: str, test_cut: int, valid_cut: int) -> str:
    """Split from a hash of the frame name: same frame, same split, forever.

    A seeded shuffle over the current pool is not stable: adding one labelled
    crop moves hundreds of existing frames between splits and silently
    invalidates every number published from the previous split. Hashing keeps
    the assignment immutable while staying uniform, so each class still lands in
    every split in roughly the requested proportions.
    """
    bucket = int.from_bytes(
        hashlib.blake2b(frame.encode("utf-8"), digest_size=4).digest(), "big") % 100
    if bucket < test_cut:
        return "test"
    if bucket < valid_cut:
        return "valid"
    return "train"


def _existing_dataset(output: Path) -> bool:
    return any(
        (output / split / folder).is_dir() and any((output / split / folder).iterdir())
        for split in SPLITS
        for folder in ("images", "labels")
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pool = args.pool

    if _existing_dataset(args.output):
        if not args.force:
            raise SystemExit(
                f"[color] {args.output} already holds a dataset. Rebuilding on top "
                "would leave crops in the split they were previously assigned to. "
                "Pass --force to replace it, or choose another --output.")
        for split in SPLITS:
            shutil.rmtree(args.output / split, ignore_errors=True)
    data = yaml.safe_load((pool / args.data_yaml).read_text(encoding="utf-8"))
    old_names = list(data["names"])

    items: list[tuple[Path, Path, str, int]] = []
    type_of: dict[str, str] = {}
    type_names: list[str] = []
    excluded = {name.strip() for name in args.exclude_types.split(",") if name.strip()}
    excluded_idx: set[int] = set()
    type_yaml = pool / args.type_yaml
    if type_yaml.exists():
        type_data = yaml.safe_load(type_yaml.read_text(encoding="utf-8"))
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
        if type_names:
            type_label = pool / args.type_labels_dir / f"{image.stem}.txt"
            if type_label.exists():
                type_parts = type_label.read_text(encoding="utf-8").split()
                if type_parts and type_parts[0].isdigit():
                    index = int(type_parts[0])
                    if index < len(type_names):
                        type_of[image.stem] = type_names[index]
                    if excluded_idx and index in excluded_idx:
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

    # Uniform per-frame hashing is stratification in expectation: every class
    # lands in every split without any frame ever changing split later.
    test_cut = round(args.test_frac * 100)
    valid_cut = test_cut + round(args.val_frac * 100)
    assign = {frame: split_for(frame, test_cut, valid_cut) for frame in by_frame}

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

    # Sidecar for stratified evaluation: the colour dataset keeps colour labels
    # only, so the vehicle type has to travel beside it or it is lost.
    with (args.output / "types.jsonl").open("w", encoding="utf-8") as fh:
        for stem in sorted(type_of):
            fh.write(json.dumps({"stem": stem, "type": type_of[stem]}) + "\n")

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
