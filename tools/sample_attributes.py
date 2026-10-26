# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Reduce the vehicle-crop pool to a compact, labellable sample.

    # Deployment view (front/rear at the gate) — the default and recommended one
    uv run python tools/sample_attributes.py --dataset ~/FNI2025/ProjectParking/dataset --target 3000

    # Legacy: the general attribute queue (mostly sideways street traffic)
    uv run python tools/sample_attributes.py --dataset ... --source attributes

Why two sources: a vehicle's colour depends on the viewing angle. A minibus seen
sideways shows its livery (blue stripes on a white body); seen from the gate it
shows its rear, mostly white. Training a colour classifier on sideways crops and
deploying it at the gate measures the wrong thing. The `gate` source keeps only
crops where the plate face is toward the camera — the view the gate camera
actually gets — so training and inference share one distribution.

Output (same schema the labeller consumes)::

    attributes/sample_manifest.jsonl

Point the labeller at it with ``tools/label_attributes.py --sample``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.attributes import (  # noqa: E402
    image_path,
    iter_manifest,
    load_labelled,
    parse_stem,
    pick_representatives,
    stratified_sample,
    write_manifest,
)
from plate_dataset.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

DEFAULT_TARGET = 3000
DEFAULT_SEED = 7
# A car's front/rear is roughly as wide as it is tall (w/h ~ 1.3); a side view is
# far longer (w/h > 2). 1.8 cleanly separates the plate face from the profile.
DEFAULT_MAX_RATIO = 1.8
GATE_SOURCES = (
    ("vehicles/images", "gate_plate"),
    ("review/vehicles/images", "gate_noplate"),
)


def valid_label_stems(root: Path) -> set[str]:
    """Stems already labelled with a real colour/type (discards excluded).

    Passing these to :func:`pick_representatives` keeps the crops you already
    annotated pinned to their track, so re-sampling never re-asks a decided
    sample. Discarded stems are left out on purpose: their track gets a fresh,
    better crop instead of the unusable one.
    """
    labels_path = root / "attributes" / "labels.jsonl"
    if not labels_path.exists():
        return set()
    labelled = load_labelled(labels_path)
    return {
        stem for stem, row in labelled.items()
        if not row.get("discard") and row.get("color")
    }


def _infer_camera(video: str) -> str:
    lowered = video.lower()
    if "ch16" in lowered or "channel16" in lowered:
        return "channel16"
    if "ch6" in lowered or "channel6" in lowered:
        return "channel6"
    return "unknown"


def collect_gate_crops(root: Path, max_ratio: float) -> list[dict]:
    """Walk the gate-view folders and keep only plate-face crops."""
    rows: list[dict] = []
    for rel, kind in GATE_SOURCES:
        for path in sorted((root / rel).glob("*/*.jpg")):
            parsed = parse_stem(path.stem)
            if parsed is None:
                continue
            try:
                with Image.open(path) as im:
                    width, height = im.size
            except OSError:
                continue
            if height == 0 or width / height > max_ratio:
                continue
            rows.append({
                "stem": path.stem,
                "kind": kind,
                "camera": _infer_camera(parsed["video"]),
                "video": parsed["video"],
                "track_id": parsed["track_id"],
                "frame": parsed["frame"],
                "crop_hw": [height, width],
                "image": str(path.relative_to(root)),
            })
    return rows


def gate_pick(rows: list[dict]) -> list[dict]:
    """One crop per track: the most plate-face one (ratio closest to a rear view)."""
    target = 1.3
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["video"], row["track_id"]), []).append(row)
    picked: list[dict] = []
    for group in groups.values():
        picked.append(min(
            group,
            key=lambda r: (
                abs(r["crop_hw"][1] / r["crop_hw"][0] - target),
                -r["crop_hw"][0] * r["crop_hw"][1],
            ),
        ))
    return picked


def source_attributes(root: Path, args: argparse.Namespace) -> list[dict]:
    manifest_path = root / "attributes" / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"[ERROR] no attribute queue at {manifest_path}")
        return []
    wanted_kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    rows = list(iter_manifest(manifest_path))
    print(f"[sample] manifest rows    : {len(rows)}")
    if wanted_kinds:
        rows = [r for r in rows if r.get("kind") in wanted_kinds]
        print(f"[sample] kinds {sorted(wanted_kinds)}    : {len(rows)}")
    present = [r for r in rows if image_path(root, r).exists()]
    print(f"[sample] crops on disk    : {len(present)}")
    reps = pick_representatives(present, prefer=valid_label_stems(root))
    print(f"[sample] unique tracks    : {len(reps)}")
    return stratified_sample(reps, args.target, args.seed,
                             keys=("kind", "camera", "vehicle_class"))


def source_gate(root: Path, args: argparse.Namespace) -> list[dict]:
    rows = collect_gate_crops(root, args.max_ratio)
    print(f"[sample] gate-face crops  : {len(rows)}  (w/h <= {args.max_ratio})")
    reps = pick_representatives(rows, prefer=valid_label_stems(root))
    print(f"[sample] unique tracks    : {len(reps)}")
    return stratified_sample(reps, args.target, args.seed, keys=("camera", "kind"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deduplicate and subsample the vehicle-crop pool for labelling.")
    parser.add_argument("--dataset", type=Path,
                        help="Dataset root (defaults to config).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source", choices=("gate", "attributes"), default="gate",
                        help="gate = front/rear plate-face views (recommended); "
                             "attributes = the general queue (mostly sideways).")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help="Approximate number of crops to keep.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-ratio", type=float, default=DEFAULT_MAX_RATIO,
                        help="Max width/height for a gate crop to count as plate-face.")
    parser.add_argument("--kinds", type=str, default="",
                        help="(attributes source) comma-separated kinds, e.g. entry,exit.")
    parser.add_argument("--output", type=Path,
                        help="Sample manifest path (defaults to attributes/sample_manifest.jsonl).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, glob = load_config(args.config)
    root = args.dataset or Path(glob.output_dir)

    if args.source == "gate":
        sample = source_gate(root, args)
    else:
        sample = source_attributes(root, args)
    if not sample:
        return

    out = args.output or (root / "attributes" / "sample_manifest.jsonl")
    written = write_manifest(out, sample)
    print(f"[sample] sampled crops    : {written}  (source={args.source}, seed={args.seed}, target={args.target})")
    print(f"[sample] written to       : {out}")


if __name__ == "__main__":
    main()
