"""Build a labelling queue from a signal that is independent of the classifier.

    uv run python tools/queue_candidates.py \
        --dataset /mnt/nina/SmartParking \
        --target furgon --kinds entry --limit 300 \
        --output "/mnt/nina/SmartParking/car type/queue_furgon.jsonl"

Why this exists: the type model answers "furgon" for 2 of every 17 furgones it
sees. Ranking candidates by the model's own prediction would therefore surface
the two it already gets right and bury the fifteen it does not -- the model
would pick its own exam. The queue is built from the COCO detector class
instead, which never sees the body type, plus the trajectory label that says the
vehicle entered the premises.

What is deliberately NOT used: a geometric prior. Measured on the labelled pool,
the box aspect ratio of furgones (median 0.71) sits *below* that of cars (0.77)
and station wagons (0.75), with fully overlapping quartiles -- at the gate the
van is seen from behind, so it is wide, not tall. Adding it would rank noise.

Every row carries ``reason`` so a reviewer can audit later why a crop was
queued, and ``--exclude-labelled`` drops crops already in the pool. That
exclusion matches on frame number, which the pool's crop names preserve but the
video name does not, so a frame number seen in two different videos would drop
both: the tool prints how many it skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.attributes import image_path, iter_manifest, pick_representatives  # noqa: E402

# COCO indices the detector emits, mapped to the body type worth queueing for.
# Only furgon is justified today: it is the one class below the learning floor
# (59 tracks) whose operational value is clear.
TARGET_CLASSES: dict[str, tuple[int, ...]] = {
    "furgon": (7,),   # truck -- the superset a van falls into
    "minibus": (5,),  # bus
    "moto": (3,),
}

_FRAME_RE = re.compile(r"frame_(\d+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue labelling candidates.")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset root holding attributes/manifest.jsonl and the crops.")
    parser.add_argument("--manifest", type=Path,
                        help="Override the manifest path (default: <dataset>/attributes/manifest.jsonl).")
    parser.add_argument("--target", choices=sorted(TARGET_CLASSES),
                        help="Named target; sets the detector classes to keep.")
    parser.add_argument("--classes",
                        help="Explicit COCO class indices, e.g. '2,7'. Overrides --target.")
    parser.add_argument("--videos", default="",
                        help="Only keep rows whose video contains one of these "
                             "comma-separated substrings, e.g. 'XVR_ch6'.")
    parser.add_argument("--kinds", default="entry",
                        help="Comma-separated trajectory kinds to keep (entry, exit, pass).")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialise", type=Path,
                        help="Copy the selected crops into this folder, for the labeller.")
    parser.add_argument("--exclude-labelled", type=Path,
                        help="Pool labels.jsonl; crops whose frame already appears there are skipped.")
    return parser


def labelled_frames(labels_path: Path) -> set[str]:
    """Frame numbers already present in the pool, from its labels.jsonl."""
    frames: set[str] = set()
    for row in iter_manifest(labels_path):
        match = _FRAME_RE.search(str(row.get("frame", "")))
        if match:
            frames.add(match.group(1))
    return frames


def resolve_classes(target: str | None, classes: str | None) -> tuple[set[int], str]:
    """Detector classes to keep, plus the label used in the audit reason."""
    if classes:
        return {int(value) for value in classes.split(",") if value.strip()}, "custom"
    if target:
        return set(TARGET_CLASSES[target]), target
    raise SystemExit("[queue] one of --target or --classes is required")


def select(rows: list[dict], wanted: set[int], kinds: set[str]) -> list[dict]:
    """Detector class and trajectory kind only: the signals that never see type."""
    return [row for row in rows
            if row.get("vehicle_class") in wanted and row.get("kind") in kinds]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = args.manifest or args.dataset / "attributes" / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"[queue] manifest not found: {manifest}")

    wanted, label = resolve_classes(args.target, args.classes)
    needles = [value.strip() for value in args.videos.split(",") if value.strip()]

    total = 0
    in_class = 0
    kept: list[dict] = []
    for row in iter_manifest(manifest):
        total += 1
        if row.get("vehicle_class") in wanted:
            in_class += 1
            kept.append(row)

    if needles:
        kept = [row for row in kept
                if any(needle in str(row.get("video", "")) for needle in needles)]
    kinds = {kind.strip() for kind in args.kinds.split(",") if kind.strip()}
    selected = select(kept, wanted, kinds)
    # One crop per vehicle: the same track queues several near-duplicate frames.
    representatives = pick_representatives(selected)

    skipped_labelled = 0
    if args.exclude_labelled and args.exclude_labelled.exists():
        seen = labelled_frames(args.exclude_labelled)
        before = len(representatives)
        representatives = [
            row for row in representatives
            if str(row.get("frame")) not in seen
        ]
        skipped_labelled = before - len(representatives)

    representatives.sort(key=lambda row: str(row.get("stem")))
    chosen = representatives[: args.limit]

    def resolved(row: dict) -> str:
        """Crop path relative to the dataset root, so the labeller can open it."""
        source = image_path(args.dataset, row)
        try:
            return str(source.relative_to(args.dataset))
        except ValueError:
            return str(source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in chosen:
            fh.write(json.dumps({
                "stem": row.get("stem"),
                "image": resolved(row),
                "video": row.get("video"),
                "track_id": row.get("track_id"),
                "frame": row.get("frame"),
                "kind": row.get("kind"),
                "vehicle_class": row.get("vehicle_class"),
                "crop_hw": row.get("crop_hw"),
                "reason": (f"detector_class={row.get('vehicle_class')}({label}); "
                           f"kind={row.get('kind')}; representative crop of the track"),
            }) + "\n")

    if args.materialise:
        args.materialise.mkdir(parents=True, exist_ok=True)
        for row in chosen:
            source = args.dataset / resolved(row)
            if source.exists():
                shutil.copy2(source, args.materialise / f"{row['stem']}.jpg")

    print(f"[queue] manifest rows {total}, class {sorted(wanted)} "
          f"({in_class} rows{', videos ' + str(needles) if needles else ''})")
    print(f"[queue] kind in {sorted(kinds)}: {len(selected)} -> {len(representatives)} "
          f"one per track ({skipped_labelled} already labelled)")
    print(f"[queue] selected {len(chosen)} (limit {args.limit}) -> {args.output}")
    if len(representatives) > args.limit:
        print(f"[queue] note: {len(representatives) - args.limit} candidates left out "
              "by --limit; raise it to queue all")


if __name__ == "__main__":
    main()
