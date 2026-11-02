# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Prepare the OCR ablation splits.

    uv run python tools/prepare_ocr_ablation.py --dataset ~/FNI2025/ProjectParking/Smart_Parking_dataset

Writes, next to the dataset root::

    ablation/train_real.txt    # real plates for the real-only run
    ablation/test.txt           # held-out REAL plates both runs are scored on
    ablation/train_synth.txt    # synthetic plates for the mixed run
    ablation/protocol.json      # video counts, seed, sizes — paste into the thesis

The split is by recording session with a fixed seed, so both training runs and
the thesis table share exactly the same test set. Re-running overwrites with
identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from plate_dataset.synth_plates.ablation import (  # noqa: E402
    collect_real_plates,
    collect_real_plates_by_text,
    split_by_plate,
    split_by_video,
)

TEST_FRACTION = 0.2
SEED = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write deterministic real/synth OCR ablation splits."
    )
    parser.add_argument("--dataset", type=Path, help="Dataset root (defaults to config).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--max-per-plate",
        type=int,
        default=0,
        help="Cap the crops one plate contributes to the test (0 = no cap). "
        "Only meaningful with --group-by plate.",
    )
    parser.add_argument(
        "--group-by",
        choices=("session", "plate"),
        default="session",
        help="Unit kept on one side of the split. 'session' keeps a recording "
        "block together; 'plate' keeps every crop of one plate together, which "
        "is what stops the test from asking the model to reread a plate it "
        "memorised. They cannot both hold: the same car crosses sessions, so "
        "the plate-session graph collapses into one component.",
    )
    return parser


def fingerprint(stems: list[str]) -> str:
    """Stable digest of the plate set, so a stale split is detectable."""
    joined = "\n".join(sorted(stems)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, glob = load_config(args.config)
    root = args.dataset or Path(glob.output_dir)

    if args.group_by == "plate":
        by_plate = collect_real_plates_by_text(root)
        train_stems, test_stems = split_by_plate(
            by_plate,
            test_fraction=args.test_fraction,
            seed=args.seed,
            max_per_plate=args.max_per_plate,
        )
        groups_total = len(by_plate)
    else:
        by_video = collect_real_plates(root)
        train_stems, test_stems = split_by_video(
            by_video, test_fraction=args.test_fraction, seed=args.seed
        )
        groups_total = len(by_video)
    real_stems = train_stems + test_stems

    synth_dir = root / "plates" / "images" / "synth"
    synth_stems = sorted(p.stem for p in synth_dir.glob("*.jpg")) if synth_dir.exists() else []

    out = root / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_real.txt").write_text("\n".join(train_stems) + "\n", encoding="utf-8")
    (out / "test.txt").write_text("\n".join(test_stems) + "\n", encoding="utf-8")
    (out / "train_synth.txt").write_text("\n".join(synth_stems) + "\n", encoding="utf-8")
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "test_fraction": args.test_fraction,
                "group_by": args.group_by,
                "groups_total": groups_total,
                "max_per_plate": args.max_per_plate,
                "train_real": len(train_stems),
                "test": len(test_stems),
                "train_synth": len(synth_stems),
                # Fingerprint of the inputs. Without it nothing tells you the
                # split is stale: the plates folder keeps growing as the
                # extractor runs, and a split made last week quietly trains on
                # two thirds of the plates that exist today.
                "real_plates_total": len(real_stems),
                "real_plates_fingerprint": fingerprint(real_stems),
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[ablation] groups={groups_total} (by {args.group_by}) real={len(real_stems)} "
        f"train_real={len(train_stems)} test={len(test_stems)} synth={len(synth_stems)}"
    )
    print(f"[ablation] fingerprint of the real plates: {fingerprint(real_stems)}")
    if not synth_stems:
        print("[ablation] no synthetic plates yet — run tools/generate_synth_plates.py first")


if __name__ == "__main__":
    main()
