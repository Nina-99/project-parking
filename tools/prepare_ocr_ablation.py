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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from plate_dataset.synth_plates.ablation import collect_real_plates, split_by_video  # noqa: E402

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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, glob = load_config(args.config)
    root = args.dataset or Path(glob.output_dir)

    by_video = collect_real_plates(root)
    train_stems, test_stems = split_by_video(
        by_video, test_fraction=args.test_fraction, seed=args.seed
    )

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
                "videos_total": len(by_video),
                "train_real": len(train_stems),
                "test": len(test_stems),
                "train_synth": len(synth_stems),
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[ablation] videos={len(by_video)} "
        f"train_real={len(train_stems)} test={len(test_stems)} synth={len(synth_stems)}"
    )
    if not synth_stems:
        print("[ablation] no synthetic plates yet — run tools/generate_synth_plates.py first")


if __name__ == "__main__":
    main()
