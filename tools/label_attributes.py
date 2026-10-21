"""Interactive labeller for vehicle body type and predominant colour.

    uv run python tools/label_attributes.py --dataset /mnt/nina/Smart_Parking/dataset/

Walks the crops the extractor queued in ``attributes/`` and records a body type
and a colour for each one, two keystrokes per sample. Labels are appended to
``attributes/labels.jsonl``, so the tool is resumable: rerun it and it only
asks for what is still pending.

These attributes are labelled by hand on purpose. Pseudo-labelling them (HSV
thresholds, say) would train the classifier on the output of the very rule it
is later compared against, and the comparison would no longer mean anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.attributes import load_labelled  # noqa: E402
from plate_dataset.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from plate_dataset.labelling import AttributeLabeller  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label vehicle body type and colour for the attribute dataset.")
    parser.add_argument("--dataset", type=Path,
                        help="Dataset root written by the extractor (defaults to config).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sample", action="store_true",
                        help="Label only the deduplicated sample_manifest.jsonl.")
    parser.add_argument("--stats", action="store_true",
                        help="Print queue counts and exit without opening a window.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, glob = load_config(args.config)

    root = args.dataset or Path(glob.output_dir)
    if not (root / "attributes" / "manifest.jsonl").exists():
        print(f"[ERROR] no attribute queue at {root / 'attributes' / 'manifest.jsonl'}")
        print("        run the extractor first (tools/extract_dataset.py).")
        return

    manifest_path = root / "attributes" / "manifest.jsonl"
    if args.sample:
        sample_path = root / "attributes" / "sample_manifest.jsonl"
        if not sample_path.exists():
            print(f"[ERROR] no sample queue at {sample_path}")
            print("        run tools/sample_attributes.py first.")
            return
        manifest_path = sample_path

    labeller = AttributeLabeller(root, glob.vehicle_types, glob.vehicle_colors,
                                 manifest_path=manifest_path)

    if args.stats:
        labelled = load_labelled(labeller.labels_path)
        pending = labeller.pending()
        print(f"dataset : {root}")
        print(f"types   : {', '.join(glob.vehicle_types)}")
        print(f"colours : {', '.join(glob.vehicle_colors)}")
        print(f"labelled: {len(labelled)}")
        print(f"pending : {len(pending)}")
        return

    labeller.run()


if __name__ == "__main__":
    main()
