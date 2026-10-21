"""Generate synthetic Bolivian plate crops.

    uv run python tools/generate_synth_plates.py --output /tmp/opencode/synth_demo --count 20
    uv run python tools/generate_synth_plates.py --dataset /home/nina/Smart_Parking_dataset --count 2000

Synthetic rows go to ``plates/{images,labels}/synth/`` with ``synth: true`` in
the manifest: same layout as the extractor, always traceable. Train on
real+synthetic, evaluate on real only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from plate_dataset.synth_plates.pipeline import SynthConfig, SynthPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render synthetic Bolivian plate crops for OCR training.")
    parser.add_argument("--output", type=Path,
                        help="Dataset root to write into (defaults to config).")
    parser.add_argument("--dataset", type=Path,
                        help="Alias for --output.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-index", type=int, default=0,
                        help="First stem number (resume an interrupted batch).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, glob = load_config(args.config)
    root = args.output or args.dataset or Path(glob.output_dir)
    written = SynthPipeline(
        root,
        SynthConfig(count=args.count, seed=args.seed, start_index=args.start_index),
    ).run()
    print(f"[synth] {written} plate(s) written to {root / 'plates' / 'images' / 'synth'}")


if __name__ == "__main__":
    main()
