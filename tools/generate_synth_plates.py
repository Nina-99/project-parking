# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Generate synthetic Bolivian plate crops.

    uv run python tools/generate_synth_plates.py --output /tmp/opencode/synth_demo --count 20
    uv run python tools/generate_synth_plates.py --dataset /home/nina/Smart_Parking_dataset --count 2000

Synthetic rows go to ``plates/{images,labels}/synth/`` with ``synth: true`` in
the manifest: same layout as the extractor, always traceable. Train on
real+synthetic, evaluate on real only.
"""

from __future__ import annotations

import argparse
import multiprocessing
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
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel processes over disjoint index ranges. Each "
                             "gets its own manifest shard so lines cannot interleave.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-index", type=int, default=0,
                        help="First stem number (resume an interrupted batch).")
    return parser


def _generate(job: tuple[Path, SynthConfig]) -> int:
    root, config = job
    return SynthPipeline(root, config).run()


def _plan(root: Path, args: argparse.Namespace) -> list[tuple[Path, SynthConfig]]:
    """Split the index range into one job per worker, each with its own manifest."""
    workers = max(1, args.workers)
    per_worker = -(-args.count // workers)  # ceil
    jobs: list[tuple[Path, SynthConfig]] = []
    for worker in range(workers):
        count = min(per_worker, args.count - worker * per_worker)
        if count <= 0:
            break
        jobs.append((root, SynthConfig(
            count=count,
            seed=args.seed,
            start_index=args.start_index + worker * per_worker,
            manifest_name=f"manifest_synth_{worker:02d}.jsonl" if workers > 1
            else "manifest.jsonl",
        )))
    return jobs


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, glob = load_config(args.config)
    root = args.output or args.dataset or Path(glob.output_dir)
    jobs = _plan(root, args)
    if len(jobs) == 1:
        written = _generate(jobs[0])
    else:
        with multiprocessing.Pool(len(jobs)) as pool:
            written = sum(pool.map(_generate, jobs))
    print(f"[synth] {written} plate(s) written to {root / 'plates' / 'images' / 'synth'}")


if __name__ == "__main__":
    main()
