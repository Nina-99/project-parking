# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Render a dataset's class composition as the table the thesis needs.

    uv run python tools/report_dataset.py \
        --dataset "/mnt/nina/SmartParking/type dataset v2" \
        --out "/mnt/nina/SmartParking/reports/type_v2.md"

Counts are read from the YOLO label files, one label per crop, so the table says
what the trainer will actually see rather than what the pool claimed. The test
column is the one to look at: it decides which classes can be reported, given
``MIN_TEST_SUPPORT``.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")
MIN_TEST_SUPPORT = 40


def count_split(root: Path, split: str, num_classes: int) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    label_dir = root / split / "labels"
    if not label_dir.is_dir():
        return counts
    for path in label_dir.glob("*.txt"):
        first = path.read_text(encoding="utf-8").split()
        if first and first[0].isdigit():
            index = int(first[0])
            if index < num_classes:
                counts[index] += 1
    return counts


def render(root: Path, title: str, min_support: int) -> str:
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = list(data["names"])
    counts = {split: count_split(root, split, len(names)) for split in SPLITS}

    totals = {index: sum(counts[s].get(index, 0) for s in SPLITS) for index in range(len(names))}
    grand = sum(totals.values())

    lines = [f"# {title}", "", f"Dataset: `{root}`", ""]
    lines += ["| clase | train | valid | test | total | % flota | test n≥40 |",
              "|---|---|---|---|---|---|---|"]
    for index, name in enumerate(names):
        test_n = counts["test"].get(index, 0)
        share = 100 * totals[index] / grand if grand else 0.0
        verdict = "sí" if test_n >= min_support else "**no**"
        lines.append(
            f"| {name} | {counts['train'].get(index, 0)} | {counts['valid'].get(index, 0)} | "
            f"{test_n} | {totals[index]} | {share:.1f}% | {verdict} |"
        )
    lines.append(f"| **total** | {sum(counts['train'].values())} | "
                 f"{sum(counts['valid'].values())} | {sum(counts['test'].values())} | "
                 f"{grand} | 100% | |")
    reportable = [names[i] for i in range(len(names))
                  if counts["test"].get(i, 0) >= min_support]
    lines += [
        "",
        f"Clases reportables (test n ≥ {min_support}): {len(reportable)} de {len(names)}"
        f" — {', '.join(reportable) if reportable else 'ninguna'}.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render dataset composition as Markdown.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--title", default="Composición del dataset")
    parser.add_argument("--min-test-support", type=int, default=MIN_TEST_SUPPORT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    text = render(args.dataset, args.title, args.min_test_support)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[dataset] {args.dataset} -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
