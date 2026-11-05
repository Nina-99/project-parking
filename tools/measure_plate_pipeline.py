# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""End-to-end measurement of the plate pipeline on hand-typed vehicle crops.

    uv run python tools/measure_plate_pipeline.py \
        --text /home/nina/FNI2025/ProjectParking/plates_det/text.jsonl \
        --detector /home/nina/FNI2025/ProjectParking/plates_det/runs/plate_det/weights/best.pt \
        --crnn /home/nina/FNI2025/ProjectParking/runs/ocr_v2/crnn_stage2_real.pt

Four numbers, because they fail for different reasons and a thesis needs to say
which one broke:

  A  detector recall      -- did it find a plate at all?
  B  end-to-end           -- detector box -> CRNN, the system as deployed
  C  CRNN on the human box -- the recognition ceiling, localization removed
  D  Paddle end-to-end     -- the operational baseline on identical inputs

B vs C is the cost of localization error. B vs D is whether our model is worth
the trouble. C vs the frozen plate-crop test (67.6%) is whether vehicle crops
are harder than the crop dataset suggested.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.synth_plates.ablation import score_pairs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--crnn", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the markdown table here as well as stdout.")
    parser.add_argument("--paddle", action="store_true",
                        help="Also run PaddleOCR end-to-end (slow: ~1s/crop).")
    parser.add_argument("--dump-predictions", type=Path, default=None,
                        help="Write per-crop predictions as JSONL (stem, plate, truth, "
                             "pred, pred_human_box). Feed the file to "
                             "bootstrap_ci.py items --task text --cluster-col plate.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    rows = [json.loads(line) for line in
            args.text.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("text")]
    if not rows:
        raise SystemExit(f"no typed texts in {args.text} yet")
    print(f"[measure] {len(rows)} crops with ground-truth text")

    from backend.adapters.outbound.ml.plate_crnn import CrnnPlateReader
    from backend.adapters.outbound.ml.plate_locator import PlateLocator, crop_plate

    device = "cuda"
    locator = PlateLocator(args.detector, device=device)
    reader = CrnnPlateReader(args.crnn, device=device)   # expects a plate crop

    paddle = None
    if args.paddle:
        from backend.adapters.outbound.ml.plate_paddle import PaddlePlateReader

        paddle = PaddlePlateReader(min_conf=0.60, lang="latin")

    detected = 0
    pairs_e2e: list[tuple[str, str]] = []
    pairs_human: list[tuple[str, str]] = []
    pairs_paddle: list[tuple[str, str]] = []
    failures: list[dict] = []
    dumped: list[dict] = []

    for row in rows:
        truth = row["text"]
        image = cv2.imread(row["image"])
        if image is None:
            continue

        box = locator.locate(image)
        if box is not None:
            detected += 1
            e2e = reader.read(crop_plate(image, box))
            got = e2e.text if e2e else ""
        else:
            got = ""

        x1, y1, x2, y2 = row["box"]
        human = reader.read(image[y1:y2, x1:x2])
        human_text = human.text if human else ""

        pairs_e2e.append((truth, got))
        pairs_human.append((truth, human_text))
        dumped.append({
            "stem": Path(row["image"]).stem,
            "plate": truth,
            "truth": truth,
            "pred": got,
            "pred_human_box": human_text,
            "detected": box is not None,
        })

        if paddle is not None:
            read = paddle.read(image)
            pairs_paddle.append((truth, read.text if read else ""))

        if got != truth and len(failures) < 12:
            failures.append({"truth": truth, "e2e": got, "human_box": human_text,
                             "detected": box is not None})

    def line(name: str, pairs: list[tuple[str, str]], n: int) -> str:
        if not pairs:
            return f"| {name} | — | — | — |"
        scores = score_pairs(pairs)
        return (f"| {name} | {scores.char_accuracy * 100:.1f}% | "
                f"{scores.exact_match * 100:.1f}% | {n} |")

    recall = detected / len(rows) * 100
    if rows and detected == 0:
        raise SystemExit(
            f"[measure] the detector found no plate in any of the {len(rows)} crops. "
            "This is not a 0% measurement: it is the wrong input. The locator expects "
            "vehicle crops with the plate inside, not tight plate crops."
        )
    lines = [
        f"Recall del detector: {detected}/{len(rows)} ({recall:.1f}%)",
        "",
        "| etapa | char acc | exact match | n |",
        "|---|---|---|---|",
        line("B. end-to-end (detector + CRNN)", pairs_e2e, len(pairs_e2e)),
        line("C. CRNN con caja humana", pairs_human, len(pairs_human)),
    ]
    if pairs_paddle:
        lines.append(line("D. PaddleOCR end-to-end", pairs_paddle, len(pairs_paddle)))

    table = "\n".join(lines) + "\n"
    print("\n" + table)

    if failures:
        print("Fallos de B (primeros 12):")
        for f in failures:
            print(f"  ref={f['truth']:9s} e2e={f['e2e']:10s} "
                  f"caja_humana={f['human_box']:10s} detectada={f['detected']}")

    report = {"n": len(rows), "detected": detected, "recall": recall,
              "e2e": dict(zip(("char_accuracy", "exact_match"),
                              (score_pairs(pairs_e2e).char_accuracy,
                               score_pairs(pairs_e2e).exact_match))),
              "human_box": dict(zip(("char_accuracy", "exact_match"),
                                    (score_pairs(pairs_human).char_accuracy,
                                     score_pairs(pairs_human).exact_match)))}
    if pairs_paddle:
        report["paddle"] = dict(zip(("char_accuracy", "exact_match"),
                                    (score_pairs(pairs_paddle).char_accuracy,
                                     score_pairs(pairs_paddle).exact_match)))
    if args.output:
        args.output.write_text(
            json.dumps(report, indent=4) + "\n\n" + table, encoding="utf-8")
        print(f"[measure] written to {args.output}")

    if args.dump_predictions:
        args.dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_predictions.open("w", encoding="utf-8") as fh:
            for row in dumped:
                fh.write(json.dumps(row) + "\n")
        print(f"[measure] wrote {len(dumped)} predictions to {args.dump_predictions}")


if __name__ == "__main__":
    main()
