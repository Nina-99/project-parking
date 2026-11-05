# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Both OCR training stages must run start to finish.

The stage-1 branch crashed on its first epoch because evaluate() had been changed
to return two scores (raw and heuristic) and only the stage-2 call site was
updated. Nothing exercised the other one, so a print statement was the only thing
between the pipeline and a crash. These tests run one epoch of each stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from train_ocr_crnn import main

STAGE1 = Path("/mnt/nina/SmartParking/runs")  # only to detect the warm cache below


def _plate_dataset(root: Path) -> Path:
    """Two plates, enough for a one-epoch run of either stage."""
    images = root / "plates" / "images" / "vid"
    labels = root / "plates" / "labels" / "vid"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    # Enough synthetic plates that the stage-1 hold-out is exercised too.
    plates = [("img_b", "5678XYZ")] + [
        (f"syn_{i:02d}", f"{1000 + i}ABC") for i in range(12)]
    for stem, text in plates:
        array = np.full((48, 192, 3), 200, dtype=np.uint8)
        Image.fromarray(array).save(images / f"{stem}.jpg")
        (labels / f"{stem}.txt").write_text(text, encoding="utf-8")

    ablation = root / "ablation"
    ablation.mkdir()
    (ablation / "train_synth.txt").write_text(
        "".join(f"syn_{i:02d}\n" for i in range(12)), encoding="utf-8")
    (ablation / "train_real.txt").write_text("img_b\n", encoding="utf-8")
    (ablation / "test.txt").write_text("img_b\n", encoding="utf-8")
    return root


def _run(root: Path, stage: str, extra: list[str]) -> Path:
    out = root / "out" / stage
    main(["--dataset", str(root), "--stage", stage, "--batch-size", "2",
          "--save-dir", str(out), *extra])
    return out


@pytest.mark.skipif(not Path("yolo26s.pt").exists(), reason="run from software/")
def test_stage1_runs_and_saves(tmp_path: Path) -> None:
    root = _plate_dataset(tmp_path / "ds")

    out = _run(root, "stage1_synth", ["--epochs-synth", "1"])

    assert (out / "crnn_stage1_synth.pt").exists()


@pytest.mark.skipif(not Path("yolo26s.pt").exists(), reason="run from software/")
def test_stage2_runs_and_writes_metrics(tmp_path: Path) -> None:
    root = _plate_dataset(tmp_path / "ds")
    dump = root / "test_predictions.jsonl"

    out = _run(root, "stage2_real", ["--epochs-real", "1",
                                     "--dump-test-predictions", str(dump)])

    assert (out / "crnn_stage2_real.pt").exists()
    metrics = json.loads((out / "final_metrics.json").read_text(encoding="utf-8"))
    assert "raw" in metrics or "exact_match" in json.dumps(metrics)

    # The dump is what bootstrap_ci.py consumes; without the cluster key the
    # interval would resample crops instead of plates.
    rows = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {"stem", "plate", "truth", "pred"} <= rows[0].keys()
    assert all(row["plate"] == row["truth"] for row in rows)

    # Re-evaluating an existing checkpoint must not require retraining.
    eval_dump = root / "eval_predictions.jsonl"
    main(["--dataset", str(root), "--stage", "stage2_real", "--eval-only",
          "--save-dir", str(out), "--dump-test-predictions", str(eval_dump)])
    assert eval_dump.exists()
    assert len(eval_dump.read_text(encoding="utf-8").splitlines()) == len(rows)


def test_a_fixed_schedule_keeps_the_learning_rate() -> None:
    """The LR must not be the hidden variable when comparing runs."""
    import torch
    from train_ocr_crnn import build_scheduler, current_lr

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    assert build_scheduler(optimizer, "fixed", patience=5) is None
    assert current_lr(optimizer) == 1e-3
    plateau = build_scheduler(optimizer, "plateau", patience=5)
    for _ in range(12):
        plateau.step(1.0)  # a loss that never improves

    assert current_lr(optimizer) < 1e-3, "plateau should have decayed the rate"
