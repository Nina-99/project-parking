"""The worker engine must preprocess exactly the way the trainers evaluate.

A mismatch here costs accuracy silently: no exception, no log line, just a model
fed slightly different pixels than the ones it was validated on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from _train_common import build_transforms  # noqa: E402

from backend.worker.engine import DetectionEngine, class_names, eval_transform  # noqa: E402

TYPE_CKPT = Path("/mnt/nina/SmartParking/runs/resnet18_final_type_v2_best.pt")
COLOR_CKPT = Path("/mnt/nina/SmartParking/checkpoints/resnet18_color_v3_best.pt")
YOLO = Path("yolo26s.pt")

needs_models = pytest.mark.skipif(
    not (TYPE_CKPT.exists() and COLOR_CKPT.exists() and YOLO.exists()),
    reason="trained checkpoints or detector not present on this machine",
)


def _crop(seed: int = 0, size: int = 320) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (size, size + 40, 3), dtype=np.uint8)
    return image


def test_preprocessing_is_identical_to_the_trainers() -> None:
    crop = _crop()
    rgb = Image.fromarray(crop[:, :, ::-1])

    ours = eval_transform(224)(rgb)
    _, theirs = build_transforms(224, task="color")

    assert torch.equal(ours, theirs(rgb)), "engine and trainer disagree on pixels"


def test_class_names_come_from_the_checkpoints_own_metrics() -> None:
    from eval_type_classifier import metrics_classes

    if not TYPE_CKPT.exists():
        pytest.skip("checkpoint not present")
    assert class_names(TYPE_CKPT) == metrics_classes(TYPE_CKPT)


def test_a_checkpoint_without_metrics_has_no_class_order(tmp_path: Path) -> None:
    orphan = tmp_path / "resnet18_type_best.pt"
    orphan.write_bytes(b"")

    assert class_names(orphan) is None


def test_the_engine_refuses_to_guess_a_class_order(tmp_path: Path) -> None:
    orphan = tmp_path / "resnet18_type_best.pt"
    orphan.write_bytes(b"")
    engine = DetectionEngine.__new__(DetectionEngine)
    engine.device = "cpu"

    with pytest.raises(RuntimeError, match="class order is unknown"):
        engine._load(orphan)


REAL_CROPS = Path("/mnt/nina/SmartParking/type dataset v2/test/images")


@needs_models
@pytest.mark.skipif(not REAL_CROPS.is_dir(), reason="frozen dataset not present")
def test_a_real_crop_gets_a_type_and_a_colour() -> None:
    real = next(iter(sorted(REAL_CROPS.glob("*.jpg"))))
    frame = cv2.imread(str(real))
    engine = DetectionEngine(
        yolo_path=YOLO, type_checkpoint=TYPE_CKPT, color_checkpoint=COLOR_CKPT,
        confidence=0.3,
    )

    detections = engine.process(frame)

    assert detections, f"the detector found no vehicle in {real.name}"
    first = detections[0]
    assert first.vehicle_type in engine.type_classes
    assert first.color in engine.color_classes
    assert 0.0 < first.vehicle_type_confidence <= 1.0
    assert 0.0 < first.color_confidence <= 1.0
