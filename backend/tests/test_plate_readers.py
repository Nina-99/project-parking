"""Both OCR adapters behind one port, and the rules they share."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from backend.adapters.outbound.ml.plate_crnn import (
    ALPHABET,
    BLANK_IDX,
    NUM_CLASSES,
    CrnnPlateReader,
    decode,
    post_process_bolivian_plate,
    preprocess,
)
from backend.adapters.outbound.ml.plate_paddle import PaddlePlateReader, is_plate_candidate
from backend.adapters.outbound.ml.plate_reader import build_plate_reader
from backend.app.core.config import settings
from backend.domain.matching import looks_like_plate

CHECKPOINT = Path(settings.OCR_CRNN_CHECKPOINT)
EMPTY = np.zeros((0, 0, 3), dtype=np.uint8)


def test_a_plate_is_short_alphanumeric_and_mixes_digits_and_letters() -> None:
    assert looks_like_plate("1234ABC")
    assert looks_like_plate("1234-abc")
    assert not looks_like_plate("TOYOTA"), "all letters is brand lettering"
    assert not looks_like_plate("123")
    assert not looks_like_plate("")
    assert not looks_like_plate(None)


def test_the_plate_reader_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OCR_ENABLED", False)

    assert build_plate_reader() is None


def test_paddle_reader_rejects_an_empty_crop_without_loading_the_engine() -> None:
    assert PaddlePlateReader().read(EMPTY) is None


def test_plate_shaped_boxes_pass_and_others_do_not() -> None:
    # 200x120 crop, plausible plate: wide, low, big enough.
    assert is_plate_candidate((10, 80, 110, 105), 120, 0.35, 1.0, 40)
    # Tall box: a window or a sign, not a plate.
    assert not is_plate_candidate((10, 10, 40, 100), 120, 0.35, 1.0, 40)
    # Too small.
    assert not is_plate_candidate((10, 80, 30, 88), 120, 0.35, 1.0, 40)
    # Too high on the vehicle.
    assert not is_plate_candidate((10, 0, 110, 20), 120, 0.35, 1.0, 40)


def test_crnn_preprocessing_matches_the_training_shape() -> None:
    tensor = preprocess(np.zeros((90, 160, 3), dtype=np.uint8))

    assert tuple(tensor.shape) == (3, 48, 192)
    assert tensor.dtype == torch.float32


def test_ctc_decode_collapses_repeats_and_blanks() -> None:
    logits = torch.full((5, 1, NUM_CLASSES), -10.0)
    letter_a = ALPHABET.index("A") + 1
    letter_b = ALPHABET.index("B") + 1
    for timestep, index in enumerate([letter_a, letter_a, BLANK_IDX, letter_b, BLANK_IDX]):
        logits[timestep, 0, index] = 10.0

    text, confidence = decode(logits)

    assert text == "AB"
    assert confidence > 0.99


def test_the_bolivian_heuristic_fixes_a_digit_letter_confusion() -> None:
    assert post_process_bolivian_plate("I23ABC") == "123ABC"
    assert post_process_bolivian_plate("123ABC") == "123ABC"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="OCR checkpoint not available")
def test_the_thesis_checkpoint_loads_into_the_adapter() -> None:
    reader = CrnnPlateReader(CHECKPOINT, device="cpu")

    # A blank crop must not crash the model; whatever it returns has to clear
    # the confidence floor or be rejected.
    result = reader.read(np.zeros((120, 120, 3), dtype=np.uint8))

    assert result is None or result.confidence >= reader.min_confidence
