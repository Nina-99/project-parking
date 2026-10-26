# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Guard rails for the type/colour training pipelines and the metric summary.

Both tasks used to share one augmentation pipeline, saturation jitter included.
On a colour classifier that teaches the network to ignore saturation -- the
signal that separates red from grey. These tests make the separation explicit
and make the regression impossible to reintroduce silently.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _train_common import MEAN, STD, build_transforms, summarise
from PIL import Image
from torchvision import transforms


def _denormalise(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    return np.clip(arr, 0.0, 1.0)


def test_colour_pipeline_has_no_chroma_jitter() -> None:
    """Saturation and hue jitter must never reach the colour model."""
    train_tf, _ = build_transforms(224, task="color")
    jitters = [t for t in train_tf.transforms if isinstance(t, transforms.ColorJitter)]
    assert jitters, "colour pipeline should still vary exposure"
    for jitter in jitters:
        assert jitter.saturation is None, "saturation jitter destroys the colour label"
        assert jitter.hue is None, "hue jitter destroys the colour label"


def test_colour_pipeline_keeps_a_red_car_red() -> None:
    """A saturated red crop stays red through every random draw.

    The invariant is the share of red, not its absolute value: exposure jitter
    darkens the whole crop uniformly, which changes brightness without changing
    the colour. What must never happen is the red channel losing its dominance.
    """
    train_tf, _ = build_transforms(224, task="color")
    red = Image.new("RGB", (256, 256), (220, 20, 20))
    for seed in range(20):
        random_state = torch.get_rng_state()
        torch.manual_seed(seed)
        rng_seed = np.random.get_state()
        np.random.seed(seed)
        try:
            pixel = _denormalise(train_tf(red))
        finally:
            torch.set_rng_state(random_state)
            np.random.set_state(rng_seed)
        median = np.median(pixel.reshape(-1, 3), axis=0)
        share = float(median[0]) / float(median.sum())
        assert median[0] == median.max(), f"seed {seed}: red is no longer dominant ({median})"
        assert share > 0.6, f"seed {seed}: red share collapsed to {share:.2f}"
        assert median[1] < 0.2 and median[2] < 0.2, f"seed {seed}: red shifted to {median}"


def test_type_pipeline_still_jitters_colour() -> None:
    """The type model is judged on shape, so photometric jitter is allowed there."""
    train_tf, _ = build_transforms(224, task="type")
    jitters = [t for t in train_tf.transforms if isinstance(t, transforms.ColorJitter)]
    assert any(j.saturation is not None for j in jitters)


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        build_transforms(224, task="colour")


def test_summarise_excludes_classes_below_minimum_support() -> None:
    names = ["a", "b", "c", "d"]
    matrix = np.array([
        [90, 10, 0, 0],
        [5, 45, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ])
    summary = summarise(matrix, names, min_support=40)

    assert summary["support"] == {"a": 100, "b": 50, "c": 2, "d": 0}
    assert summary["evaluated"] == {"a": True, "b": True, "c": False, "d": False}
    assert summary["excluded_classes"] == ["c", "d"]
    assert summary["macro_f1_evaluated"] == pytest.approx((0.9231 + 0.8571) / 2, abs=1e-3)
    assert summary["f1_weighted_real"] == pytest.approx((0.9231 * 100 + 0.8571 * 50) / 150, abs=1e-3)


def test_summarise_ignores_a_perfect_tiny_class_in_the_legacy_average() -> None:
    """The whole point: 2 samples scored perfectly must not lift the headline.

    Class ``c`` is classified perfectly, so it contributes F1 = 1.0 to the
    legacy average over present classes -- inflating the number while carrying
    no evidence. The evaluated average must not see it.
    """
    names = ["a", "b", "c", "d"]
    matrix = np.array([
        [90, 10, 0, 0],
        [5, 45, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ])
    summary = summarise(matrix, names, min_support=40)

    assert summary["macro_f1_present"] > summary["macro_f1_evaluated"]
    assert summary["evaluated_support"] == 150
    assert summary["ci95"]["c"] > 0.3, "a 2-sample class cannot have a tight interval"
