"""Train the vehicle colour classifier with torchvision transfer learning.

    uv run python tools/train_color_classifier.py \
        --dataset "/mnt/nina/SmartParking/color dataset"

The label is the body colour of one vehicle crop, built by
``tools/build_color_dataset.py`` from the human-labelled pool. The dataset
layout is identical to the type one, which is why this entry point is a thin
wrapper: the difference that matters is the augmentation pipeline, and it is
selected by the task name, not by a flag.

Why a separate entry point at all: a colour model trained with the type pipeline
sees saturation jitter and learns to ignore saturation, which is the signal that
separates red from grey. Keeping the two trainers apart makes that mistake
impossible to reach by copying a command line.

Reported metrics ignore classes with fewer than ``MIN_TEST_SUPPORT`` test
samples -- a class with two crops has no measurable F1, and averaging it into the
headline turns noise into an apparent result. The full confusion matrix is still
printed so nothing is hidden.

Scope note: the minibus population is excluded from the colour dataset (see
``build_color_dataset.py``) because its paint is dominated by livery whose
dominant colour depends on the viewing angle, which makes the label
incomparable with a private car's base paint.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _train_common import main_for_task  # noqa: E402

if __name__ == "__main__":
    main_for_task("color")
