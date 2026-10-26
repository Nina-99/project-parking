# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Train the vehicle body-type classifier with torchvision transfer learning.

    uv run python tools/train_type_classifier.py \
        --dataset "/mnt/nina/SmartParking/type dataset"

Consumes a YOLO *detection* export as a *classification* set: every crop holds
one main vehicle whose box covers most of the frame, so the label is the class
of the largest box. Nothing is re-annotated -- the same labels are read, just
interpreted as a class per image.

Why a classifier and not a detector: the deployment pipeline already crops the
vehicle with the COCO detector, so this model receives a crop and only has to
name its type. A detector trained on crops would learn a full-frame box, which
is useless on a real frame.

The class distribution is heavily skewed (vagoneta dominates at ~46%, moto is
~9%), so the run uses class weights and is judged by macro-F1 over the classes
with enough test support, never by raw accuracy -- a model that always answers
"vagoneta" would score high on accuracy and zero on macro-F1.

The colour counterpart is ``train_color_classifier.py``. Both share
``_train_common.py`` but never share an augmentation pipeline: saturation jitter
is harmless for silhouette and fatal for a colour label.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _train_common import main_for_task  # noqa: E402

if __name__ == "__main__":
    main_for_task("type")
