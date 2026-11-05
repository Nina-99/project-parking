# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""CRNN plate reader: the thesis model behind the same port.

The checkpoint is a ResNet18 + BiLSTM + CTC trained by
``tools/train_ocr_crnn.py``. The architecture and preprocessing are duplicated
from the trainer on purpose: importing a training script from the runtime would
couple the worker to ``tools/``. A test loads the real checkpoint, so any drift
fails loudly instead of silently mislabeling plates.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import torch
import torch.nn as nn
from torchvision import models

from backend.domain.matching import looks_like_plate, normalize_plate
from backend.domain.ports import Frame
from backend.domain.vehicle import Plate, PlateChar

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IDX2CHAR = {index + 1: char for index, char in enumerate(ALPHABET)}
BLANK_IDX = 0
NUM_CLASSES = len(ALPHABET) + 1

# Below this probability a "runner-up" is softmax dust, not an alternative the
# operator could reasonably pick. Filtering here keeps the UI honest.
ALTERNATIVE_FLOOR = 0.01

IMG_HEIGHT = 48
IMG_WIDTH = 192
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Same parameters as PlateOcrDataset in tools/train_ocr_crnn.py: clipLimit 2.0,
# 4x4 tiles, applied to the L channel of LAB.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


class ResNetCRNN(nn.Module):
    """The trainer's architecture, with the same stride surgery."""

    def __init__(self, num_classes: int = NUM_CLASSES, hidden_size: int = 512) -> None:
        super().__init__()
        resnet = models.resnet18(weights=None)

        # Keep the sequence width: halve height only in the deeper stages.
        resnet.layer2[0].conv1.stride = (2, 1)
        resnet.layer2[0].downsample[0].stride = (2, 1)
        resnet.layer3[0].conv1.stride = (2, 1)
        resnet.layer3[0].downsample[0].stride = (2, 1)
        resnet.layer4[0].conv1.stride = (2, 1)
        resnet.layer4[0].downsample[0].stride = (2, 1)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.bn_seq = nn.BatchNorm1d(512)
        self.bilstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.avgpool(features).squeeze(2)
        features = self.bn_seq(features).permute(0, 2, 1)
        recurrent, _ = self.bilstm(features)
        logits = self.classifier(self.dropout(recurrent))
        return cast(torch.Tensor, logits.permute(1, 0, 2))  # (W', B, C) for CTC


def preprocess(vehicle_crop: Frame) -> torch.Tensor:
    """BGR crop -> the tensor the checkpoint was trained on.

    The CLAHE pass is not optional. The trainer applies it to every image
    (``PlateOcrDataset.use_clahe``) because real plate crops are night captures:
    median brightness 77 vs 106 for synthetic, contrast 21 vs 34. A checkpoint
    trained with CLAHE and served without it reads the wrong distribution --
    measured on the frozen test set that is 14% character accuracy instead of
    67.6%. If this line is ever removed, the OCR numbers in the thesis stop
    describing what this adapter does.
    """
    rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (tensor - mean) / std


def decode(logits: torch.Tensor) -> tuple[str, float]:
    """Greedy CTC decode plus the mean probability of the emitted characters."""
    text, confidence, _ = decode_detailed(logits, top_k=1)
    return text, confidence


def decode_detailed(
    logits: torch.Tensor, top_k: int = 3
) -> tuple[str, float, tuple[PlateChar, ...]]:
    """Greedy CTC decode with each character's confidence and runners-up.

    The alternatives come from the same softmax the greedy path already
    computes, so this adds no second forward pass. ``top_k=1`` reproduces the
    old behaviour exactly and leaves the alternatives empty.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    probabilities = logits.softmax(2)
    k = min(top_k, probabilities.shape[2])
    top_values, top_indices = probabilities.topk(k, dim=2)
    # (W, B, k) -> (B, W, k): one row per batch item, as the old decode did.
    values = top_values.permute(1, 0, 2).cpu().numpy()
    indices = top_indices.permute(1, 0, 2).cpu().numpy()

    text_parts: list[str] = []
    scores: list[float] = []
    characters: list[PlateChar] = []
    for row_values, row_indices in zip(values, indices, strict=True):
        previous = BLANK_IDX
        for timestep in range(row_values.shape[0]):
            best = int(row_indices[timestep, 0])
            if best in (BLANK_IDX, previous):
                previous = best
                continue
            char = IDX2CHAR[best]
            score = float(row_values[timestep, 0])
            alternatives = tuple(
                (IDX2CHAR[int(index)], float(value))
                for value, index in zip(
                    row_values[timestep], row_indices[timestep], strict=True,
                )
                if int(index) not in (BLANK_IDX, best)
                and float(value) >= ALTERNATIVE_FLOOR
            )
            text_parts.append(char)
            scores.append(score)
            characters.append(
                PlateChar(char=char, confidence=score, alternatives=alternatives)
            )
            previous = best

    text = "".join(text_parts)
    mean = sum(scores) / len(scores) if scores else 0.0
    return text, mean, tuple(characters)


def post_process_bolivian_plate(text: str) -> str:
    """Digit/letter confusion fixes for the [digits][3 letters] format.

    Off by default: measured on the frozen test set it *lowers* character
    accuracy (67.0% vs 67.8% raw) because real plates carry letters inside the
    numeric block -- ``216OZAC`` is a valid plate, and rewriting that ``O`` to a
    zero turns a correct read into a wrong one.
    """
    if len(text) < 6:
        return text
    letters_part = text[-3:]
    digits_part = text[:-3]
    char_to_digit = {
        "O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
        "A": "4", "S": "5", "B": "8", "G": "6",
    }
    digit_to_char = {
        "0": "O", "1": "I", "2": "Z", "7": "Z", "5": "S",
        "8": "B", "6": "G", "4": "A",
    }
    digits = "".join(char_to_digit.get(char, char) for char in digits_part)
    letters = "".join(digit_to_char.get(char, char) for char in letters_part)
    return digits + letters


class CrnnPlateReader:
    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str | None = None,
        min_confidence: float = 0.60,
        apply_heuristics: bool = False,
        locator: object | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = min_confidence
        self.apply_heuristics = apply_heuristics
        # No locator means the caller already hands us a plate crop (the frozen
        # test set does). With a locator we accept a vehicle crop, as the worker
        # actually sends.
        self.locator = locator
        self.model = ResNetCRNN()
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def read(self, vehicle_crop: Frame) -> Plate | None:
        if vehicle_crop.size == 0:
            return None
        if self.locator is not None:
            from backend.adapters.outbound.ml.plate_locator import crop_plate

            box = self.locator.locate(vehicle_crop)
            if box is None:
                return None
            vehicle_crop = crop_plate(vehicle_crop, box)
        tensor = preprocess(vehicle_crop).unsqueeze(0).to(self.device)
        text, confidence, characters = decode_detailed(self.model(tensor))
        if self.apply_heuristics:
            corrected = post_process_bolivian_plate(text)
            # The heuristic rewrites characters, so the per-character read no
            # longer describes the string. Drop it rather than show a confidence
            # next to a character the model did not emit.
            characters = characters if corrected == text else ()
            text = corrected
        if confidence < self.min_confidence:
            return None
        cleaned = normalize_plate(text)
        if not looks_like_plate(cleaned):
            return None
        if len(cleaned) != len(characters):
            characters = ()
        return Plate(text=cleaned, confidence=confidence, characters=characters)
