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
from backend.domain.vehicle import Plate

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IDX2CHAR = {index + 1: char for index, char in enumerate(ALPHABET)}
BLANK_IDX = 0
NUM_CLASSES = len(ALPHABET) + 1

IMG_HEIGHT = 48
IMG_WIDTH = 192
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


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
    """BGR crop -> the tensor the checkpoint was trained on."""
    rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (tensor - mean) / std


def decode(logits: torch.Tensor) -> tuple[str, float]:
    """Greedy CTC decode plus the mean probability of the emitted characters."""
    probabilities = logits.softmax(2)
    predictions = probabilities.argmax(2).permute(1, 0).cpu().numpy()
    scores = probabilities.max(2).values.permute(1, 0).cpu().numpy()

    characters: list[str] = []
    confidence: list[float] = []
    for row, row_scores in zip(predictions, scores, strict=True):
        previous = BLANK_IDX
        for index, score in zip(row, row_scores, strict=True):
            index = int(index)
            if index != BLANK_IDX and index != previous:
                characters.append(IDX2CHAR[index])
                confidence.append(float(score))
            previous = index

    text = "".join(characters)
    return text, (sum(confidence) / len(confidence) if confidence else 0.0)


def post_process_bolivian_plate(text: str) -> str:
    """Digit/letter confusion fixes for the [digits][3 letters] format."""
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
        apply_heuristics: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = min_confidence
        self.apply_heuristics = apply_heuristics
        self.model = ResNetCRNN()
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def read(self, vehicle_crop: Frame) -> Plate | None:
        if vehicle_crop.size == 0:
            return None
        tensor = preprocess(vehicle_crop).unsqueeze(0).to(self.device)
        text, confidence = decode(self.model(tensor))
        if self.apply_heuristics:
            text = post_process_bolivian_plate(text)
        if confidence < self.min_confidence:
            return None
        cleaned = normalize_plate(text)
        if not looks_like_plate(cleaned):
            return None
        return Plate(text=cleaned, confidence=confidence)
