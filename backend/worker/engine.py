"""Vehicle detection and attribute classification, for the worker to call.

One process loads the models once and reuses them: YOLO finds the vehicle, the
two classifiers name its body type and its paint colour. Both classifiers were
trained on crops produced by that same detector, so the boxes are not rescaled or
re-framed on the way in.

Preprocessing is not "close enough" here. Training evaluates with
``transforms.Resize((224, 224))`` and everything else -- the interpolation, the
channel order, the normalisation constants -- has to match, or the model quietly
loses accuracy with no error anywhere. ``test_engine.py`` asserts this module and
``tools/_train_common.build_transforms`` produce bit-identical tensors.

Class names come from the checkpoint's own ``*_metrics.json``, never from a
hardcoded list: the order is whatever the run that produced the weights used, and
guessing it swaps labels without failing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# COCO indices the detector emits, as vehicle classes.
VEHICLE_CLASSES = (2, 3, 5, 7)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMG_SIZE = 224


@dataclass(frozen=True)
class Detection:
    """One vehicle in a frame, with whatever the classifiers could say about it."""

    box: tuple[int, int, int, int]
    vehicle_class: int
    detection_confidence: float
    vehicle_type: str | None = None
    vehicle_type_confidence: float | None = None
    color: str | None = None
    color_confidence: float | None = None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def build_classifier(backbone: str, num_classes: int) -> nn.Module:
    """The same factory the trainers use, so a checkpoint always loads."""
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone {backbone!r}")
    return model


def class_names(checkpoint: Path) -> list[str] | None:
    """Class order from the metrics file that belongs to this checkpoint.

    The prefix match comes first on purpose: with a type and a colour checkpoint
    sharing a folder, a blind glob can return the other model's order.
    """
    stem = checkpoint.stem
    prefix = stem[:-5] if stem.endswith("_best") else stem
    candidates = [checkpoint.parent / f"{prefix}_metrics.json"]
    candidates += sorted(checkpoint.parent.glob("*_metrics.json"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("classes"):
            return list(data["classes"])
    return None


def eval_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """The training-time *evaluation* pipeline. Must not drift from the trainers."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class DetectionEngine:
    """YOLO plus the two attribute classifiers, loaded once."""

    _instance: DetectionEngine | None = None

    def __init__(
        self,
        yolo_path: Path,
        type_checkpoint: Path,
        color_checkpoint: Path,
        device: str | None = None,
        confidence: float = 0.5,
        img_size: int = IMG_SIZE,
    ) -> None:
        from ultralytics import YOLO  # imported here: the app must not need it to boot

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence = confidence
        self.img_size = img_size
        self.transform = eval_transform(img_size)
        self.detector = YOLO(str(yolo_path))
        self.type_model, self.type_classes = self._load(type_checkpoint)
        self.color_model, self.color_classes = self._load(color_checkpoint)

    def _load(self, checkpoint: Path) -> tuple[nn.Module, list[str]]:
        names = class_names(checkpoint)
        if names is None:
            raise RuntimeError(
                f"{checkpoint} has no matching *_metrics.json, so its class order is "
                "unknown. Refusing to guess: a swapped order mislabels every vehicle "
                "without raising anything.")
        metrics_path = checkpoint.parent / f"{checkpoint.stem.replace('_best', '')}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        backbone = metrics.get("backbone", "resnet18")
        model = build_classifier(backbone, len(names))
        model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        model.to(self.device).eval()
        return model, names

    @torch.no_grad()
    def classify(self, crop: np.ndarray) -> tuple[str | None, float, str | None, float]:
        """BGR crop -> (type, type confidence, colour, colour confidence)."""
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        type_conf, type_index = self.type_model(tensor).softmax(1).max(1)
        color_conf, color_index = self.color_model(tensor).softmax(1).max(1)
        return (
            self.type_classes[int(type_index)],
            float(type_conf),
            self.color_classes[int(color_index)],
            float(color_conf),
        )

    def process(self, frame: np.ndarray) -> list[Detection]:
        """Every vehicle in the frame, largest first, attributes included."""
        result = self.detector.predict(
            frame, verbose=False, device=self.device, conf=self.confidence,
            classes=list(VEHICLE_CLASSES), imgsz=640,
        )[0]
        height, width = frame.shape[:2]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for xyxy, class_id, score in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            strict=True,
        ):
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            vehicle_type, type_conf, color, color_conf = self.classify(frame[y1:y2, x1:x2])
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                vehicle_class=int(class_id),
                detection_confidence=float(score),
                vehicle_type=vehicle_type,
                vehicle_type_confidence=type_conf,
                color=color,
                color_confidence=color_conf,
            ))
        detections.sort(key=lambda det: det.area, reverse=True)
        return detections

    @classmethod
    def get_instance(cls) -> DetectionEngine:
        """The worker's singleton: loading three models per frame is not an option."""
        if cls._instance is None:
            from backend.app.core.config import settings

            cls._instance = cls(
                yolo_path=Path(settings.YOLO_MODEL_PATH),
                type_checkpoint=Path(settings.TYPE_MODEL_PATH),
                color_checkpoint=Path(settings.COLOR_MODEL_PATH),
                confidence=settings.YOLO_CONFIDENCE,
            )
        return cls._instance
