"""
Detection service layer for YOLO model integration.

Provides a factory function for loading the Ultralytics YOLO model.
The actual model loading is deferred — the dev stub returns None.
"""
from __future__ import annotations

from typing import Any


def load_model(model_path: str = "yolo26n.pt") -> Any | None:
    """
    Load a YOLO model via Ultralytics.

    Args:
        model_path: Path or model name for the Ultralytics YOLO model.
                    Defaults to "yolo26n.pt" (YOLO26 nano).

    Returns:
        A loaded Ultralytics YOLO model instance, or None if the
        model could not be loaded (dev fallback).

    Note:
        This function is a stub in development. When the actual
        model is available, uncomment the import and use:
            from ultralytics import YOLO
            return YOLO(model_path)
    """
    # from ultralytics import YOLO
    # return YOLO(model_path)
    return None
