"""Plate dataset extractor.

Two-pass, zone-based extraction of license-plate samples from recorded
parking-entrance videos.

The extractor separates vehicles that enter the premises from vehicles that
only pass along the street by reasoning over the *trajectory* of each tracked
vehicle (direction, scale growth, zone crossing) instead of a single
positional rectangle. See ``pipeline.Extractor`` for the orchestration.
"""

from plate_dataset.config import CameraConfig, GlobalConfig, load_config
from plate_dataset.pipeline import Extractor

__all__ = ["CameraConfig", "GlobalConfig", "load_config", "Extractor"]
