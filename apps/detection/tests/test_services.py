"""Tests for the detection service layer."""
from __future__ import annotations

from apps.detection.services import load_model


class TestLoadModel:
    """Tests for the load_model function."""

    def test_load_model_returns_none_in_dev(self) -> None:
        """In development, load_model should return None (stub)."""
        result = load_model()
        assert result is None

    def test_load_model_accepts_custom_path(self) -> None:
        """load_model should accept a custom model path argument."""
        result = load_model(model_path="custom_model.pt")
        assert result is None
