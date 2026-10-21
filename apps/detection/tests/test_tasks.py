"""Tests for detection app Celery tasks."""
from __future__ import annotations

from apps.detection.tasks import sample_detection_task


class TestSampleDetectionTask:
    """Tests for the sample detection Celery task."""

    def test_task_returns_expected_string(self) -> None:
        """The sample task should return the expected placeholder string."""
        result = sample_detection_task()
        assert result == "Detection task executed"
