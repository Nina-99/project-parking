"""Tests for parking app Celery tasks."""
from __future__ import annotations

from apps.parking.tasks import sample_parking_task


class TestSampleParkingTask:
    """Tests for the sample parking Celery task."""

    def test_task_returns_expected_string(self) -> None:
        """The sample task should return the expected placeholder string."""
        result = sample_parking_task()
        assert result == "Parking task executed"
