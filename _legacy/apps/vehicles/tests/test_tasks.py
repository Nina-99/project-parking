"""Tests for vehicles app Celery tasks."""

from __future__ import annotations

from apps.vehicles.tasks import sample_vehicle_task


class TestSampleVehicleTask:
    """Tests for the sample vehicle Celery task."""

    def test_task_returns_expected_string(self) -> None:
        """The sample task should return the expected placeholder string."""
        result = sample_vehicle_task()
        assert result == "Vehicle task executed"
