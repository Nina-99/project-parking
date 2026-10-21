"""
Tests for Celery autodiscovery and app wiring.

Verifies that the Celery app loads and detects tasks from
the detection app.
"""

from __future__ import annotations

from config.celery import app


class TestCeleryAutodiscovery:
    """Tests for Celery app wiring."""

    def test_celery_app_has_detection_task(self) -> None:
        """The Celery app should have registered the detection sample task."""
        registered_tasks = app.tasks
        task_name = "apps.detection.tasks.sample_detection_task"
        assert task_name in registered_tasks, (
            f"Expected {task_name} to be registered in Celery. "
            f"Registered tasks include: {list(registered_tasks.keys())[:10]}"
        )
