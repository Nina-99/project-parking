"""Tests for users app Celery tasks."""

from __future__ import annotations

from apps.users.tasks import sample_user_task


class TestSampleUserTask:
    """Tests for the sample user Celery task."""

    def test_task_returns_expected_string(self) -> None:
        """The sample task should return the expected placeholder string."""
        result = sample_user_task()
        assert result == "User task executed"
