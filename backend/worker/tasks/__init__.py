"""Celery tasks. Importing this package registers every task with the app."""

from backend.worker.tasks.detection import ping_task, process_frame

__all__ = ["ping_task", "process_frame"]
