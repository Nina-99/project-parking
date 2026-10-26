# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Celery tasks. Importing this package registers every task with the app."""

from backend.worker.tasks.detection import ping_task, process_frame
from backend.worker.tasks.maintenance import purge_old_data

__all__ = ["ping_task", "process_frame", "purge_old_data"]
