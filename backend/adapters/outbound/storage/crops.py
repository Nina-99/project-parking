# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Store the crop that the incident search shows next to a candidate."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import cv2

from backend.adapters.inbound.worker.ports import Frame
from backend.domain.events import Box


class CropFileStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def save(self, frame: Frame, box: Box, camera_id: str) -> str | None:
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        directory = self._root / camera_id
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            path = directory / f"{stamp}_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(str(path), crop)
        except OSError:
            return None  # a missing thumbnail must not lose the row
        return str(path)

    def purge(self, older_than: datetime) -> int:
        """Delete crops older than a cutoff; returns how many files went away."""
        if not self._root.exists():
            return 0
        cutoff = older_than.timestamp()
        deleted = 0
        for path in self._root.rglob("*.jpg"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError:
                continue
        return deleted
