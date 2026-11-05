# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Event clips: a few seconds around a confirmed crossing.

The clip is evidence of *movement*, not live video. It exists so the operator can
see how the vehicle crossed, but it is written only when an event is confirmed:
there is no continuous stream leaving the machine.

Clips are files, not rows. The vehicle identifier is unique, so the path is
derivable and the database does not need a column that can disagree with the
filesystem. A missing clip is honest; the recorte remains as the fallback.
"""
from __future__ import annotations

import contextlib
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from backend.adapters.inbound.worker.ports import Frame

CLIP_SUBDIR = "clips"
MAX_WIDTH = 640


@dataclass(frozen=True)
class ClipFrame:
    at: datetime
    image: Frame


def prepare(frame: Frame, max_width: int = MAX_WIDTH) -> Frame:
    """Downscale before buffering: a clip is evidence, not a master recording."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame, (max_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


class ClipStore:
    """Writes WebM/VP8 clips under ``<media root>/clips``.

    VP8 was chosen because it opens with the OpenCV build available at
    deployment and plays in the browsers the panel targets. H.264 is not
    available in that build.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, vehicle_id: str) -> Path:
        return self._root / f"{vehicle_id}.webm"

    def exists(self, vehicle_id: str) -> bool:
        return self.path_for(vehicle_id).is_file()

    def save(self, vehicle_id: str, frames: list[ClipFrame]) -> str | None:
        """Encode and store a clip; None when there is nothing worth keeping."""
        if len(frames) < 2:
            return None

        height, width = frames[0].image.shape[:2]
        if width <= 0 or height <= 0:
            return None

        span = (frames[-1].at - frames[0].at).total_seconds()
        # The sampling loop runs at 2 fps, so writing at a fixed rate would play
        # the clip at the wrong speed. Derive it from the timestamps instead.
        fps = len(frames) / span if span > 0 else 2.0
        fps = min(max(fps, 1.0), 10.0)

        target = self.path_for(vehicle_id)
        temp = target.with_name(f".{uuid.uuid4().hex[:8]}.webm")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(temp),
                cv2.VideoWriter_fourcc(*"VP80"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                return None
            try:
                for item in frames:
                    if item.image.shape[:2] == (height, width):
                        writer.write(item.image)
            finally:
                writer.release()
            os.replace(temp, target)
        except (OSError, cv2.error):
            return None
        finally:
            with contextlib.suppress(OSError):
                if temp.exists():
                    temp.unlink()
        return str(target)

    def purge(self, older_than: datetime) -> int:
        if not self._root.exists():
            return 0
        cutoff = older_than.timestamp()
        deleted = 0
        for path in self._root.glob("*.webm"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError:
                continue
        return deleted


def resolve_clip_path(
    vehicle_id: str,
    root: Path | None = None,
    media_root: Path | None = None,
) -> Path | None:
    """A clip inside the media root, or None. Never a path from a caller."""
    base = (media_root or root or Path(".")).resolve()
    candidate = (base / CLIP_SUBDIR / f"{vehicle_id}.webm").resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        return None
    return candidate
