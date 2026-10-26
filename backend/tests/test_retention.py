# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Crops expire; the vehicle record does not."""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from backend.adapters.outbound.storage.crops import CropFileStore

NOW = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)


def _file(root: Path, name: str, age_days: float) -> Path:
    path = root / "gate-1" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jpeg")
    stamp = (NOW - timedelta(days=age_days)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def test_only_crops_older_than_the_cutoff_are_deleted(tmp_path: Path) -> None:
    old = _file(tmp_path, "old.jpg", age_days=10)
    fresh = _file(tmp_path, "fresh.jpg", age_days=1)

    deleted = CropFileStore(tmp_path).purge(NOW - timedelta(days=5))

    assert deleted == 1
    assert not old.exists()
    assert fresh.exists()


def test_a_missing_root_is_not_an_error(tmp_path: Path) -> None:
    assert CropFileStore(tmp_path / "nope").purge(NOW) == 0


def test_non_jpeg_files_are_left_alone(tmp_path: Path) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("keep me")
    os.utime(other, (0, 0))

    assert CropFileStore(tmp_path).purge(NOW) == 0
    assert other.exists()
