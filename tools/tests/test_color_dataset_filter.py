# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The colour dataset must drop excluded vehicle types at build time.

Minibuses are excluded from the colour dataset because their livery makes the
colour label view-dependent. If that filter regresses, the label silently
becomes meaningless again, so it is worth a real run over a tiny pool rather
than a unit test of a helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from build_color_dataset import SPLITS, main
from PIL import Image


def _pool(root: Path) -> Path:
    """Three crops: two cars, one minibus."""
    pool = root / "pool"
    (pool / "images").mkdir(parents=True)
    (pool / "labels").mkdir()
    (pool / "labels_color").mkdir()
    (pool / "data.yaml").write_text(
        "nc: 2\nnames: ['automovil', 'minibus']\n", encoding="utf-8")
    (pool / "data_color.yaml").write_text(
        "nc: 2\nnames: ['blanco', 'rojo']\n", encoding="utf-8")
    crops = [("car_a_b00", 0, 0), ("car_b_b00", 0, 1), ("bus_c_b00", 1, 0)]
    for stem, type_idx, color_idx in crops:
        Image.new("RGB", (8, 8), (200, 200, 200)).save(pool / "images" / f"{stem}.jpg")
        (pool / "labels" / f"{stem}.txt").write_text(
            f"{type_idx} 0.5 0.5 1 1\n", encoding="utf-8")
        (pool / "labels_color" / f"{stem}.txt").write_text(
            f"{color_idx} 0.5 0.5 1 1\n", encoding="utf-8")
    return pool


def _built(root: Path, name: str, extra: list[str]) -> tuple[int, dict]:
    out = root / name
    main(["--pool", str(root / "pool"), "--output", str(out),
          "--min-count", "1", *extra])
    images = list(out.rglob("*.jpg"))
    return len(images), yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))


def test_excluded_type_is_dropped(tmp_path: Path) -> None:
    _pool(tmp_path)
    total, data = _built(tmp_path, "out", ["--exclude-types", "minibus"])
    assert total == 2
    assert data["nc"] == 2


def test_without_the_flag_nothing_is_dropped(tmp_path: Path) -> None:
    _pool(tmp_path)
    total, _ = _built(tmp_path, "out", [])
    assert total == 3


def test_unknown_type_fails_loudly(tmp_path: Path) -> None:
    _pool(tmp_path)
    with pytest.raises(SystemExit, match="unknown --exclude-types"):
        _built(tmp_path, "out", ["--exclude-types", "camion"])


def _add_crops(pool: Path, count: int, prefix: str, type_idx: int = 0,
               color_idx: int = 0) -> None:
    for index in range(count):
        stem = f"{prefix}_{index:02d}"
        Image.new("RGB", (8, 8), (200, 200, 200)).save(pool / "images" / f"{stem}.jpg")
        (pool / "labels" / f"{stem}.txt").write_text(
            f"{type_idx} 0.5 0.5 1 1\n", encoding="utf-8")
        (pool / "labels_color" / f"{stem}.txt").write_text(
            f"{color_idx} 0.5 0.5 1 1\n", encoding="utf-8")


def _splits(out: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in SPLITS:
        for path in (out / split / "images").glob("*.jpg"):
            result[path.stem] = split
    return result


def test_rebuilding_over_an_existing_dataset_is_refused(tmp_path: Path) -> None:
    _pool(tmp_path)
    _built(tmp_path, "out", ["--exclude-types", "minibus"])

    with pytest.raises(SystemExit, match="already holds a dataset"):
        main(["--pool", str(tmp_path / "pool"), "--output", str(tmp_path / "out"),
              "--min-count", "1"])


def test_force_replaces_instead_of_mixing(tmp_path: Path) -> None:
    _pool(tmp_path)
    _built(tmp_path, "out", [])
    (tmp_path / "out" / "train" / "images" / "stale_crop.jpg").write_bytes(b"")

    main(["--pool", str(tmp_path / "pool"), "--output", str(tmp_path / "out"),
          "--min-count", "1", "--force"])

    assert not (tmp_path / "out" / "train" / "images" / "stale_crop.jpg").exists()


def test_growing_the_pool_never_moves_an_existing_crop(tmp_path: Path) -> None:
    """The regression that matters: new labels must not resplit old data."""
    pool = _pool(tmp_path)
    _add_crops(pool, 20, "car")
    _built(tmp_path, "before", [])

    _add_crops(pool, 6, "car", color_idx=1)
    _built(tmp_path, "after", [])

    before = _splits(tmp_path / "before")
    after = _splits(tmp_path / "after")

    assert set(before) <= set(after)
    moved = {stem: (before[stem], after[stem]) for stem in before if before[stem] != after[stem]}
    assert not moved, f"crops changed split after the pool grew: {moved}"
