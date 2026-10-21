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
from build_color_dataset import main
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
