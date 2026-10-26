# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The composition table decides which classes can be reported."""

from __future__ import annotations

from pathlib import Path

from report_dataset import render


def _dataset(root: Path) -> Path:
    (root / "train" / "labels").mkdir(parents=True)
    (root / "valid" / "labels").mkdir(parents=True)
    (root / "test" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("nc: 3\nnames: ['a', 'b', 'c']\n", encoding="utf-8")
    for split, blocks in (("train", {"a": 100, "b": 40, "c": 2}),
                          ("valid", {"a": 20, "b": 8, "c": 1}),
                          ("test", {"a": 50, "b": 10, "c": 1})):
        index = {"a": 0, "b": 1, "c": 2}
        for name, count in blocks.items():
            for i in range(count):
                (root / split / "labels" / f"{name}_{i}.txt").write_text(
                    f"{index[name]} 0.5 0.5 1 1\n", encoding="utf-8")
    return root


def test_counts_come_from_the_labels_not_from_the_pool(tmp_path: Path) -> None:
    text = render(_dataset(tmp_path / "ds"), "t", 40)

    assert "| a | 100 | 20 | 50 | 170 |" in text
    assert "| c | 2 | 1 | 1 | 4 |" in text


def test_the_table_says_which_classes_are_reportable(tmp_path: Path) -> None:
    text = render(_dataset(tmp_path / "ds"), "t", 40)

    assert "| a | 100 | 20 | 50 | 170 | 73.3% | sí |" in text
    assert "| b | 40 | 8 | 10 | 58 | 25.0% | **no** |" in text
    assert "Clases reportables (test n ≥ 40): 1 de 3 — a." in text


def test_missing_splits_do_not_crash(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    for path in (root / "valid" / "labels").glob("*.txt"):
        path.unlink()

    text = render(root, "t", 40)

    assert "| a | 100 | 0 | 50 |" in text
