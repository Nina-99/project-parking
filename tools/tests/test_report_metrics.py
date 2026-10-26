# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The thesis tables must be a faithful rendering of the stored metrics."""

from __future__ import annotations

import json
from pathlib import Path

from report_metrics import main

NEW = {
    "backbone": "resnet18",
    "task": "color",
    "classes": ["blanco", "plata", "amarillo"],
    "test_macro_f1_evaluated": 0.9,
    "per_class_f1": {"blanco": 0.9, "plata": 0.8, "amarillo": 0.1},
    "per_class_precision": {"blanco": 0.9, "plata": 0.8, "amarillo": 0.1},
    "per_class_recall": {"blanco": 0.9, "plata": 0.8, "amarillo": 0.5},
    "per_class_ci95": {"blanco": 0.05, "plata": 0.07, "amarillo": 0.38},
    "per_class_support": {"blanco": 100, "plata": 80, "amarillo": 2},
    "per_class_evaluated": {"blanco": True, "plata": True, "amarillo": False},
    "excluded_classes": ["amarillo"],
    "confusion": [[90, 10, 0], [15, 64, 1], [1, 0, 1]],
}

LEGACY = {
    "backbone": "resnet18",
    "classes": ["blanco", "plata"],
    "test_macro_f1": 0.5,
    "per_class_f1": {"blanco": 0.8, "plata": 0.6},
    "confusion": [[80, 20], [30, 70]],
}


def _render(tmp_path: Path, payloads: dict[str, dict]) -> str:
    paths = []
    for name, payload in payloads.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(str(path))
    out = tmp_path / "report.md"
    main(["--metrics", *paths, "--out", str(out)])
    return out.read_text(encoding="utf-8")


def test_the_report_shows_support_and_the_evaluated_verdict(tmp_path: Path) -> None:
    text = _render(tmp_path, {"m.json": NEW})

    assert "| blanco | 0.900 | 0.900 | 0.900 | 0.050 | 100 | sí |" in text
    assert "| amarillo |" in text and "| no |" in text


def test_the_aggregate_excludes_unreported_classes(tmp_path: Path) -> None:
    text = _render(tmp_path, {"m.json": NEW})
    # (0.9 + 0.8) / 2, never dragged down by amarillo
    assert "**0.8500**" in text


def test_a_legacy_artefact_gets_its_support_from_the_matrix(tmp_path: Path) -> None:
    text = _render(tmp_path, {"old.json": LEGACY})

    # precision and recall recomputed: tp=80, predicted 80+30, actual 80+20
    assert "| blanco | 0.727 | 0.800 | 0.800 |" in text
    assert "| 100 |" in text  # 80 + 20, recovered from the confusion row
    assert "—" in text  # no verdict stored, nothing invented


def test_the_confusion_matrix_is_reproduced_verbatim(tmp_path: Path) -> None:
    text = _render(tmp_path, {"m.json": NEW})
    assert "| **blanco** | 90 | 10 | 0 |" in text
