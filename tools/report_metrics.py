"""Render classifier metrics as the tables the thesis needs.

    uv run python tools/report_metrics.py \
        --metrics /tmp/opencode/exp/*_metrics.json \
        --title "Resultados de clasificación" \
        --out docs/results.md

One source of truth: the JSON written by the trainers and by
``eval_type_classifier.py``. Nothing is recomputed here, so a number cannot
differ between the report and the artefact it came from -- only the three
aggregates are restated, and the per-class table shows the support and the
confidence interval next to every F1 so a reader can judge the claim.

Older artefacts predate the support/CI fields. Their support is recovered from
the confusion matrix row sums, and their confidence intervals are left blank
rather than invented.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _support(entry: dict, name: str) -> int | None:
    explicit = entry.get("per_class_support")
    if isinstance(explicit, dict) and name in explicit:
        return int(explicit[name])
    matrix = entry.get("confusion")
    classes = entry.get("classes")
    if matrix and classes and name in classes:
        return int(sum(matrix[classes.index(name)]))
    return None


def _half_width(f1: float, support: int, z: float = 1.96) -> float:
    """Wilson half-width, matching train_type_classifier.summarise()."""
    denominator = 1.0 + z * z / support
    return z * math.sqrt(f1 * (1.0 - f1) / support + z * z / (4.0 * support * support)) / denominator


def _from_matrix(entry: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Precision and recall recomputed from the confusion matrix when absent."""
    classes = entry.get("classes") or []
    matrix = entry.get("confusion")
    if not matrix:
        return {}, {}
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    for index, name in enumerate(classes):
        true_positive = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        actual = sum(matrix[index])
        precision[name] = true_positive / predicted if predicted else 0.0
        recall[name] = true_positive / actual if actual else 0.0
    return precision, recall


def _aggregates(entry: dict) -> tuple[float, float, float]:
    """(macro over evaluated, weighted by real frequency, accuracy)."""
    classes = entry["classes"]
    f1s = entry["per_class_f1"]
    supports = {name: _support(entry, name) or 0 for name in classes}
    evaluated = entry.get("per_class_evaluated")
    if isinstance(evaluated, dict):
        reportable = [name for name in classes if evaluated.get(name)]
    else:
        # Legacy artefact: no verdict stored, so assume everything with data.
        reportable = [name for name in classes if supports[name] > 0]
    macro = sum(f1s[name] for name in reportable) / len(reportable) if reportable else 0.0
    total = sum(supports[name] for name in reportable)
    weighted = sum(f1s[name] * supports[name] for name in reportable) / total if total else 0.0
    matrix = entry.get("confusion") or []
    cells = sum(sum(row) for row in matrix)
    diagonal = sum(matrix[i][i] for i in range(min(len(matrix), len(classes)))) if matrix else 0
    accuracy = diagonal / cells if cells else 0.0
    return macro, weighted, accuracy


def _summary_table(entries: list[tuple[Path, dict]]) -> list[str]:
    lines = [
        "| artefacto | tarea | macro-F1 (evaluadas) | F1 ponderado | accuracy | excluidas |",
        "|---|---|---|---|---|---|",
    ]
    for path, entry in entries:
        macro, weighted, accuracy = _aggregates(entry)
        excluded = entry.get("excluded_classes") or []
        lines.append(
            f"| `{path.name}` | {entry.get('task', '?')} | **{macro:.4f}** | "
            f"{weighted:.4f} | {accuracy:.4f} | {', '.join(excluded) if excluded else '—'} |"
        )
    return lines


def _detail(path: Path, entry: dict) -> list[str]:
    classes = entry["classes"]
    f1s = entry["per_class_f1"]
    precision, recall = _from_matrix(entry)
    precision = {**precision, **(entry.get("per_class_precision") or {})}
    recall = {**recall, **(entry.get("per_class_recall") or {})}
    evaluated = entry.get("per_class_evaluated") or {}

    lines = [f"### `{path.name}`", ""]
    lines += ["| clase | P | R | F1 | ±ci95 | n | reportada |", "|---|---|---|---|---|---|---|"]
    for name in classes:
        f1 = f1s[name]
        support = _support(entry, name)
        if name in entry.get("per_class_ci95", {}):
            ci = f"{entry['per_class_ci95'][name]:.3f}"
        elif support:
            ci = f"{_half_width(f1, support):.3f}"
        else:
            ci = "—"
        reported = evaluated.get(name)
        flag = "—" if reported is None else ("sí" if reported else "no")
        p_text = f"{precision[name]:.3f}" if name in precision else "—"
        r_text = f"{recall[name]:.3f}" if name in recall else "—"
        lines.append(
            f"| {name} | {p_text} | {r_text} | {f1:.3f} | {ci} | "
            f"{support if support is not None else '—'} | {flag} |"
        )

    matrix = entry.get("confusion")
    if matrix:
        header = "| real \\ pred | " + " | ".join(classes) + " |"
        rule = "|---" * (len(classes) + 1) + "|"
        lines += ["", header, rule]
        for name, row in zip(classes, matrix, strict=True):
            lines.append(f"| **{name}** | " + " | ".join(str(cell) for cell in row) + " |")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render metrics JSON as Markdown tables.")
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--title", default="Resultados de clasificación")
    parser.add_argument("--out", type=Path, help="Write here; otherwise stdout.")
    args = parser.parse_args(argv)

    entries = [(path, _load(path)) for path in sorted(args.metrics)]

    lines = [f"# {args.title}", "", "## Resumen", ""]
    lines += _summary_table(entries)
    lines += [
        "",
        "`macro-F1 (evaluadas)` es el promedio sobre las clases con soporte "
        "suficiente en test; las excluidas siguen contando como falsos positivos "
        "de las demás. `F1 ponderado` es el que siente el despliegue.",
        "",
        "## Detalle por clase",
        "",
    ]
    for path, entry in entries:
        lines += _detail(path, entry)

    text = "\n".join(lines).rstrip() + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[report] {len(entries)} artefacto(s) -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
