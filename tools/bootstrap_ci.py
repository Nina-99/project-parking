# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Confidence intervals that respect the unit of analysis.

Three subcommands, one per shape of input the evaluation actually produces:

    confusion   Bootstrap macro-F1 from a stored confusion matrix.
    items       Bootstrap any per-item metric, optionally by cluster.
    proportion  Wilson/Clopper-Pearson interval and one-sided test.

Why this exists: a binomial interval assumes independent Bernoulli trials. Test
crops are not independent -- several crops share a plate or a recording session,
and macro-F1 is an average of per-class ratios, not a proportion. Applying a
binomial interval to those numbers reports an error bar narrower than the
evidence supports. The bootstrap below resamples the unit that was actually
sampled (image, plate, or session) and inherits its dependence structure.

Examples:

    # Macro-F1 interval for the colour classifier from its metrics JSON
    uv run python tools/bootstrap_ci.py confusion \
        ../datasets/runs/resnet18_final_color_v5_metrics.json --threshold 0.85

    # Per-item predictions, resampling whole plates instead of crops
    uv run python tools/bootstrap_ci.py items color_test.jsonl \
        --cluster-regex '^(?P<cluster>[0-9]{4}-[0-9]{2}-[0-9]{2})' \
        --threshold 0.85

    # OCR: macro character accuracy over plates, resampling plates
    uv run python tools/bootstrap_ci.py items ocr_test.jsonl \
        --task text --cluster-col plate --threshold 0.90

The interval is a percentile bootstrap (Efron & Tibshirani, 1993). Cluster
resampling follows Field & Welsh (2007). Neither accounts for variance between
training runs; report that separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_REPLICATES = 2000
DEFAULT_SEED = 20261104
DEFAULT_ALPHA = 0.05


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """Wilson score interval. Stable at 0 and 1, unlike the Wald interval."""
    if n == 0:
        return (0.0, 1.0)
    z = _z(alpha)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Clamp so the reported interval always contains its own point estimate;
    # at p = 0 or 1 the closed form can land a float epsilon short of it.
    return (min(max(0.0, centre - half), p), max(min(1.0, centre + half), p))


def clopper_pearson(k: int, n: int, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """Exact interval by inverting the binomial test at each tail.

    Solved by bisection on the binomial CDF so the module needs no SciPy.
    """
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else _solve(lambda p: _binom_sf(k - 1, n, p) - alpha / 2, 0.0, 1.0)
    hi = 1.0 if k == n else _solve(lambda p: _binom_cdf(k, n, p) - alpha / 2, 0.0, 1.0)
    return (max(0.0, lo), min(1.0, hi))


def binom_test_greater(k: int, n: int, p0: float) -> float:
    """P(X >= k | p = p0), the p-value for H1: p > p0."""
    return _binom_sf(k - 1, n, p0)


def binom_test_less(k: int, n: int, p0: float) -> float:
    """P(X <= k | p = p0), the p-value for H1: p < p0."""
    return _binom_cdf(k, n, p0)


def _binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    log_pmf = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    return math.exp(log_pmf)


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(_binom_pmf(i, n, p) for i in range(k + 1)))


def _binom_sf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 1.0
    if k >= n:
        return 0.0
    return max(0.0, min(1.0, 1.0 - _binom_cdf(k, n, p)))


def _solve(fn, lo: float, hi: float, tol: float = 1e-10) -> float:
    """Bisection for a monotone fn, used by the exact interval."""
    f_lo, f_hi = fn(lo), fn(hi)
    if f_lo * f_hi > 0:  # no sign change: clamp to the boundary
        return lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(200):
        mid = (lo + hi) / 2
        if fn(mid) * f_lo <= 0:
            hi = mid
        else:
            lo, f_lo = mid, fn(mid)
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def _z(alpha: float) -> float:
    """Two-sided normal quantile for the Wilson interval."""
    table = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}
    if alpha in table:
        return table[alpha]
    return 1.959963984540054


def percentile_ci(samples: np.ndarray, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    if samples.size == 0:
        raise SystemExit(
            "[bootstrap] no replicates: the input produced no measurable items"
        )
    lower = float(np.percentile(samples, 100 * alpha / 2))
    upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return (lower, upper)


# --------------------------------------------------------------------------- #
# classification metrics from a confusion matrix
# --------------------------------------------------------------------------- #
def macro_f1(matrix: np.ndarray, evaluated: list[int]) -> float:
    """Macro-F1 over a fixed set of classes.

    The set is fixed by the caller, not recomputed per replicate: the estimand
    is "macro-F1 over the classes that met the support gate in the observed
    test set". Letting the gate re-decide on every resample would move the
    target instead of estimating it.
    """
    if not evaluated:
        return float("nan")
    scores = []
    for i in evaluated:
        tp = matrix[i, i]
        support = matrix[i, :].sum()
        predicted = matrix[:, i].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def accuracy(matrix: np.ndarray) -> float:
    total = matrix.sum()
    return float(np.trace(matrix) / total) if total else 0.0


def bootstrap_confusion(
    matrix: np.ndarray,
    supports: list[int],
    evaluated: list[int],
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Stratified parametric bootstrap of a multinomial confusion matrix.

    Each true class is resampled from its observed row distribution. This
    captures the sampling error of the cell counts given the fitted model. It
    does not capture variance across training runs, which needs repeated runs.
    """
    rng = np.random.default_rng(seed)
    n_classes = matrix.shape[0]
    rows = [matrix[i, :].astype(float) for i in range(n_classes)]
    probs = [row / row.sum() if row.sum() else row for row in rows]
    f1_samples = np.empty(replicates)
    acc_samples = np.empty(replicates)
    for r in range(replicates):
        draw = np.zeros_like(matrix, dtype=float)
        for i in range(n_classes):
            n_i = supports[i]
            if n_i == 0:
                continue
            draw[i, :] = rng.multinomial(n_i, probs[i])
        f1_samples[r] = macro_f1(draw, evaluated)
        acc_samples[r] = accuracy(draw)
    f1_samples = f1_samples[~np.isnan(f1_samples)]
    return {
        "macro_f1": macro_f1(matrix, evaluated),
        "macro_f1_ci": percentile_ci(f1_samples),
        "accuracy": accuracy(matrix),
        "accuracy_ci": percentile_ci(acc_samples),
        "n": int(matrix.sum()),
        "replicates": int(f1_samples.size),
    }


# --------------------------------------------------------------------------- #
# per-item input
# --------------------------------------------------------------------------- #
def read_items(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance; short plate strings, so the plain DP is enough."""
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def cluster_keys(items: list[dict], args: argparse.Namespace) -> list[str]:
    if args.cluster_col:
        try:
            return [str(row[args.cluster_col]) for row in items]
        except KeyError as exc:
            sys.exit(f"[bootstrap] missing cluster column {exc} in the input")
    if args.cluster_regex:
        pattern = re.compile(args.cluster_regex)
        keys = []
        for row in items:
            match = pattern.search(str(row.get(args.key_col, "")))
            if not match or "cluster" not in match.groupdict():
                sys.exit(
                    f"[bootstrap] {args.cluster_regex!r} did not yield a named group "
                    f"'cluster' for {row.get(args.key_col)!r}"
                )
            keys.append(match.group("cluster"))
        return keys
    return [str(i) for i in range(len(items))]


def macro_f1_from_items(truth: list[str], pred: list[str], evaluated: set[str]) -> float:
    labels = [c for c in evaluated if c]
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(truth, pred) if t == label and p == label)
        support = sum(1 for t in truth if t == label)
        predicted = sum(1 for p in pred if p == label)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else float("nan")


def bootstrap_items(
    items: list[dict],
    args: argparse.Namespace,
) -> dict:
    keys = cluster_keys(items, args)
    unique = sorted(set(keys))
    groups: dict[str, list[int]] = {k: [] for k in unique}
    for idx, key in enumerate(keys):
        groups[key].append(idx)

    truth = [str(row["truth"]).strip() for row in items]
    pred_col = getattr(args, "pred_col", None) or "pred"
    try:
        pred = [str(row[pred_col]).strip() for row in items]
    except KeyError:
        sys.exit(f"[bootstrap] the input has no {pred_col!r} column")

    if args.task == "text":
        per_item = []
        for t, p in zip(truth, pred):
            distance = edit_distance(t, p)
            per_item.append(1.0 - distance / max(len(t), 1))
        exact = [1.0 if t == p else 0.0 for t, p in zip(truth, pred)]

        if args.macro_by_cluster:
            # A plate with two crops must weigh the same as a plate with one,
            # which is what the chapter reports ("macro over plates").
            cluster_values = [
                (
                    float(np.mean([per_item[i] for i in groups[key]])),
                    float(np.mean([exact[i] for i in groups[key]])),
                )
                for key in unique
            ]
            cluster_value_by_key = dict(zip(unique, cluster_values, strict=True))
            point = (
                float(np.mean([v[0] for v in cluster_values])),
                float(np.mean([v[1] for v in cluster_values])),
            )
        else:
            cluster_values = None
            cluster_value_by_key = None

            def stat(indices: list[int]) -> tuple[float, float]:
                return (float(np.mean([per_item[i] for i in indices])),
                        float(np.mean([exact[i] for i in indices])))

            point = stat(list(range(len(items))))
    else:
        counts: dict[str, int] = {}
        for t in truth:
            counts[t] = counts.get(t, 0) + 1
        evaluated = {c for c, n in counts.items() if n >= args.min_support}

        cluster_values = None
        cluster_value_by_key = None

        def stat(indices: list[int]) -> tuple[float, float]:
            t = [truth[i] for i in indices]
            p = [pred[i] for i in indices]
            correct = sum(1 for a, b in zip(t, p) if a == b)
            return (macro_f1_from_items(t, p, evaluated), correct / len(indices))

        point = stat(list(range(len(items))))

    rng = random.Random(args.seed)
    f1_samples, acc_samples = [], []
    for _ in range(args.replicates):
        chosen = [rng.choice(unique) for _ in unique]
        if cluster_values is not None:
            value = float(np.mean([cluster_value_by_key[k][0] for k in chosen]))
            acc = float(np.mean([cluster_value_by_key[k][1] for k in chosen]))
        else:
            indices = [idx for key in chosen for idx in groups[key]]
            if not indices:
                continue
            value, acc = stat(indices)
        if not math.isnan(value):
            f1_samples.append(value)
        acc_samples.append(acc)

    return {
        "primary": point[0],
        "primary_ci": percentile_ci(np.array(f1_samples)),
        "secondary": point[1],
        "secondary_ci": percentile_ci(np.array(acc_samples)),
        "n_items": len(items),
        "n_clusters": len(unique),
        "replicates": len(f1_samples),
        "task": args.task,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def decision(lower: float, upper: float, threshold: float) -> str:
    if lower >= threshold:
        return "supported (interval entirely at or above threshold)"
    if upper < threshold:
        return "rejected (interval entirely below threshold)"
    return "inconclusive (interval contains threshold)"


def print_report(data: dict, threshold: float | None, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    for key, value in data.items():
        if key in {"threshold", "decision"}:
            continue
        if isinstance(value, tuple):
            lo, hi = value
            print(f"{key:14s} [{lo:.4f}, {hi:.4f}]  width={hi - lo:.4f}")
        elif isinstance(value, float):
            print(f"{key:14s} {value:.4f}")
        else:
            print(f"{key:14s} {value}")
    if threshold is not None:
        ci = data.get("macro_f1_ci") or data.get("primary_ci")
        if ci:
            print(f"{'threshold':14s} {threshold:.4f}")
            print(f"{'decision':14s} {decision(ci[0], ci[1], threshold)}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Confidence intervals that respect the unit of analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    conf = sub.add_parser("confusion", help="Bootstrap macro-F1 from a metrics JSON.")
    conf.add_argument("metrics", type=Path, help="JSON with 'confusion' and 'classes'.")
    conf.add_argument("--min-support", type=int, default=40,
                      help="Classes below this support stay outside the macro average.")
    conf.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    conf.add_argument("--seed", type=int, default=DEFAULT_SEED)
    conf.add_argument("--threshold", type=float)
    conf.add_argument("--json", action="store_true", dest="as_json")

    items = sub.add_parser("items", help="Bootstrap a metric from per-item predictions.")
    items.add_argument("predictions", type=Path, help="JSONL or CSV with truth/pred.")
    items.add_argument("--task", choices=("classification", "text"), default="classification")
    items.add_argument("--task-text", action="store_true", help=argparse.SUPPRESS)
    items.add_argument("--cluster-col", help="Column holding the cluster id (plate, session).")
    items.add_argument("--cluster-regex", help="Regex with a named group 'cluster'.")
    items.add_argument("--pred-col", default="pred",
                       help="Column holding the prediction (default: pred). Use it to "
                            "compare, say, the detector box against the human box.")
    items.add_argument("--key-col", default="stem", help="Column the regex runs on.")
    items.add_argument("--min-support", type=int, default=40)
    items.add_argument("--macro-by-cluster", action="store_true",
                       help="Text task only: average within each cluster first, then "
                            "across clusters, so a plate with several crops weighs the "
                            "same as a plate with one.")
    items.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    items.add_argument("--seed", type=int, default=DEFAULT_SEED)
    items.add_argument("--threshold", type=float)
    items.add_argument("--json", action="store_true", dest="as_json")

    prop = sub.add_parser("proportion", help="Exact and Wilson intervals for k/n.")
    prop.add_argument("k", type=int)
    prop.add_argument("n", type=int)
    prop.add_argument("--threshold", type=float, required=True)
    prop.add_argument("--json", action="store_true", dest="as_json")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "confusion":
        payload = json.loads(args.metrics.read_text(encoding="utf-8"))
        if "confusion" not in payload or "classes" not in payload:
            sys.exit("[bootstrap] metrics JSON needs 'confusion' and 'classes'")
        matrix = np.array(payload["confusion"], dtype=int)
        supports = [int(matrix[i, :].sum()) for i in range(matrix.shape[0])]
        evaluated = [i for i, n in enumerate(supports) if n >= args.min_support]
        result = bootstrap_confusion(matrix, supports, evaluated,
                                     args.replicates, args.seed)
        if args.threshold is not None:
            result["decision"] = decision(*result["macro_f1_ci"], args.threshold)
            result["threshold"] = args.threshold
        print_report(result, args.threshold, args.as_json)
        return

    if args.command == "items":
        if args.task_text:
            args.task = "text"
        items = read_items(args.predictions)
        if not items:
            sys.exit(
                f"[bootstrap] {args.predictions} has no rows. Nothing was measured, "
                "so there is no interval to compute."
            )
        # An empty reference cannot be scored: Levenshtein against "" would drive
        # the per-item accuracy negative. Drop those rows and say so.
        scored = [row for row in items if str(row.get("truth", "")).strip()]
        if len(scored) != len(items):
            print(
                f"[bootstrap] warning: dropping {len(items) - len(scored)} rows with "
                "an empty truth",
                file=sys.stderr,
            )
        if not scored:
            sys.exit(
                f"[bootstrap] {args.predictions} has no row with a non-empty truth"
            )
        result = bootstrap_items(scored, args)
        if args.threshold is not None:
            result["decision"] = decision(*result["primary_ci"], args.threshold)
            result["threshold"] = args.threshold
        print_report(result, args.threshold, args.as_json)
        return

    if args.k > args.n:
        sys.exit("[bootstrap] k cannot exceed n")
    lo_w, hi_w = wilson(args.k, args.n)
    lo_c, hi_c = clopper_pearson(args.k, args.n)
    result = {
        "point": args.k / args.n,
        "wilson": (lo_w, hi_w),
        "clopper_pearson": (lo_c, hi_c),
        "p_less": binom_test_less(args.k, args.n, args.threshold),
        "p_greater": binom_test_greater(args.k, args.n, args.threshold),
        "threshold": args.threshold,
        "decision": decision(lo_w, hi_w, args.threshold),
    }
    print_report(result, args.threshold, args.as_json)


if __name__ == "__main__":
    main()
