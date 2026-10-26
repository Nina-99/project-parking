# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Vehicle attribute vocabulary and the labelling queue.

Body type and predominant colour are *semantic* attributes: no deterministic
rule derives them from a crop, so they are assigned by a human. An automatic
labeller (HSV thresholds, say) would hand the classifier its own teacher, and
the baseline-versus-network comparison would then measure nothing more than how
faithfully the network copies that rule.

This module owns the vocabulary, the queue bookkeeping (which crops still lack
a label) and the append-only label store.

Layout under the dataset root::

    attributes/images/<video>/<stem>.jpg   # vehicle crop awaiting a label
    attributes/manifest.jsonl              # one row per queued crop
    attributes/labels.jsonl                # one row per human label
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

# Default vocabulary. These names reflect what is actually registered in the
# study area (INE "Parque Automotor Bolivia"), not the categories of a public
# dataset: the official `automovil` is used instead of the generic `sedan`.
# These names become directory names when the set is exported for training, so
# renaming a class after labelling invalidates every label already written.
DEFAULT_TYPES: tuple[str, ...] = (
    "automovil",
    "vagoneta",
    "camioneta",
    "minibus",
    "motocicleta",
    "otro",
)

# Colour palette aligned with the Latin-American benchmark of Lima & Laroca,
# "Toward Enhancing Vehicle Color Recognition in Adverse Conditions": beige,
# black, blue, brown, gray, green, orange, red, silver, white, yellow. Vehicles
# their benchmark calls "multicolored" (fantasía) fall into `otro` here.
#
# `silver` and `gray` are merged into a single `gris_plata`: at surveillance
# resolution, under sun and reflections, the metallic sheen of silver is not a
# boundary a human can draw consistently, and the benchmark itself reports that
# gray is routinely mistaken for other shades. Merging removes that label noise
# from the ground truth instead of asking two annotators to agree on the
# impossible.
DEFAULT_COLORS: tuple[str, ...] = (
    "blanco",
    "gris_plata",
    "negro",
    "rojo",
    "azul",
    "verde",
    "amarillo",
    "marron",
    "beige",
    "naranja",
    "otro",
)

# Detector class ids the extractor keeps (see ``CameraConfig.vehicle_classes``).
# They hint the *category* only: COCO has no sedan / SUV / pickup split, so the
# hint is shown to the labeller and never written as ground truth.
COCO_VEHICLE_NAMES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Keys used by the labeller: home-row digits first (1..9, then 0 for the tenth
# class), then `q` for the eleventh. Reserved control keys (`s` skip, `b` back,
# `x` quit, `d` discard) are deliberately outside this set.
_LABEL_KEYS = "1234567890q"

# Reason stored when a crop is discarded as unjudgeable (bad illumination,
# heavy occlusion, vehicle barely visible). Discards are recorded, not deleted:
# the count is itself a result worth reporting, and the file stays on disk in
# case the decision is revisited.
DISCARD_REASON = "unusable"


def key_to_index(ch: str, count: int) -> int | None:
    """Map a keyboard key to a class index, or ``None`` when out of range."""
    if len(ch) != 1:
        return None
    pos = _LABEL_KEYS.find(ch)
    if pos < 0 or pos >= count:
        return None
    return pos


def key_for_index(index: int) -> str:
    """Key the labeller expects for class ``index`` — used to draw the legend.

    The legend must show the *actual* key, not ``index + 1``: the tenth class is
    selected with ``0`` and the eleventh with ``q``, so displaying "10" would
    invite the user to type two keys and pick the wrong class.
    """
    if 0 <= index < len(_LABEL_KEYS):
        return _LABEL_KEYS[index]
    return "?"


def iter_manifest(path: Path) -> Iterator[dict]:
    """Yield the rows of a JSONL file, skipping blank and truncated lines.

    A batch that dies mid-write can leave a half-written final line; the queue
    must stay usable instead of aborting on it.
    """
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_labelled(path: Path) -> dict[str, dict]:
    """Read the append-only label store, keyed by sample stem (last wins)."""
    labelled: dict[str, dict] = {}
    for row in iter_manifest(path):
        stem = row.get("stem")
        if isinstance(stem, str):
            labelled[stem] = row
    return labelled


def pending_samples(manifest_path: Path, labels_path: Path) -> list[dict]:
    """Queued crops that still have no label, in manifest order."""
    labelled = load_labelled(labels_path)
    seen: set[str] = set()
    pending: list[dict] = []
    for row in iter_manifest(manifest_path):
        stem = row.get("stem")
        if not isinstance(stem, str) or stem in labelled or stem in seen:
            continue
        seen.add(stem)
        pending.append(row)
    return pending


def image_path(root: Path, sample: dict) -> Path:
    """Absolute path of the crop for ``sample``.

    A sample may carry an explicit ``image`` path (relative to the dataset root)
    when it does not live under ``attributes/images`` — the gate-view sampler
    points at ``vehicles/`` and ``review/vehicles/`` instead. Without it, the
    default attribute-queue layout is assumed.
    """
    rel = sample.get("image")
    if isinstance(rel, str) and rel:
        return Path(root) / rel
    video = str(sample.get("video", ""))
    stem = str(sample.get("stem", ""))
    return Path(root) / "attributes" / "images" / video / f"{stem}.jpg"


_STEM_RE = re.compile(r"^(?P<video>.+)_track(?P<track>\d+)_f(?P<frame>\d+)$")


def parse_stem(stem: str) -> dict | None:
    """Split ``<video>_track<id>_f<frame>`` into its parts, or ``None``."""
    match = _STEM_RE.match(stem)
    if match is None:
        return None
    return {
        "video": match.group("video"),
        "track_id": int(match.group("track")),
        "frame": int(match.group("frame")),
    }


def pick_representatives(
    rows: "Iterator[dict] | Sequence[dict]",
    prefer: "set[str] | None" = None,
) -> list[dict]:
    """Keep one crop per ``(video, track_id)``: the most *typical* one.

    The extractor queues several top-k frames for every vehicle track, so the
    raw manifest is dominated by near-duplicates of the same car. Picking the
    largest crop (the obvious choice) is wrong: a vehicle passing right next to
    the camera yields a huge close-up of its roof, useless for body type and
    colour. The area distribution is sharply bimodal — a dense band of normal
    crops plus a thin tail of close-ups — so the representative is the crop
    whose area sits closest to the *pool median*, which lives in the dense band
    and therefore rejects both the close-up and the tiny distant frame.

    ``prefer`` lets already-labelled stems win over the median pick, so fixing
    the sampler does not throw away annotation work. Discarded stems must not be
    passed here: their track should be re-sampled with a better crop.
    """
    prefer = prefer or set()

    def area(row: dict) -> int:
        hw = row.get("crop_hw") or [0, 0]
        return int(hw[0]) * int(hw[1]) if len(hw) >= 2 else 0

    valid = [
        row for row in rows
        if isinstance(row.get("stem"), str)
        and isinstance(row.get("video"), str)
        and row.get("track_id") is not None
    ]

    # Target = median area of the candidate pool: a "normal" vehicle distance.
    areas = sorted(area(row) for row in valid)
    if areas:
        mid = len(areas) // 2
        target = areas[mid] if len(areas) % 2 else (areas[mid - 1] + areas[mid]) / 2
    else:
        target = 0

    groups: dict[tuple, list[dict]] = {}
    for row in valid:
        groups.setdefault((row["video"], row["track_id"]), []).append(row)

    picked: list[dict] = []
    for group in groups.values():
        chosen = next((r for r in group if r.get("stem") in prefer), None)
        if chosen is None:
            chosen = min(
                group,
                key=lambda r: (abs(area(r) - target), -float(r.get("vehicle_score") or 0.0)),
            )
        picked.append(chosen)
    return picked


def stratified_sample(
    rows: "Sequence[dict]",
    target: int,
    seed: int,
    keys: Sequence[str] = ("camera", "vehicle_class"),
) -> list[dict]:
    """Deterministically subsample ``rows`` down to ``target``, stratified.

    The raw activity is uneven: busy hours contribute far more tracks than quiet
    ones, and cars outnumber buses and trucks. A plain random sample would
    inherit that skew, so the network would barely ever see the rare classes.
    Sampling evenly per ``(camera, vehicle_class)`` group keeps the rare classes
    represented. Reproducible via ``seed`` so the thesis split can be re-created.
    """
    rows = list(rows)
    if target <= 0 or len(rows) <= target:
        return rows

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        groups.setdefault(key, []).append(row)

    rng = random.Random(seed)
    picked: list[dict] = []
    total = len(rows)
    for key in sorted(groups, key=repr):
        group = sorted(groups[key], key=lambda r: str(r.get("stem")))
        quota = max(1, round(len(group) / total * target))
        quota = min(quota, len(group))
        picked.extend(rng.sample(group, quota))

    # Proportional rounding can overshoot; trim deterministically.
    if len(picked) > target:
        picked = sorted(picked, key=lambda r: str(r.get("stem")))
        rng.shuffle(picked)
        picked = picked[:target]
    return picked


def write_manifest(path: Path, rows: Sequence[dict]) -> int:
    """Write rows as JSONL, returning the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def validate_label(
    vehicle_type: str, color: str, types: Sequence[str], colors: Sequence[str]
) -> None:
    """Reject a label outside the declared vocabulary."""
    if vehicle_type not in types:
        raise ValueError(
            f"unknown vehicle type {vehicle_type!r}; expected one of {list(types)}"
        )
    if color not in colors:
        raise ValueError(f"unknown colour {color!r}; expected one of {list(colors)}")


def append_label(
    path: Path,
    stem: str,
    vehicle_type: str,
    color: str,
    types: Sequence[str],
    colors: Sequence[str],
    source: str = "human",
) -> dict:
    """Append one validated label and return the row written.

    The store is append-only so a crash during labelling loses at most the
    sample being labelled, and re-labelling a stem is a correction rather than
    a corruption. Only the vocabulary is validated here: the label *quality* is
    the labeller's responsibility.
    """
    validate_label(vehicle_type, color, types, colors)
    row = {
        "stem": stem,
        "type": vehicle_type,
        "color": color,
        "source": source,
        "ts": datetime.now(UTC).isoformat(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def append_discard(
    path: Path,
    stem: str,
    reason: str = DISCARD_REASON,
    source: str = "human",
) -> dict:
    """Record that a queued crop is not judgeable, and drop it from the queue.

    Some crops cannot be labelled honestly: the vehicle is barely in frame,
    almost black under a burnt shadow, or blurred beyond recognition. The label
    is not "unknown colour" — it is "this sample must not train or score
    anything". Writing the verdict to the same append-only store means
    :func:`pending_samples` stops asking for it on the next run, and the count
    of discards becomes a reportable datum (data quality of the extractor).
    """
    row = {
        "stem": stem,
        "type": None,
        "color": None,
        "discard": True,
        "reason": reason,
        "source": source,
        "ts": datetime.now(UTC).isoformat(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
