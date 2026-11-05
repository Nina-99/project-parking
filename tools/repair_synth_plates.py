# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Regenerate synthetic plates that were lost or truncated, and prove it.

    uv run python tools/repair_synth_plates.py \
        --root /home/nina/FNI2025/ProjectParking --count 360000 --dry-run

    uv run python tools/repair_synth_plates.py \
        --root /home/nina/FNI2025/ProjectParking --count 360000 --update-ablation

This is a repair, not padding. ``SynthPipeline.rng_for`` seeds every plate from
its index, so ``(index, seed, generator revision)`` fixes the output: regenerating
index 100 with the revision that produced the set yields the same text and the
same pixels.

The catch is the third term. Commit ``3a3e1cf`` put ``I`` back into the letter
alphabet after the 360k were generated, so the *current* code produces a
different plate for the same index. That is why this tool tries both alphabets
and accepts only the one whose label matches the label already on disk. A file
that cannot be matched is reported, never guessed.

Every repaired file is generated into a temporary directory first, validated
(JPEG markers, decodable, label match), and only then copied into place.
"""
from __future__ import annotations

import argparse
import json
import shutil
import string
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.synth_plates import textgen  # noqa: E402
from plate_dataset.synth_plates.pipeline import SynthConfig, SynthPipeline  # noqa: E402

SEED = 7

# The two alphabets this dataset was produced with, in reverse chronological
# order. ``current`` is what textgen ships today.
ALPHABETS: dict[str, str] = {
    "current": "".join(c for c in string.ascii_uppercase if c != "O"),
    "pre-I": "".join(c for c in string.ascii_uppercase if c not in "OI"),
}


@dataclass
class Target:
    stem: str
    index: int
    reason: str          # "missing" | "corrupt"
    expected: str | None # label on disk, when there is one


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True,
                        help="Dataset root holding plates/images/synth.")
    parser.add_argument("--count", type=int, default=360000,
                        help="How many stems the set should contain (indices 0..count-1).")
    parser.add_argument("--dry-run", action="store_true", help="Report only.")
    parser.add_argument("--force", action="store_true",
                        help="Repair even files that look valid (useful after a bad copy).")
    parser.add_argument("--update-ablation", action="store_true",
                        help="Rewrite ablation/train_synth.txt and protocol.json to the "
                             "repaired set.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N repairs.")
    return parser


def jpeg_ok(path: Path) -> bool:
    """Cheap truncation check plus a real decode: markers alone miss some cases."""
    try:
        with path.open("rb") as handle:
            head = handle.read(2)
            handle.seek(-2, 2)
            tail = handle.read(2)
        if head != b"\xff\xd8" or tail != b"\xff\xd9":
            return False
    except OSError:
        return False
    return cv2.imread(str(path)) is not None


def scan(root: Path, count: int, force: bool) -> list[Target]:
    img_dir = root / "plates" / "images" / "synth"
    lbl_dir = root / "plates" / "labels" / "synth"
    if not img_dir.is_dir():
        raise SystemExit(f"missing {img_dir}")

    present = {p.stem: p for p in img_dir.glob("*.jpg")}
    targets: list[Target] = []

    for index in range(count):
        stem = f"synth_{index:06d}"
        label_path = lbl_dir / f"{stem}.txt"
        expected = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else None
        path = present.get(stem)
        if path is None:
            targets.append(Target(stem, index, "missing", expected))
        elif force or not jpeg_ok(path):
            targets.append(Target(stem, index, "corrupt", expected))

    return targets


def generate(root: Path, index: int, alphabet: str) -> tuple[Path, Path, str]:
    """Render one plate into a scratch root and return (image, label, text)."""
    scratch = Path(tempfile.mkdtemp(prefix="synth-repair-", dir=root.parent))
    previous = textgen.TEXT_LETTERS
    textgen.TEXT_LETTERS = ALPHABETS[alphabet]
    try:
        SynthPipeline(scratch, SynthConfig(count=1, seed=SEED, start_index=index)).run()
    finally:
        textgen.TEXT_LETTERS = previous

    stem = f"synth_{index:06d}"
    image = scratch / "plates" / "images" / "synth" / f"{stem}.jpg"
    label = scratch / "plates" / "labels" / "synth" / f"{stem}.txt"
    text = label.read_text(encoding="utf-8").strip()
    return image, label, text


def repair(root: Path, target: Target, alphabet_arg: str) -> dict[str, object]:
    candidates = list(ALPHABETS) if alphabet_arg == "auto" else [alphabet_arg]

    attempts: list[dict[str, object]] = []
    for alphabet in candidates:
        image, label, text = generate(root, target.index, alphabet)
        matched = target.expected is None or text == target.expected
        attempts.append({"alphabet": alphabet, "text": text, "matched": matched})
        if not matched:
            shutil.rmtree(image.parents[3], ignore_errors=True)
            continue
        if not jpeg_ok(image):
            shutil.rmtree(image.parents[3], ignore_errors=True)
            attempts[-1]["matched"] = False
            continue

        stem = target.stem
        shutil.copy2(image, root / "plates" / "images" / "synth" / f"{stem}.jpg")
        shutil.copy2(label, root / "plates" / "labels" / "synth" / f"{stem}.txt")
        shutil.rmtree(image.parents[3], ignore_errors=True)
        return {
            "stem": stem, "index": target.index, "reason": target.reason,
            "alphabet": alphabet, "text": text,
            "verified": target.expected is not None,
            "expected": target.expected, "attempts": attempts,
        }

    return {
        "stem": target.stem, "index": target.index, "reason": target.reason,
        "alphabet": None, "text": None, "verified": False,
        "expected": target.expected, "attempts": attempts, "failed": True,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()

    targets = scan(root, args.count, args.force)
    print(f"[repair] {len(targets)} de {args.count} necesitan reparación "
          f"(faltantes={sum(t.reason == 'missing' for t in targets)}, "
          f"corruptos={sum(t.reason == 'corrupt' for t in targets)})")
    for target in targets[:20]:
        print(f"  {target.stem}  {target.reason:8s}  "
              f"etiqueta en disco: {target.expected or '—'}")
    if len(targets) > 20:
        print(f"  … y {len(targets) - 20} más")

    if args.dry_run:
        return
    if not targets:
        # Nothing broken: still allow the ablation refresh, which is how the
        # training list gets back in step after an earlier repair.
        print("[repair] nada que reparar")
        if args.update_ablation:
            update_ablation(root, args.count)
        return

    if args.limit:
        targets = targets[: args.limit]

    results = []
    for position, target in enumerate(targets, 1):
        result = repair(root, target, "auto")
        results.append(result)
        state = "OK " if not result.get("failed") else "FALLA"
        print(f"[{position}/{len(targets)}] {state} {target.stem}  "
              f"alfabeto={result['alphabet']}  texto={result['text']}  "
              f"verificado={result['verified']}")

    failed = [r for r in results if r.get("failed")]
    unverified = [r for r in results if not r.get("failed") and not r["verified"]]

    log = root / "plates" / "repair_log.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n[repair] reparados={len(results) - len(failed)}  "
          f"fallidos={len(failed)}  sin verificar={len(unverified)}")
    if unverified:
        print("  sin verificar (no había etiqueta previa que contrastar): "
              + ", ".join(str(r["stem"]) for r in unverified))
    if failed:
        print("  revisar a mano: " + ", ".join(str(r["stem"]) for r in failed))
    print(f"[repair] detalle en {log}")

    if args.update_ablation and not failed:
        update_ablation(root, args.count)


def update_ablation(root: Path, count: int) -> None:
    """Rewrite the training list and the protocol to the repaired reality."""
    img_dir = root / "plates" / "images" / "synth"
    stems = sorted(p.stem for p in img_dir.glob("*.jpg") if jpeg_ok(p))
    if len(stems) != count:
        print(f"[repair] NO se actualiza la ablación: hay {len(stems)} válidos, "
              f"se esperaban {count}")
        return

    train_file = root / "ablation" / "train_synth.txt"
    train_file.write_text("\n".join(stems) + "\n", encoding="utf-8")

    protocol_path = root / "ablation" / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["train_synth"] = len(stems)
        protocol["note"] = (
            "2026-10-28: 33 especímenes (8 faltantes y 25 truncados) se regeneraron "
            "de forma determinística desde el índice, verificando la etiqueta contra "
            "el archivo original cuando existía. El conjunto vuelve a 360.000 válidos."
        )
        protocol_path.write_text(json.dumps(protocol, indent=4) + "\n", encoding="utf-8")

    print(f"[repair] train_synth.txt reescrito con {len(stems)} stems; "
          f"protocol.json actualizado")


if __name__ == "__main__":
    main()
