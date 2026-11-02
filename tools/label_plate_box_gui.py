"""Human review of proposed plate boxes inside vehicle crops.

    uv run python tools/label_plate_box_gui.py \
        --proposed /home/nina/FNI2025/ProjectParking/plates_det/proposed.jsonl \
        --output /home/nina/FNI2025/ProjectParking/plates_det/verified.jsonl

    uv run python tools/label_plate_box_gui.py --export-ocr \
        --output /home/nina/FNI2025/ProjectParking/plates_det/verified.jsonl \
        --crop-root /home/nina/FNI2025/ProjectParking/datasets/ocr_real_box \
        --text /home/nina/FNI2025/ProjectParking/plates_det/text.jsonl

Step 2 of three. ``pseudo_label_plates.py`` proposes, this tool lets a person
accept / correct / redraw / reject, and ``build_plate_detection_dataset.py``
exports.

Why a person: the thesis forbids training on automatics alone. PaddleOCR marks
*where* to look; the box that becomes a label was seen and approved by a human.
A rejected box is not wasted -- it becomes a negative, which is what stops the
detector from hallucinating a plate on every car.

The proposal usually clips the plate (Paddle's recogniser tightens to the glyphs
it managed to read). ``k`` throws it away and a drag draws a fresh one, so the
reviewer is never forced to start from a bad box.

Every frame also shows the *model view*: the current box cut out, passed through
the same CLAHE and resized to the 192x48 the CRNN actually consumes. The eye
reads plates at night that the network cannot, so reviewing without it is
deciding blind. What you see there is the truth; what you see in the photo is
not.

Step 4: ``--export-ocr`` turns the verified boxes into OCR training crops. The
crop convention is a parameter of the export, never a per-crop human judgement:

    box     the verified box, plus the same pad ``crop_plate`` applies at runtime
    inset   the verified box shrunk by a fixed fraction on each side

``box`` is the default because it is what the pipeline actually serves: the
exported crop and the served crop must describe the same pixels. ``inset`` is an
opt-in arm for framing sensitivity, not a glyph detector. Keep the fractions
small: on this repo's verified boxes the glyphs already fill most of the height,
so an aggressive trim amputates their tops and teaches the model a crop that
never occurs. The fractions land in the manifest, so a run is always replayable.

Controls
    drag                draw a new box from scratch (replaces the proposal)
    arrows              nudge the box 1 px (hold to repeat)
    + / -               grow / shrink 2 px per side
    k                   delete the box and start over
    u                   undo: restore the proposed box
    Enter / Space       accept the box on screen
    n                   there is no visible plate here -> negative
    s                   skip: leave this crop out of the dataset entirely
    q / Esc             quit and save what is done

Progress is appended to ``--output`` as it happens, so Ctrl-C and a power cut
cost at most the crop on screen.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from build_plate_detection_dataset import session_of

MAX_VIEW_W = 860
MAX_VIEW_H = 700

GREEN = (80, 220, 80)
RED = (60, 60, 240)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
PANEL_W = 400

# The CRNN's input, duplicated from backend/adapters/outbound/ml/plate_crnn.py.
# Training scripts must not import from the runtime, but the preview has to show
# the real geometry: a wrong height here makes the reviewer approve crops the
# network will see differently.
MODEL_W = 192
MODEL_H = 48
# The margin ``crop_plate`` adds around the detector box at runtime (4% x, 8% y).
# The preview and the export both use it, so what is reviewed is what is served.
PAD_X = 0.04
PAD_Y = 0.08
# Same parameters as the trainer: clipLimit 2.0, 4x4 tiles, on LAB's L channel.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

# Bumped when a crop convention changes meaning. Stored in every manifest row so
# a checkpoint can always be traced back to the crops that fed it.
EXPORT_VERSION = "ocr-crop-v1"

# A Bolivian plate is three or four digits followed by three letters. Sampling
# the number as an integer made leading zeros impossible; this makes a typo
# impossible to ship. Foreign plates and clipped crops land here too, which is
# deliberate: they are not training material for this alphabet.
PLATE_TEXT = re.compile(r"^(?:\d{3}|\d{4})[A-Z]{3}$")

Box = tuple[int, int, int, int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposed",
        type=Path,
        default=None,
        help="Proposals to review. Required unless --export-ocr.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-negatives",
        type=int,
        default=400,
        help="How many no-proposal crops to review as negatives. "
        "They are the hard negatives the detector needs; all "
        "4k would be a waste of a human's time.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Mirror the review file here every 25 crops. The "
        "review is the only irreplaceable artifact in this "
        "pipeline; PaddleOCR's proposals can be regenerated "
        "in minutes, a human's afternoon cannot.",
    )
    export = parser.add_argument_group(
        "export", "Headless crop export from the verified review file."
    )
    export.add_argument(
        "--export-ocr",
        action="store_true",
        help="Skip the GUI and export crops from --output instead.",
    )
    export.add_argument(
        "--crop-root",
        type=Path,
        default=None,
        help="Dataset root for the crops. Required with --export-ocr. Writes "
        "<root>/plates/{images,labels}/<session>/ plus <root>/manifest.jsonl.",
    )
    export.add_argument(
        "--conventions",
        default="box",
        help="Comma-separated crop conventions: box, inset. 'box' is the served "
        "framing; 'inset' is an opt-in sensitivity arm.",
    )
    export.add_argument(
        "--inset-x",
        type=float,
        default=0.05,
        help="Fraction of the box width trimmed per side for the inset convention. "
        "Large values amputate glyphs.",
    )
    export.add_argument(
        "--inset-y",
        type=float,
        default=0.08,
        help="Fraction of the box height trimmed per side for the inset convention. "
        "Large values amputate glyphs.",
    )
    export.add_argument(
        "--pad-x",
        type=float,
        default=PAD_X,
        help="Horizontal pad of the box convention, matching the runtime crop_plate (4%%).",
    )
    export.add_argument(
        "--pad-y",
        type=float,
        default=PAD_Y,
        help="Vertical pad of the box convention, matching the runtime crop_plate (8%%).",
    )
    export.add_argument(
        "--text",
        type=Path,
        default=None,
        help="text.jsonl from label_plate_text_gui.py, joined by stem.",
    )
    export.add_argument(
        "--allow-nonconforming",
        action="store_true",
        help="Keep labels that are not NNNLLL / NNNNLLL (foreign plates, half-visible "
        "crops). Off by default: they are not training material.",
    )
    export.add_argument("--force", action="store_true", help="Overwrite crops that already exist.")
    return parser


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def done_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["stem"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_verified_rows(path: Path) -> list[dict]:
    """Accepted boxes, last review per stem winning.

    A stem can be reviewed twice (a correction after a backup restore), and the
    review file is append-only by design. Taking the first row would export the
    superseded box; ``build_plate_detection_dataset`` resolves it the same way,
    so the detector and the OCR crops agree on which box is real.
    """
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[row["stem"]] = row
    return [
        row
        for row in latest.values()
        if row.get("box") and row.get("source") not in {"negative", "skipped"}
    ]


def clamp_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 2))
    y1 = max(0, min(y1, height - 2))
    x2 = max(x1 + 2, min(x2, width))
    y2 = max(y1 + 2, min(y2, height))
    return x1, y1, x2, y2


def model_view(image: np.ndarray) -> np.ndarray:
    """The 192x48 image the CRNN will actually consume, in BGR for display.

    Mirrors ``plate_crnn.preprocess`` up to the tensor step: CLAHE on LAB's L
    channel, then resize. The CLAHE is not cosmetic -- without it the real
    night crops are read at 14% character accuracy instead of ~68%.
    """
    if image.size == 0:
        return np.zeros((MODEL_H, MODEL_W, 3), dtype=np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.resize(corrected, (MODEL_W, MODEL_H), interpolation=cv2.INTER_CUBIC)


def crop_box(image: np.ndarray, box: Box) -> np.ndarray:
    """The pixels inside ``box``, clamped to the image, or an empty array."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)
    return image[y1:y2, x1:x2]


def inset_box(box: Box, frac_x: float, frac_y: float) -> Box:
    """Shrink ``box`` by a fraction of its own size on each side.

    A fixed relative inset, not a glyph detector: on the reviewed set the
    verified boxes carry a header strip and sometimes a bumper, and a uniform
    trim is reproducible where a threshold would drift with exposure. The
    fraction is recorded in the manifest so a run can be replayed exactly.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx = int(round(w * max(0.0, min(frac_x, 0.45))))
    dy = int(round(h * max(0.0, min(frac_y, 0.45))))
    return clamp_box((x1 + dx, y1 + dy, x2 - dx, y2 - dy), 10**6, 10**6)


def pad_box(box: Box, frac_x: float, frac_y: float) -> Box:
    """Grow ``box`` the way ``crop_plate`` does at runtime (4% x, 8% y)."""
    x1, y1, x2, y2 = box
    dx = max(1, int((x2 - x1) * frac_x))
    dy = max(1, int((y2 - y1) * frac_y))
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def crop_variants(
    image: np.ndarray,
    box: Box,
    *,
    conventions: tuple[str, ...] = ("box",),
    inset_x: float = 0.05,
    inset_y: float = 0.08,
    pad_x: float = PAD_X,
    pad_y: float = PAD_Y,
) -> dict[str, np.ndarray]:
    """Every crop convention derived from one verified box.

    One human annotation feeds all arms of the framing ablation: ``box`` is what
    the detector hands over, ``inset`` mimics a detector that trims to the glyph
    band. Variants that collapse to nothing are dropped, never written as a
    black sliver the network would learn from.
    """
    variants: dict[str, np.ndarray] = {}
    for convention in conventions:
        if convention == "box":
            crop = crop_box(image, pad_box(box, pad_x, pad_y))
        elif convention == "inset":
            crop = crop_box(image, inset_box(box, inset_x, inset_y))
        else:  # pragma: no cover - argparse restricts the choices
            raise ValueError(f"unknown crop convention: {convention}")
        if crop.size and crop.shape[0] >= 8 and crop.shape[1] >= 24:
            variants[convention] = crop
    return variants


def side_panel(crop: np.ndarray | None, height: int) -> np.ndarray:
    """Right-hand panel: the model view above, the cut-out below."""
    panel = np.zeros((height, PANEL_W, 3), dtype=np.uint8)
    cv2.putText(
        panel,
        f"MODEL VIEW {MODEL_W}x{MODEL_H} (CLAHE)",
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        GREY,
        1,
        cv2.LINE_AA,
    )
    view = model_view(crop) if crop is not None and crop.size else None
    if view is None:
        view = np.zeros((MODEL_H, MODEL_W, 3), dtype=np.uint8)
    shown = cv2.resize(
        view,
        (PANEL_W - 16, int(MODEL_H * (PANEL_W - 16) / MODEL_W)),
        interpolation=cv2.INTER_NEAREST,
    )
    panel[24 : 24 + shown.shape[0], 8 : 8 + shown.shape[1]] = shown

    top = 24 + shown.shape[0] + 22
    if crop is None or crop.size == 0:
        cv2.putText(panel, "no box", (8, top), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
        return panel

    cv2.putText(
        panel,
        "CUT-OUT (served framing: +4% x, +8% y)",
        (8, top - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        GREY,
        1,
        cv2.LINE_AA,
    )
    room_h = max(32, height - top - 10)
    scale = min((PANEL_W - 16) / crop.shape[1], room_h / crop.shape[0])
    zoomed = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    panel[top : top + zoomed.shape[0], 8 : 8 + zoomed.shape[1]] = zoomed
    return panel


def load_texts(path: Path | None) -> dict[str, str]:
    """Text labels by stem, last write winning, uppercased as the CRNN expects."""
    if path is None or not path.exists():
        return {}
    texts: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        text = str(record.get("text") or "").strip().upper()
        if text:
            texts[record["stem"]] = text
    return texts


def existing_convention(root: Path) -> str | None:
    """The convention already exported into ``root``, if any."""
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        return None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line).get("convention")
    return None


def export_ocr(args: argparse.Namespace) -> int:
    """Write OCR crops and labels in the layout the trainer actually reads.

    ``<root>/plates/{images,labels}/<session>/<stem>`` mirrors the extractor, so
    ``prepare_ocr_ablation.py`` can split it by recording session and
    ``PlateOcrDataset`` can load it without a translation layer.

    One convention per root: ``PlateOcrDataset`` matches images by stem, so two
    conventions under one root would silently double every sample with a
    different framing. Headless on purpose -- exporting is a build step.
    """
    rows = load_verified_rows(args.output)
    if not rows:
        raise SystemExit(f"no reviewed boxes in {args.output}")

    texts = load_texts(args.text)
    conventions = tuple(c.strip() for c in args.conventions.split(",") if c.strip())
    if len(conventions) != 1:
        raise SystemExit(
            "one convention per dataset root: run the export once per convention "
            "into its own --crop-root (e.g. ocr_real_box, ocr_real_inset)"
        )
    convention = conventions[0]

    root = args.crop_root
    previous = existing_convention(root)
    if previous is not None and previous != convention:
        raise SystemExit(
            f"{root} already holds '{previous}' crops; exporting '{convention}' "
            f"here would put two framings behind the same stems. Use another root."
        )

    image_root = root / "plates" / "images"
    label_root = root / "plates" / "labels"
    root.mkdir(parents=True, exist_ok=True)

    # Every file this run keeps. The manifest is the tool's record of what it
    # put on disk, so pruning is scoped to files the manifest already lists:
    # walking the whole root would delete anything another producer put there
    # (synthetic plates, crops copied in from another dataset root). The trainer
    # globs the directory, so a crop whose label failed validation must not
    # survive as an orphan next to valid samples.
    manifest_path = root / "manifest.jsonl"
    previous_rows = load_rows(manifest_path) if manifest_path.exists() else []
    kept: set[str] = set()

    written = 0
    no_text = 0
    unreadable = 0
    degenerate = 0
    write_failed = 0
    malformed: list[tuple[str, str]] = []
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for row in rows:
            text = texts.get(row["stem"])
            if not text:
                no_text += 1
                continue
            if not args.allow_nonconforming and not PLATE_TEXT.match(text):
                malformed.append((row["stem"], text))
                continue
            image = cv2.imread(row["image"])
            if image is None:
                unreadable += 1
                continue
            box = tuple(int(v) for v in row["box"])
            crop = crop_variants(
                image,
                box,
                conventions=(convention,),
                inset_x=args.inset_x,
                inset_y=args.inset_y,
                pad_x=args.pad_x,
                pad_y=args.pad_y,
            ).get(convention)
            if crop is None:
                degenerate += 1
                continue

            session = session_of(row["stem"])
            (image_root / session).mkdir(parents=True, exist_ok=True)
            (label_root / session).mkdir(parents=True, exist_ok=True)
            destination = image_root / session / f"{row['stem']}.jpg"
            # imwrite returns False instead of raising on a full disk or a bad
            # path. Writing the row anyway produced a manifest that claimed 698
            # crops while 673 existed, and nothing said so.
            if (args.force or not destination.exists()) and not cv2.imwrite(str(destination), crop):
                write_failed += 1
                continue
            label_path = label_root / session / f"{row['stem']}.txt"
            label_path.write_text(text + "\n", encoding="utf-8")
            crop_rel = str(destination.relative_to(root))
            label_rel = str(label_path.relative_to(root))
            kept.update({crop_rel, label_rel})
            manifest.write(
                json.dumps(
                    {
                        "stem": row["stem"],
                        "image": row["image"],
                        "box": list(box),
                        "source": row.get("source"),
                        "convention": convention,
                        "params": {
                            "inset_x": args.inset_x,
                            "inset_y": args.inset_y,
                            "pad_x": args.pad_x,
                            "pad_y": args.pad_y,
                        },
                        "crop": crop_rel,
                        "label": label_rel,
                        "session": session,
                        "width": int(crop.shape[1]),
                        "height": int(crop.shape[0]),
                        "text": text,
                        "version": EXPORT_VERSION,
                        "ts": datetime.now(UTC).isoformat(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    # The trainer globs the directory, not the manifest, so the directory must
    # equal the manifest. Walking it (instead of trusting the previous manifest)
    # also cleans orphans left by a run that predates the pruning.
    pruned = 0
    for old in previous_rows:
        for key in ("crop", "label"):
            relative = old.get(key)
            if not relative or relative in kept or "synth" in Path(relative).parts:
                continue
            stale = root / relative
            if stale.is_file():
                stale.unlink()
                pruned += 1
    for directory in sorted(root.glob("plates/*/*")):
        if directory.name == "synth" or not directory.is_dir():
            continue
        if not any(directory.iterdir()):
            directory.rmdir()

    print(f"[export] {written} crops ({convention}) -> {root}")
    if pruned:
        print(
            f"[export] {pruned} stale crop/label files removed: the directory is "
            f"the dataset, not the manifest"
        )
    if no_text:
        print(
            f"[export] {no_text} reviewed boxes have no text label yet; a crop "
            f"without a label teaches CTC to emit nothing, so they stay out "
            f"(pass --text to join label_plate_text_gui.py's output)"
        )
    if unreadable:
        print(f"[export] {unreadable} rows skipped: image missing or unreadable")
    if degenerate:
        print(f"[export] {degenerate} rows skipped: box too small to crop")
    if write_failed:
        print(
            f"[export] {write_failed} crops could not be written (disk or path); "
            f"no label or manifest row was kept for them"
        )
    if malformed:
        print(
            f"[export] {len(malformed)} labels are not NNNLLL / NNNNLLL and stayed "
            f"out; fix them at the source, do not guess:"
        )
        for stem, text in malformed[:15]:
            print(f"[export]    {text!r:12} {stem}")
        if len(malformed) > 15:
            print(f"[export]    ... and {len(malformed) - 15} more")
    return written


class Reviewer:
    def __init__(self, output: Path, backup_dir: Path | None = None) -> None:
        self.output = output
        self.backup_dir = backup_dir
        self.saved = 0
        # Working state, always in IMAGE coordinates. Mixing view and image
        # coordinates here is how a corrected box ends up three times too large.
        self.box: Box | None = None
        self.proposal: Box | None = None
        self.box_source = "proposal"
        self.scale = 1.0
        self.image_size = (0, 0)
        self.drag_start: tuple[int, int] | None = None

    # ---------- coordinates ----------

    def view(self, box: Box) -> Box:
        s = self.scale
        return (int(box[0] * s), int(box[1] * s), int(box[2] * s), int(box[3] * s))

    def image(self, x: float, y: float) -> tuple[int, int]:
        s = self.scale or 1.0
        return int(round(x / s)), int(round(y / s))

    # ---------- state ----------

    def start_row(self, row: dict) -> None:
        proposal = tuple(row["box"]) if row.get("box") else None
        self.proposal = (
            (int(proposal[0]), int(proposal[1]), int(proposal[2]), int(proposal[3]))
            if proposal
            else None
        )
        self.box = self.proposal
        self.box_source = "proposal" if self.proposal else "drawn"
        self.drag_start = None

    def adjust(self, dx: int, dy: int, width: int, height: int) -> None:
        if self.box is None:
            return
        x1, y1, x2, y2 = self.box
        self.box = clamp_box((x1 + dx, y1 + dy, x2 + dx, y2 + dy), width, height)
        self.box_source = "drawn"

    def grow(self, delta: int, width: int, height: int) -> None:
        if self.box is None:
            return
        x1, y1, x2, y2 = self.box
        self.box = clamp_box((x1 - delta, y1 - delta, x2 + delta, y2 + delta), width, height)
        self.box_source = "drawn"

    # ---------- mouse ----------

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.box = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            x0, y0 = self.drag_start
            ix1, iy1 = self.image(min(x0, x), min(y0, y))
            ix2, iy2 = self.image(max(x0, x), max(y0, y))
            self.box = clamp_box((ix1, iy1, ix2, iy2), *self.image_size)
            self.box_source = "drawn"
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x0, y0 = self.drag_start
            if abs(x - x0) <= 4 or abs(y - y0) <= 4:
                self.box = None  # a click, not a drag: no box
            self.drag_start = None

    # ---------- io ----------

    def save(self, row: dict, source: str) -> None:
        record = {
            "stem": row["stem"],
            "image": row["image"],
            "box": list(self.box) if self.box else None,
            "source": source,
        }
        if self.box:
            record["w"] = self.box[2] - self.box[0]
            record["h"] = self.box[3] - self.box[1]
        with self.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.saved += 1
        if self.backup_dir is not None and self.saved % 25 == 0:
            self.backup()

    def backup(self) -> None:
        """Copy the review file aside. Called periodically and on exit."""
        if self.backup_dir is None or not self.output.exists():
            return
        import shutil

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.output, self.backup_dir / self.output.name)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.export_ocr:
        if args.crop_root is None:
            raise SystemExit("--export-ocr needs --crop-root")
        export_ocr(args)
        return
    if args.proposed is None:
        raise SystemExit("--proposed is required to review boxes")

    rows = load_rows(args.proposed)
    if not rows:
        raise SystemExit(f"no rows in {args.proposed}")

    finished = done_stems(args.output)
    proposals = [r for r in rows if r.get("box") and r["stem"] not in finished]
    negatives = [r for r in rows if not r.get("box") and r["stem"] not in finished]

    rng = random.Random(args.seed)
    rng.shuffle(negatives)
    queue = proposals + negatives[: args.sample_negatives]

    print(
        f"[review] {len(proposals)} proposals + "
        f"{min(len(negatives), args.sample_negatives)} negatives = "
        f"{len(queue)} crops to review",
        flush=True,
    )
    if not queue:
        print("[review] nothing left; every crop in this file is already reviewed")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reviewer = Reviewer(args.output, args.backup_dir)

    window = "plate box review"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, reviewer.on_mouse)

    index = 0
    current_stem: str | None = None
    base_view = None
    while index < len(queue):
        row = queue[index]

        # Re-render the crop only when the row changes. Doing it every frame
        # would reset the working box 50 times a second, and re-read the JPEG.
        if row["stem"] != current_stem:
            image = cv2.imread(row["image"])
            if image is None:
                index += 1
                continue
            height, width = image.shape[:2]
            reviewer.image_size = (width, height)
            reviewer.scale = min(MAX_VIEW_W / width, MAX_VIEW_H / height, 3.0)
            reviewer.start_row(row)
            base_view = cv2.resize(
                image, None, fx=reviewer.scale, fy=reviewer.scale, interpolation=cv2.INTER_CUBIC
            )
            current_stem = row["stem"]

        height, width = reviewer.image_size
        view = base_view.copy()

        cut: np.ndarray | None = None
        if reviewer.box is not None:
            x1, y1, x2, y2 = reviewer.view(reviewer.box)
            colour = GREEN if reviewer.box_source == "proposal" else RED
            cv2.rectangle(view, (x1, y1), (x2, y2), colour, 2)
            label = "plate" if reviewer.box_source == "proposal" else "corrected"
            cv2.putText(
                view,
                label,
                (x1, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
                cv2.LINE_AA,
            )
            cut = crop_box(image, pad_box(reviewer.box, PAD_X, PAD_Y))

        frame = np.hstack([view, side_panel(cut, view.shape[0])])
        header = np.zeros((46, frame.shape[1], 3), dtype=np.uint8)
        kind = "PROPOSAL" if reviewer.proposal else "NO PLATE FOUND"
        cv2.putText(
            header,
            f"{index + 1}/{len(queue)}  {kind}  saved={reviewer.saved}",
            (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            "drag=new  arrows=nudge  +/-=grow  k=delete  u=undo  Enter=ok  n=none  s=skip  q=quit",
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            GREY,
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window, np.vstack([header, frame]))
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue

        if key in (13, 32):  # Enter / Space: accept
            if reviewer.box is not None:
                reviewer.save(row, reviewer.box_source)
                index += 1
        elif key == ord("n"):
            reviewer.box = None
            reviewer.save(row, "negative")
            index += 1
        elif key == ord("s"):
            reviewer.save(row, "skipped")
            index += 1
        elif key == ord("k"):
            reviewer.box = None
            reviewer.box_source = "drawn"
        elif key == ord("u"):
            reviewer.box = reviewer.proposal
            reviewer.box_source = "proposal"
        elif key in (ord("+"), ord("=")):
            reviewer.grow(2, width, height)
        elif key in (ord("-"), ord("_")):
            reviewer.grow(-2, width, height)
        elif key in (81, 2424832):  # left
            reviewer.adjust(-1, 0, width, height)
        elif key in (83, 2555904):  # right
            reviewer.adjust(1, 0, width, height)
        elif key in (82, 2490368):  # up
            reviewer.adjust(0, -1, width, height)
        elif key in (84, 2621440):  # down
            reviewer.adjust(0, 1, width, height)
        elif key in (ord("q"), 27):
            break

    cv2.destroyWindow(window)
    reviewer.backup()
    print(f"[review] saved {reviewer.saved} crops this session -> {args.output}")
    if args.backup_dir is not None:
        print(f"[review] backup -> {args.backup_dir / args.output.name}")


if __name__ == "__main__":
    main()
