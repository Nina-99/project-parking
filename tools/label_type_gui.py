"""Interactive labeller for vehicle TYPE and COLOUR, with boxes and suggestions.

    uv run python tools/label_type_gui.py \
        --images /mnt/nina/SmartParking/entries_fast \
        --output /mnt/nina/SmartParking/car type \
        --detector yolo26s.pt \
        --classifier /mnt/nina/SmartParking/checkpoints/resnet18_type_best.pt \
        --resume

    # Later: re-check a 10% sample (intra-annotator agreement, type + colour)
    uv run python tools/label_type_gui.py --review \
        --output /mnt/nina/SmartParking/car type --review-sample 0.1

Per box the reviewer does two steps: TYPE (keys 1-6) then COLOUR. YOLO proposes
the boxes and a learned model suggests the class; the human accepts, corrects,
deletes or redraws. Every label saved here was seen by a person — *pre-labelling*,
not pseudo-labelling, which is what keeps the thesis dataset valid.

Each accepted box becomes an independent crop with two labels::

    images/<frame>_b<idx>.jpg
    labels/<frame>_b<idx>.txt         # type index,   "idx 0.5 0.5 1 1"
    labels_color/<frame>_b<idx>.txt   # colour index, "idx 0.5 0.5 1 1"
    data.yaml / data_color.yaml       # names, in label-index order
    labels.jsonl                      # audit: type, colour, confidences
    reviewed.jsonl                    # per-frame progress (resume)

A frame counts as done only when every accepted box has BOTH labels, so work done
before the colour pass existed is preserved and only completed.

Controls
    drag                draw a new box (classified on release)
    click inside box    select it
    right click box     delete it
    Tab                 next box (resets to the type step)
    Enter               type set? -> go to colour; both set? -> toggle accept
    1..6                type step: set the type (advances to colour)
    1..9, 0, q, w       colour step: set the colour and accept the box
    a                   accept every box that already has both labels
    k                   delete selected box
    u / Ctrl+Z          undo the last box/class/colour/accept action
    z                   cycle the crop zoom (1x, 1.5x, 2x, 3x)
    n / s               save this frame, mark done, go to next
    d                   skip for now (the frame returns next session)
    x / ESC             save and quit

Review mode (--review)
    p                   previous crop (undo an accidental advance)
    k                   delete the crop from the whole dataset (image + both labels)
    n / s               next crop · x / ESC save & quit
    (no detector or classifier is loaded; labels come from disk)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms
from ultralytics import YOLO

WINDOW = "type + colour labelling"
PANEL_W = 400
PANEL_H = 800
DEFAULT_FRAME_W = 1200
VEHICLE_CLASSES = (2, 3, 5, 7)
DEFAULT_TYPES = ("automovil", "camioneta", "furgon", "minibus", "moto", "vagoneta")
DEFAULT_COLORS = ("blanco", "gris", "plata", "negro", "rojo", "azul", "verde",
                  "amarillo", "marron", "beige", "naranja", "otro")
TYPE_KEYS = "123456"
COLOR_KEYS = "1234567890qw"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CROP_MEAN = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
CROP_STD = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)


def _prep(crop: np.ndarray) -> torch.Tensor:
    """BGR crop -> normalised (1,3,224,224) tensor, done by hand.

    torchvision's ToTensor refuses a tensor input in this version, and doing it
    explicitly keeps training and inference preprocessing identical anyway.
    """
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return ((tensor - CROP_MEAN) / CROP_STD).unsqueeze(0)


class TypeLabeller:
    def __init__(self, args) -> None:
        self.args = args
        self.output = Path(args.output)
        self.images_dir = self.output / "images"
        self.labels_dir = self.output / "labels"
        self.colors_dir = self.output / "labels_color"
        for folder in (self.images_dir, self.labels_dir, self.colors_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.manifest = self.output / "labels.jsonl"
        self.progress = self.output / "reviewed.jsonl"
        self.review_predictions: dict[str, tuple] = {}
        self.model_type = None
        self.model_color = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.review = bool(getattr(args, "review", False))
        self.crop_mode = bool(getattr(args, "queue", None))
        if not self.review and not self.crop_mode and not args.images:
            raise SystemExit(
                "[ERROR] one of --images, --review or --queue is required")

        if self.review:
            self.types = self._names_from(self.output / "data.yaml") or list(DEFAULT_TYPES)
            self.colors = self._names_from(self.output / "data_color.yaml") or list(DEFAULT_COLORS)
            self.detector = None
            # In review the models are reference only: they are shown, never written.
            self.type_model, self.type_classes = self._load_model(
                args.classifier, "type", DEFAULT_TYPES)
            self.color_model, self.color_classes = self._load_model(
                args.color_classifier, "colour", DEFAULT_COLORS)
            self.review_manifest = self.output / "review.jsonl"
            self.images = self._review_images()
            done = self._load_seen(self.review_manifest) if args.resume else set()
            if done:
                self.images = [p for p in self.images if p.stem not in done]
            print(f"[review] {len(self.images)} crop(s) to re-check"
                  f"{f' ({len(done)} done)' if done else ''}")
        elif self.crop_mode:
            # A crop is already the vehicle: no detector, no re-detection. The
            # classifiers still run so the labeller sees a suggestion, but the
            # suggestion is reference only and never written.
            self.types = self._names_from(self.output / "data.yaml") or list(DEFAULT_TYPES)
            self.colors = self._names_from(self.output / "data_color.yaml") or list(DEFAULT_COLORS)
            self.detector = None
            self.type_model, self.type_classes = self._load_model(
                args.classifier, "type", DEFAULT_TYPES)
            self.color_model, self.color_classes = self._load_model(
                args.color_classifier, "colour", DEFAULT_COLORS)
            self.images = queue_images(args.queue, args.dataset or Path("."))
            done = self._load_done(self.progress) if args.resume else set()
            if done:
                self.images = [p for p in self.images if p.stem not in done]
            print(f"[queue] {len(self.images)} crop(s) pending"
                  f"{f' ({len(done)} done)' if done else ''}")
        else:
            self.types = list(DEFAULT_TYPES)
            self.colors = list(DEFAULT_COLORS)
            self.detector = YOLO(args.detector)
            self.type_model, self.type_classes = self._load_model(args.classifier, "type", DEFAULT_TYPES)
            self.color_model, self.color_classes = self._load_model(
                args.color_classifier, "colour", DEFAULT_COLORS)
            self.images = sorted(
                p for p in Path(args.images).rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            done = set()
            if args.resume:
                done = self._load_done(self.progress)
            if done:
                self.images = [p for p in self.images if p.stem not in done]
            print(f"[label] {len(self.images)} frame(s) pending"
                  f"{f' ({len(done)} done)' if done else ''}")

        self.boxes: list[dict] = []
        self.selected = -1
        self.step = "type"
        self.zoom = 1.0
        self.frame_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.panning = False
        self.pan_start: tuple[int, int] | None = None
        self.pan_origin: tuple[float, float] | None = None
        self.frame = None
        self.display = None
        self.scale = 1.0
        self.frame_display_h = 0
        self.frame_max_w = max(400, int(getattr(args, "frame_width", DEFAULT_FRAME_W)))
        self.frame_index = 0
        self.frame_total = max(1, len(self.images))
        self.original_type: int | None = None
        self.original_color: int | None = None
        self.undo_stack: list[list[dict]] = []
        self.review_rows: list[dict] = []
        self.type_counts = self._count_labels(self.labels_dir, self.types)
        self.color_counts = self._count_labels(self.colors_dir, self.colors)
        self.drag_start: tuple[int, int] | None = None
        self.drag_now: tuple[int, int] | None = None
        self._write_yaml(self.output / "data.yaml", self.types)
        self._write_yaml(self.output / "data_color.yaml", self.colors)

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _names_from(path: Path) -> list[str] | None:
        if not path.exists():
            return None
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError):
            return None
        names = data.get("names") if isinstance(data, dict) else None
        return list(names) if names else None

    @staticmethod
    def _metrics_classes(checkpoint: Path | None) -> list[str] | None:
        if not checkpoint:
            return None
        # Prefer the metrics file that belongs to this checkpoint: with type and
        # colour checkpoints sharing a folder, a blind glob could return the
        # other model's class order.
        stem = checkpoint.stem
        prefix = stem[:-5] if stem.endswith("_best") else stem
        candidates = [checkpoint.parent / f"{prefix}_metrics.json"]
        candidates += sorted(checkpoint.parent.glob("*_metrics.json"))
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("classes"):
                return list(data["classes"])
        return None

    def _load_model(self, checkpoint: Path | None, label: str, fallback: tuple[str, ...]):
        if not checkpoint:
            print(f"[label] sin modelo de {label}: sin sugerencias de {label}")
            return None, None
        from torchvision import models
        classes = self._metrics_classes(checkpoint) or list(fallback)
        try:
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
            model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            model.to(self.device).eval()
            print(f"[label] modelo de {label}: {classes}")
            return model, classes
        except Exception as exc:
            print(f"[label] modelo de {label} no disponible ({exc})")
            return None, None

    @staticmethod
    def _write_yaml(path: Path, names: list[str]) -> None:
        names_str = ", ".join(repr(n) for n in names)
        path.write_text(
            "# Generated by label_type_gui.py\n"
            "# Run tools/resplit_by_track.py to create train/valid/test.\n"
            f"nc: {len(names)}\nnames: [{names_str}]\n", encoding="utf-8")

    @staticmethod
    def _count_labels(folder: Path, names: list[str]) -> dict[str, int]:
        counts = {name: 0 for name in names}
        for path in folder.glob("*.txt"):
            parts = path.read_text(encoding="utf-8").split()
            if parts and parts[0].isdigit() and int(parts[0]) < len(names):
                counts[names[int(parts[0])]] += 1
        return counts

    @staticmethod
    def _load_done(path: Path) -> set[str]:
        """Frames whose latest progress row is complete (type AND colour).

        Frame names contain dots (``C.OCCIDENTE_...``), so ``Path.stem`` would
        treat ``.OCCIDENTE...`` as an extension and collapse every frame to
        ``"C"``. Strip only a real image extension, never pathlib's suffix logic.
        """
        latest: dict[str, bool] = {}
        if not path.exists():
            return set()
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("frame") or row.get("image")
            if not isinstance(key, str):
                continue
            name = key.rsplit("/", 1)[-1]
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                name = name.rsplit(".", 1)[0]
            latest[name] = bool(row.get("complete", False))
        return {k for k, complete in latest.items() if complete}

    def _preset(self, frame_stem: str) -> tuple[dict[int, int], dict[int, int]]:
        """Existing labels for a frame, keyed by box index (for re-shows)."""
        types: dict[int, int] = {}
        colors: dict[int, int] = {}
        pattern = re.compile(rf"^{re.escape(frame_stem)}_b(\d+)")
        for path in self.labels_dir.glob(f"{frame_stem}_b*.txt"):
            match = pattern.match(path.stem)
            parts = path.read_text(encoding="utf-8").split()
            if match and parts:
                types[int(match.group(1))] = int(parts[0])
        for path in self.colors_dir.glob(f"{frame_stem}_b*.txt"):
            match = pattern.match(path.stem)
            parts = path.read_text(encoding="utf-8").split()
            if match and parts:
                colors[int(match.group(1))] = int(parts[0])
        return types, colors

    def _review_images(self) -> list[Path]:
        """Order review crops by model-vs-stored disagreement, then uncertainty.

        The model is reference only: predictions are cached for display and for
        ordering, never written. Disagreements come first because they are where
        a mis-keyed label most likely hides.
        """
        images = sorted(self.images_dir.glob("*.jpg"))
        if not images:
            return []

        stored: dict[str, tuple[int | None, int | None]] = {}
        preds: dict[str, tuple] = {}
        for path in images:
            stored[path.stem] = (
                self._read_label(self.labels_dir / f"{path.stem}.txt"),
                self._read_label(self.colors_dir / f"{path.stem}.txt"),
            )
            img = cv2.imread(str(path))
            if img is None:
                preds[path.stem] = (None, None)
                continue
            preds[path.stem] = (
                self._predict(self.type_model, self.type_classes, img),
                self._predict(self.color_model, self.color_classes, img),
            )
        self.review_predictions = preds

        def disagrees(path: Path) -> bool:
            t_pred, c_pred = preds.get(path.stem, (None, None))
            t_stored, c_stored = stored.get(path.stem, (None, None))
            t_bad = bool(t_pred and t_stored is not None
                         and self.types[t_stored] != t_pred[0])
            c_bad = bool(c_pred and c_stored is not None
                         and self.colors[c_stored] != c_pred[0])
            return t_bad or c_bad

        def confidence(path: Path) -> float:
            t_pred, c_pred = preds.get(path.stem, (None, None))
            values = [x[1] for x in (t_pred, c_pred) if x is not None]
            return min(values) if values else 1.0

        fraction = getattr(self.args, "review_sample", 1.0)
        if 0 < fraction < 1 and images:
            import random
            rng = random.Random(getattr(self.args, "seed", 7))
            images = rng.sample(images, max(1, round(len(images) * fraction)))
        images = sorted(images, key=lambda p: (not disagrees(p), confidence(p), p.name))
        n_bad = sum(1 for p in images if disagrees(p))
        print(f"[review] ordenados: {n_bad} discrepancias primero")
        return images

    def _push_undo(self) -> None:
        self.undo_stack.append([dict(item) for item in self.boxes])
        del self.undo_stack[:-100]

    def _undo(self) -> None:
        if self.undo_stack:
            self.boxes = self.undo_stack.pop()
            self.selected = min(self.selected, len(self.boxes) - 1)

    def _cycle_zoom(self) -> None:
        """Zoom the *display* 1x -> 2x -> 3x, centred on the selected box.

        Purely visual: crops are always taken from the original frame pixels, so
        saving never depends on the zoom level.
        """
        self.frame_zoom = {1.0: 2.0, 2.0: 3.0}.get(self.frame_zoom, 1.0)
        h, w = self.frame.shape[:2]
        if 0 <= self.selected < len(self.boxes):
            bx1, by1, bx2, by2 = self.boxes[self.selected]["box"]
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        else:
            cx, cy = w / 2.0, h / 2.0
        reg_w, reg_h = w / self.frame_zoom, h / self.frame_zoom
        self.pan_x = cx - reg_w / 2.0
        self.pan_y = cy - reg_h / 2.0

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def _predict(self, model, classes: list[str] | None, crop: np.ndarray):
        """Return ``(class_name, confidence)`` so the palette can be a superset."""
        if model is None or not classes or crop.size == 0:
            return None
        probs = model(_prep(crop).to(self.device)).softmax(1)[0]
        conf, idx = probs.max(0)
        return classes[int(idx)], float(conf)

    def _detect(self) -> list[dict]:
        if self.crop_mode:
            h, w = self.frame.shape[:2]
            crop = self.frame
            t = self._predict(self.type_model, self.type_classes, crop)
            c = self._predict(self.color_model, self.color_classes, crop)
            return [{
                "box": (0, 0, w, h),
                "cls": self.types.index(t[0]) if t and t[0] in self.types else None,
                "conf": t[1] if t else 0.0,
                "color": self.colors.index(c[0]) if c and c[0] in self.colors else None,
                "cconf": c[1] if c else 0.0,
                "accepted": False,
            }]
        result = self.detector.predict(
            self.frame, verbose=False, device=self.args.device,
            conf=self.args.conf, classes=list(VEHICLE_CLASSES), imgsz=640,
        )[0]
        boxes: list[dict] = []
        if result.boxes is None:
            return boxes
        for xyxy in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(self.frame.shape[1], x2)
            y2 = min(self.frame.shape[0], y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            crop = self.frame[y1:y2, x1:x2]
            t = self._predict(self.type_model, self.type_classes, crop)
            c = self._predict(self.color_model, self.color_classes, crop)
            boxes.append({
                "box": (x1, y1, x2, y2),
                "cls": self.types.index(t[0]) if t and t[0] in self.types else None,
                "conf": t[1] if t else 0.0,
                "color": self.colors.index(c[0]) if c and c[0] in self.colors else None,
                "cconf": c[1] if c else 0.0,
                "accepted": False,
            })
        return boxes

    # -- rendering ------------------------------------------------------------
    def _render(self) -> np.ndarray:
        h, w = self.frame.shape[:2]
        base = min(1.0, self.frame_max_w / float(w))
        view_w, view_h = int(w * base), int(h * base)
        z = max(1.0, self.frame_zoom)
        reg_w, reg_h = w / z, h / z
        self.pan_x = min(max(self.pan_x, 0.0), max(0.0, w - reg_w))
        self.pan_y = min(max(self.pan_y, 0.0), max(0.0, h - reg_h))
        x0, y0 = int(round(self.pan_x)), int(round(self.pan_y))
        x1r = min(w, x0 + int(round(reg_w)))
        y1r = min(h, y0 + int(round(reg_h)))
        region = self.frame[y0:y1r, x0:x1r]
        if region.size == 0:
            region = self.frame
        disp = cv2.resize(region, (view_w, view_h), interpolation=cv2.INTER_AREA)
        self.scale = view_w / max(1.0, reg_w)
        self.frame_display_h = disp.shape[0]

        def to_disp(px: float, py: float) -> tuple[int, int]:
            return (int((px - self.pan_x) * self.scale),
                    int((py - self.pan_y) * self.scale))

        for i, item in enumerate(self.boxes):
            bx1, by1, bx2, by2 = item["box"]
            p1 = to_disp(bx1, by1)
            p2 = to_disp(bx2, by2)
            done = item["cls"] is not None and item["color"] is not None
            color = (0, 200, 0) if item["accepted"] else ((180, 200, 0) if done else (0, 200, 255))
            cv2.rectangle(disp, p1, p2, color, 3 if i == self.selected else 1)
            tag = self.types[item["cls"]] if item["cls"] is not None else "?"
            if item["color"] is not None:
                tag += "/" + self.colors[item["color"]]
            cv2.putText(disp, tag, (p1[0] + 3, max(14, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        if self.drag_start and self.drag_now:
            cv2.rectangle(disp, to_disp(*self.drag_start), to_disp(*self.drag_now),
                          (255, 0, 255), 2)

        canvas_h = max(disp.shape[0], PANEL_H)
        frame_canvas = np.full((canvas_h, disp.shape[1], 3), 20, dtype=np.uint8)
        frame_canvas[:disp.shape[0]] = disp
        if self.review:
            bar = ("1-6 tipo | color 1-9,0,q,w | Tab paso | p atras | "
                   "k ELIMINAR del dataset | n/s siguiente | x salir")
        else:
            bar = ("Tab caja | tipo 1-6 | color 1-9,0,q,w | Enter | z zoom | a todo | "
                   "k borrar | u undo | n/s sig | d saltar | x salir")
        cv2.rectangle(frame_canvas, (0, canvas_h - 24), (disp.shape[1], canvas_h), (0, 0, 0), -1)
        cv2.putText(frame_canvas, bar, (8, canvas_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (235, 235, 235), 1, cv2.LINE_AA)

        panel = np.full((canvas_h, PANEL_W, 3), 30, dtype=np.uint8)
        y = 24

        def put(text: str, color=(220, 220, 220), scale: float = 0.45) -> None:
            nonlocal y
            cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, 1, cv2.LINE_AA)
            y += 19

        put(f"frame {self.frame_index + 1}/{self.frame_total}", (180, 220, 255))
        if self.review and self.frame_zoom > 1.0:
            put(f"zoom x{self.frame_zoom:g} (rueda media: mover)", (0, 220, 255), scale=0.4)
        if self.type_model is None and not self.review:
            put("SIN MODELO de TIPO", (0, 90, 255), scale=0.5)
        if self.color_model is None and not self.review:
            put("SIN MODELO de COLOR", (0, 90, 255), scale=0.5)

        if 0 <= self.selected < len(self.boxes):
            item = self.boxes[self.selected]
            x1, y1, x2, y2 = item["box"]
            crop = self.frame[y1:y2, x1:x2]
            if crop.size and not self.review:
                avail_h = int(canvas_h * 0.22)
                thumb = self._fit_crop(crop, PANEL_W - 20, avail_h, 1.0)
                panel[y:y + thumb.shape[0], 10:10 + thumb.shape[1]] = thumb
                y += thumb.shape[0] + 16
            paso = "TIPO" if self.step == "type" else "COLOR"
            put(f"caja {self.selected + 1}/{len(self.boxes)}  PASO: {paso}  "
                f"zoom x{self.frame_zoom:g}", (0, 220, 255), scale=0.48)
            tlabel = self.types[item["cls"]] if item["cls"] is not None else "-"
            clabel = self.colors[item["color"]] if item["color"] is not None else "-"
            if self.review:
                put(f"guardado: {tlabel} / {clabel}", (210, 220, 255))
                t_pred, c_pred = self.review_predictions.get(self.frame_stem, (None, None))
                if t_pred or c_pred:
                    pt = f"{t_pred[0]} ({t_pred[1]:.2f})" if t_pred else "-"
                    pc = f"{c_pred[0]} ({c_pred[1]:.2f})" if c_pred else "-"
                    put(f"modelo  : {pt} / {pc}", (0, 200, 255))
            else:
                put(f"tipo  : {tlabel}  ({item['conf']:.2f})")
                put(f"color : {clabel}  ({item['cconf']:.2f})")
            put(f"estado: {'ACEPTADO' if item['accepted'] else 'pendiente'}",
                (0, 220, 0) if item["accepted"] else (0, 200, 255))
        else:
            put("sin caja seleccionada")

        pending = sum(1 for b in self.boxes if not self._complete(b))
        put(f"cajas pendientes: {pending}",
            (0, 220, 0) if pending == 0 else (0, 200, 255), scale=0.48)

        if self.review and self.review_rows:
            n = len(self.review_rows)
            tm = sum(1 for r in self.review_rows if r["type_match"])
            cm = sum(1 for r in self.review_rows if r["color_match"])
            put(f"acuerdo tipo : {tm}/{n} = {100 * tm / n:.0f}%", (0, 220, 0))
            put(f"acuerdo color: {cm}/{n} = {100 * cm / n:.0f}%", (0, 220, 0))

        y += 4
        put("baul? -> automovil : vagoneta", (150, 200, 255), scale=0.4)
        if self.step == "type" or self.review:
            put("TIPO  (1-6)", (180, 180, 180), scale=0.42)
            for i, name in enumerate(self.types):
                cur = 0 <= self.selected < len(self.boxes) and self.boxes[self.selected]["cls"] == i
                put(f"{i + 1} {name} ({self.type_counts.get(name, 0)})",
                    (0, 230, 0) if cur else (220, 220, 220), scale=0.5 if cur else 0.43)
        if self.step == "color" or self.review:
            put("COLOR (1-9,0,q,w)", (180, 180, 180), scale=0.42)
            for i, name in enumerate(self.colors):
                cur = 0 <= self.selected < len(self.boxes) and self.boxes[self.selected]["color"] == i
                put(f"{COLOR_KEYS[i]} {name} ({self.color_counts.get(name, 0)})",
                    (0, 230, 0) if cur else (220, 220, 220), scale=0.5 if cur else 0.43)
        return np.hstack([frame_canvas, panel])

    @staticmethod
    def _fit_crop(crop: np.ndarray, avail_w: int, avail_h: int, zoom: float) -> np.ndarray:
        h, w = crop.shape[:2]
        scale = min(avail_w / w, avail_h / h) * zoom
        resized = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        canvas = np.full((avail_h, avail_w, 3), 30, dtype=np.uint8)
        y0 = max(0, (resized.shape[0] - avail_h) // 2)
        x0 = max(0, (resized.shape[1] - avail_w) // 2)
        part = resized[y0:y0 + avail_h, x0:x0 + avail_w]
        canvas[:part.shape[0], :part.shape[1]] = part
        return canvas

    # -- mouse ----------------------------------------------------------------
    def _on_mouse(self, event, x, y, flags, _param) -> None:
        if self.frame is None or x >= self.display.shape[1] - PANEL_W:
            return
        if y >= self.frame_display_h:
            return
        ix = int(self.pan_x + x / self.scale)
        iy = int(self.pan_y + y / self.scale)
        if event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start = (x, y)
            self.pan_origin = (self.pan_x, self.pan_y)
            return
        if event == cv2.EVENT_MOUSEMOVE and self.panning and self.pan_start:
            dx = (x - self.pan_start[0]) / self.scale
            dy = (y - self.pan_start[1]) / self.scale
            self.pan_x = self.pan_origin[0] - dx
            self.pan_y = self.pan_origin[1] - dy
            return
        if event == cv2.EVENT_MBUTTONUP:
            self.panning = False
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (ix, iy)
            self.drag_now = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_now = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x1, y1 = self.drag_start
            x2, y2 = ix, iy
            self.drag_start = self.drag_now = None
            if abs(x2 - x1) < 8 or abs(y2 - y1) < 8:
                self._select_at(ix, iy)
                return
            x1, x2 = sorted((max(0, x1), min(self.frame.shape[1], x2)))
            y1, y2 = sorted((max(0, y1), min(self.frame.shape[0], y2)))
            crop = self.frame[y1:y2, x1:x2]
            t = self._predict(self.type_model, self.type_classes, crop)
            c = self._predict(self.color_model, self.color_classes, crop)
            self._push_undo()
            self.boxes.append({
                "box": (x1, y1, x2, y2),
                "cls": self.types.index(t[0]) if t and t[0] in self.types else None,
                "conf": t[1] if t else 0.0,
                "color": self.colors.index(c[0]) if c and c[0] in self.colors else None,
                "cconf": c[1] if c else 0.0,
                "accepted": False,
            })
            self.selected = len(self.boxes) - 1
            self.step = "type"
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, item in enumerate(self.boxes):
                bx1, by1, bx2, by2 = item["box"]
                if bx1 <= ix <= bx2 and by1 <= iy <= by2:
                    self._push_undo()
                    del self.boxes[i]
                    self.selected = min(self.selected, len(self.boxes) - 1)
                    break

    def _select_at(self, ix: int, iy: int) -> None:
        for i, item in enumerate(self.boxes):
            x1, y1, x2, y2 = item["box"]
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                self.selected = i
                self.step = "type"
                return

    # -- persistence ----------------------------------------------------------
    def _complete(self, item: dict) -> bool:
        return item["cls"] is not None and item["color"] is not None

    def _clear_frame_crops(self, frame_stem: str) -> None:
        """Drop previous crops of this frame so re-saving never duplicates."""
        for folder in (self.images_dir, self.labels_dir, self.colors_dir):
            for path in folder.glob(f"{frame_stem}_b*"):
                try:
                    path.unlink()
                except OSError:
                    pass

    def _save_frame(self) -> None:
        savable = [b for b in self.boxes if b["cls"] is not None]
        if savable:
            self._clear_frame_crops(self.frame_stem)
        saved = 0
        with_color = 0
        for i, item in enumerate(self.boxes):
            if item["cls"] is None:
                continue  # box never labelled: not a sample
            x1, y1, x2, y2 = item["box"]
            crop = self.frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            name = f"{self.frame_stem}_b{i:02d}"
            cv2.imwrite(str(self.images_dir / f"{name}.jpg"), crop)
            (self.labels_dir / f"{name}.txt").write_text(
                f"{item['cls']} 0.5 0.5 1 1\n", encoding="utf-8")
            # Partial work is kept: the type label survives even if the colour is
            # still missing, so a re-show only asks for the colour.
            if item["color"] is not None:
                (self.colors_dir / f"{name}.txt").write_text(
                    f"{item['color']} 0.5 0.5 1 1\n", encoding="utf-8")
                with_color += 1
            with self.manifest.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "frame": self.frame_stem, "box": list(item["box"]),
                    "type": self.types[item["cls"]],
                    "color": self.colors[item["color"]] if item["color"] is not None else None,
                    "type_conf": round(item["conf"], 4),
                    "color_conf": round(item["cconf"], 4),
                    "image": f"images/{name}.jpg", "ts": datetime.now(UTC).isoformat(),
                }, ensure_ascii=False) + "\n")
            saved += 1
        self.type_counts = self._count_labels(self.labels_dir, self.types)
        self.color_counts = self._count_labels(self.colors_dir, self.colors)
        print(f"  [save] {self.frame_stem}: {saved} crop(s), {with_color} with colour",
              flush=True)

    def _mark_done(self, status: str) -> None:
        """A frame is complete only when every remaining box has type AND colour."""
        complete = all(self._complete(b) for b in self.boxes)
        saved = sum(1 for b in self.boxes if self._complete(b))
        row = {"frame": self.frame_stem, "status": status, "boxes": len(self.boxes),
               "saved": saved, "complete": complete, "ts": datetime.now(UTC).isoformat()}
        with self.progress.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- label loop -----------------------------------------------------------
    def run(self) -> None:
        if self.review:
            self.run_review()
            return
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        for n, path in enumerate(self.images, 1):
            self.frame = cv2.imread(str(path))
            if self.frame is None:
                continue
            self.frame_stem = path.stem
            self.frame_index = n - 1
            self.zoom = 1.0
            self.frame_zoom = 1.0
            self.pan_x = self.pan_y = 0.0
            self.undo_stack = []
            self.boxes = self._detect()
            preset_t, preset_c = self._preset(self.frame_stem)
            for i, item in enumerate(self.boxes):
                if i in preset_t:
                    item["cls"] = preset_t[i]
                    item["conf"] = 1.0
                if i in preset_c:
                    item["color"] = preset_c[i]
                    item["cconf"] = 1.0
                item["accepted"] = self._complete(item)
            self.selected = 0 if self.boxes else -1
            self.step = "type" if self.selected >= 0 and self.boxes[0]["cls"] is None else "color"
            print(f"[{n}/{len(self.images)}] {path.name}: {len(self.boxes)} box(es)", flush=True)

            while True:
                self.display = self._render()
                cv2.imshow(WINDOW, self.display)
                key = cv2.waitKey(20) & 0xFF
                if key == 255:
                    continue
                char = chr(key) if 0 <= key < 0x110000 else ""
                if key in (ord("x"), 27):
                    self._save_frame()
                    self._mark_done("quit")
                    cv2.destroyAllWindows()
                    print("[label] saved and quit")
                    return
                if key in (ord("n"), ord("s")):
                    self._save_frame()
                    self._mark_done("done")
                    break
                if key == ord("d"):
                    print(f"  [skip] {self.frame_stem}")
                    break
                if key in (ord("u"), 26):
                    self._undo()
                elif key == ord("z"):
                    self._cycle_zoom()
                elif key == 9 and self.boxes:  # Tab
                    self.selected = (self.selected + 1) % len(self.boxes)
                    self.step = "type"
                elif key == ord("a"):
                    self._push_undo()
                    for item in self.boxes:
                        item["accepted"] = self._complete(item)
                elif key == ord("k") and 0 <= self.selected < len(self.boxes):
                    self._push_undo()
                    del self.boxes[self.selected]
                    self.selected = min(self.selected, len(self.boxes) - 1)
                elif 0 <= self.selected < len(self.boxes) and char in TYPE_KEYS and self.step == "type":
                    self._push_undo()
                    self.boxes[self.selected]["cls"] = int(char) - 1
                    self.step = "color"
                elif 0 <= self.selected < len(self.boxes) and char in COLOR_KEYS and self.step == "color":
                    self._push_undo()
                    self.boxes[self.selected]["color"] = COLOR_KEYS.index(char)
                    self.boxes[self.selected]["accepted"] = True
                    if self.selected + 1 < len(self.boxes):
                        self.selected += 1
                    self.step = "type"
                elif key == 13 and 0 <= self.selected < len(self.boxes):
                    item = self.boxes[self.selected]
                    if self._complete(item):
                        self._push_undo()
                        item["accepted"] = not item["accepted"]
                    elif item["cls"] is not None:
                        self.step = "color"
        cv2.destroyAllWindows()
        print("[label] done")

    # -- review mode ----------------------------------------------------------
    @staticmethod
    def _load_seen(path: Path) -> set[str]:
        """Images already re-checked in review mode (any row counts)."""
        seen: set[str] = set()
        if not path.exists():
            return seen
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("image") or row.get("frame")
            if not isinstance(key, str):
                continue
            name = key.rsplit("/", 1)[-1]
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                name = name.rsplit(".", 1)[0]
            seen.add(name)
        return seen

    def _delete_crop(self, stem: str) -> None:
        """Remove a crop from the whole dataset: image, type label and colour label."""
        for folder, ext in ((self.images_dir, ".jpg"),
                            (self.labels_dir, ".txt"),
                            (self.colors_dir, ".txt")):
            target = folder / f"{stem}{ext}"
            if target.exists():
                try:
                    target.unlink()
                except OSError as exc:
                    print(f"[delete] could not remove {target.name}: {exc}")
        self.images = [p for p in self.images if p.stem != stem]
        with self.review_manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"image": stem, "deleted": True,
                                 "ts": datetime.now(UTC).isoformat()}) + "\n")
        print(f"  [delete] {stem} removed from the dataset", flush=True)

    def run_review(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        i = 0
        while 0 <= i < len(self.images):
            path = self.images[i]
            self.frame = cv2.imread(str(path))
            if self.frame is None:
                i += 1
                continue
            self.frame_stem = path.stem
            self.frame_index = i
            self.zoom = 1.0
            self.frame_zoom = 1.0
            self.pan_x = self.pan_y = 0.0
            self.undo_stack = []
            self.original_type = self._read_label(self.labels_dir / f"{path.stem}.txt")
            self.original_color = self._read_label(self.colors_dir / f"{path.stem}.txt")
            h, w = self.frame.shape[:2]
            self.boxes = [{"box": (0, 0, w, h), "cls": self.original_type, "conf": 1.0,
                           "color": self.original_color, "cconf": 1.0, "accepted": True}]
            self.selected = 0
            self.step = "type"
            print(f"[{i + 1}/{len(self.images)}] {path.name}", flush=True)
            action = "next"
            while True:
                self.display = self._render()
                cv2.imshow(WINDOW, self.display)
                key = cv2.waitKey(20) & 0xFF
                if key == 255:
                    continue
                char = chr(key) if 0 <= key < 0x110000 else ""
                if key in (ord("x"), 27):
                    self._save_review()
                    cv2.destroyAllWindows()
                    self._print_agreement()
                    return
                if key in (ord("n"), ord("s")):
                    self._save_review()
                    action = "next"
                    break
                if key == ord("p"):
                    action = "prev"
                    break
                if key == ord("k"):
                    self._delete_crop(path.stem)
                    action = "deleted"
                    break
                if key == ord("d"):
                    action = "next"
                    break
                if key == ord("z"):
                    self._cycle_zoom()
                elif key == 9:
                    self.step = "color" if self.step == "type" else "type"
                elif char in TYPE_KEYS and self.step == "type":
                    self._push_undo()
                    self.boxes[0]["cls"] = int(char) - 1
                    self.step = "color"
                elif char in COLOR_KEYS and self.step == "color":
                    self._push_undo()
                    self.boxes[0]["color"] = COLOR_KEYS.index(char)
                    self.step = "type"
                elif key in (ord("u"), 26):
                    self._undo()
            if action == "prev":
                i = max(0, i - 1)
            elif action == "deleted":
                pass  # the list shrank; same index now points to the next crop
            else:
                i += 1
        cv2.destroyAllWindows()
        self._print_agreement()

    @staticmethod
    def _read_label(path: Path) -> int | None:
        if not path.exists():
            return None
        parts = path.read_text(encoding="utf-8").split()
        return int(parts[0]) if parts and parts[0].isdigit() else None

    def _save_review(self) -> None:
        final_t = self.boxes[0]["cls"] if self.boxes else None
        final_c = self.boxes[0]["color"] if self.boxes else None
        changed = final_t != self.original_type or final_c != self.original_color
        if self.args.fix and changed:
            # Write the correction back so the dataset actually improves.
            if final_t is not None:
                (self.labels_dir / f"{self.frame_stem}.txt").write_text(
                    f"{final_t} 0.5 0.5 1 1\n", encoding="utf-8")
            if final_c is not None:
                (self.colors_dir / f"{self.frame_stem}.txt").write_text(
                    f"{final_c} 0.5 0.5 1 1\n", encoding="utf-8")
        row = {
            "image": self.frame_stem,
            "type_original": self.original_type, "type_final": final_t,
            "type_match": final_t == self.original_type,
            "color_original": self.original_color, "color_final": final_c,
            "color_match": final_c == self.original_color,
            "fixed": bool(self.args.fix and changed),
            "ts": datetime.now(UTC).isoformat(),
        }
        with self.review_manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.review_rows.append(row)

    def _print_agreement(self) -> None:
        rows: list[dict] = []
        if self.review_manifest.exists():
            for line in self.review_manifest.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            return
        n = len(rows)
        tm = sum(1 for r in rows if r.get("type_match"))
        cm = sum(1 for r in rows if r.get("color_match"))
        print(f"\n[review] acuerdo tipo : {tm}/{n} = {100 * tm / n:.1f}%")
        print(f"[review] acuerdo color: {cm}/{n} = {100 * cm / n:.1f}%")
        fixed = sum(1 for r in rows if r.get("fixed"))
        if fixed:
            print(f"[review] corregidos y guardados: {fixed}")
        for i, name in enumerate(self.types):
            sub = [r for r in rows if r.get("type_original") == i]
            if sub:
                h = sum(1 for r in sub if r.get("type_match"))
                print(f"  tipo {name:12s} {h}/{len(sub)} = {100 * h / len(sub):.0f}%")
        for i, name in enumerate(self.colors):
            sub = [r for r in rows if r.get("color_original") == i]
            if sub:
                h = sum(1 for r in sub if r.get("color_match"))
                print(f"  color {name:12s} {h}/{len(sub)} = {100 * h / len(sub):.0f}%")


def queue_images(queue_path: Path, base: Path) -> list[Path]:
    """Crop paths listed by ``queue_candidates.py``, resolved against ``base``."""
    images: list[Path] = []
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rel = row.get("image")
        if isinstance(rel, str) and rel:
            images.append(base / rel)
    return images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label vehicle type and colour.")
    parser.add_argument("--images", type=Path, help="Directory of frames (label mode).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", default="yolo26s.pt")
    parser.add_argument("--classifier", type=Path, help="ResNet18 type checkpoint.")
    parser.add_argument("--color-classifier", type=Path, help="ResNet18 colour checkpoint.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--review", action="store_true",
                        help="Re-label saved crops to measure intra-annotator agreement.")
    parser.add_argument("--queue", type=Path,
                        help="JSONL from queue_candidates.py: label pre-cropped candidates "
                             "instead of detecting vehicles in full frames.")
    parser.add_argument("--dataset", type=Path,
                        help="Dataset root the queue paths are relative to.")
    parser.add_argument("--fix", action="store_true",
                        help="With --review: write the corrected labels back to disk.")
    parser.add_argument("--review-sample", type=float, default=1.0,
                        help="Fraction of saved crops to re-check (e.g. 0.1 for 10%%).")
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_W,
                        help="Displayed frame width in pixels (total = this + panel).")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    TypeLabeller(args).run()


if __name__ == "__main__":
    main()
