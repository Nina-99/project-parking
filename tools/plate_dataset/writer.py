"""Dataset output: two label formats, a manifest and a review bucket.

Layout produced under ``root``::

    vehicles/images/<video>/<name>.jpg   # vehicle crop, plate visible
    vehicles/labels/<video>/<name>.txt   # YOLO bbox of the plate in the crop
    plates/images/<video>/<name>.jpg     # tight plate crop (OCR samples)
    plates/labels/<video>/<name>.txt     # plate text (OCR ground truth)
    review/...                           # pseudo-labels below the OCR threshold
    negatives/images/<video>/<name>.jpg  # passing vehicles (hard negatives)
    attributes/images/<video>/<name>.jpg # vehicle crop awaiting a human label
    attributes/manifest.jsonl            # queue of crops awaiting type + colour
    manifest.jsonl                       # one row per emitted sample
"""

from __future__ import annotations

import functools
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from plate_dataset.plate_ocr import PlateRead


def _resilient(func):
    """Do not let a filesystem hiccup abort a multi-hour batch.

    Removable NTFS/exFAT drives occasionally reject writes (e.g. EINVAL) until
    remounted; the sample is skipped with a warning instead of killing the run.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except OSError as exc:
            self.skips += 1
            print(f"[WARN] write skipped ({type(exc).__name__}: {exc})", flush=True)
            return None

    return wrapper


class DatasetWriter:
    def __init__(self, root: Path, save_negatives: bool = True) -> None:
        self.root = Path(root)
        self.save_negatives = save_negatives
        self.vehicles_images = self.root / "vehicles" / "images"
        self.vehicles_labels = self.root / "vehicles" / "labels"
        self.plates_images = self.root / "plates" / "images"
        self.plates_labels = self.root / "plates" / "labels"
        self.review_root = self.root / "review"
        self.negatives_images = self.root / "negatives" / "images"
        self.attr_images = self.root / "attributes" / "images"
        self.attr_manifest = self.root / "attributes" / "manifest.jsonl"
        self._manifest = self.root / "manifest.jsonl"
        self._initialized: set[Path] = set()
        self.skips = 0  # samples lost to filesystem write errors

    def _ensure(self, *dirs: Path) -> None:
        for d in dirs:
            if d not in self._initialized:
                d.mkdir(parents=True, exist_ok=True)
                self._initialized.add(d)

    @staticmethod
    def _stem(video: str, track_id: int, frame_idx: int) -> str:
        return f"{video}_track{track_id:04d}_f{frame_idx:07d}"

    @staticmethod
    def _jsonable(value):
        """Coerce numpy scalars so the manifest stays JSON-serialisable."""
        if isinstance(value, dict):
            return {k: DatasetWriter._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DatasetWriter._jsonable(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value

    def _log(self, row: dict, path: Path | None = None) -> None:
        row["ts"] = datetime.now(UTC).isoformat()
        target = path or self._manifest
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(self._jsonable(row), ensure_ascii=False) + "\n")

    @_resilient
    def write_entry(self, video: str, camera: str, track_id: int, frame_idx: int,
                    vehicle_crop: np.ndarray, plate: PlateRead, score: float,
                    review: bool, kind: str = "entry") -> None:
        stem = self._stem(video, track_id, frame_idx)
        crop_h, crop_w = vehicle_crop.shape[:2]
        x1, y1, x2, y2 = plate.bbox
        plate_crop = vehicle_crop[max(0, y1):y2, max(0, x1):x2]

        if review:
            img_dir = self.review_root / "vehicles" / "images" / video
            lbl_dir = self.review_root / "vehicles" / "labels" / video
            pl_img_dir = self.review_root / "plates" / "images" / video
            pl_lbl_dir = self.review_root / "plates" / "labels" / video
        else:
            img_dir = self.vehicles_images / video
            lbl_dir = self.vehicles_labels / video
            pl_img_dir = self.plates_images / video
            pl_lbl_dir = self.plates_labels / video

        self._ensure(img_dir, lbl_dir, pl_img_dir, pl_lbl_dir)
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), vehicle_crop)
        cv2.imwrite(str(pl_img_dir / f"{stem}.jpg"), plate_crop)

        cx = ((x1 + x2) / 2.0) / crop_w
        cy = ((y1 + y2) / 2.0) / crop_h
        bw = (x2 - x1) / float(crop_w)
        bh = (y2 - y1) / float(crop_h)
        (lbl_dir / f"{stem}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8"
        )
        (pl_lbl_dir / f"{stem}.txt").write_text(plate.text + "\n", encoding="utf-8")

        self._log({
            "kind": kind,
            "review": review,
            "camera": camera,
            "video": video,
            "track_id": track_id,
            "frame": frame_idx,
            "label": plate.text,
            "ocr_conf": round(plate.confidence, 4),
            "plate_sharpness": round(plate.sharpness, 2),
            "vehicle_score": round(score, 2),
            "vehicle_bbox": [0, 0, crop_w, crop_h],
            "plate_bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    @_resilient
    def write_unlabeled_entry(self, video: str, camera: str, track_id: int,
                              frame_idx: int, vehicle_crop: np.ndarray,
                              score: float, kind: str = "entry") -> None:
        """Gate event whose plate could not be read: keep it for manual labelling."""
        if vehicle_crop.size == 0:
            return
        stem = self._stem(video, track_id, frame_idx)
        img_dir = self.review_root / "vehicles" / "images" / video
        self._ensure(img_dir)
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), vehicle_crop)
        self._log({
            "kind": f"{kind}_no_plate",
            "review": True,
            "camera": camera,
            "video": video,
            "track_id": track_id,
            "frame": frame_idx,
            "vehicle_score": round(score, 2),
        })

    @_resilient
    def write_negative(self, video: str, camera: str, track_id: int, frame_idx: int,
                       vehicle_crop: np.ndarray, score: float) -> None:
        if not self.save_negatives or vehicle_crop.size == 0:
            return
        stem = self._stem(video, track_id, frame_idx)
        out_dir = self.negatives_images / video
        self._ensure(out_dir)
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), vehicle_crop)
        self._log({
            "kind": "pass",
            "camera": camera,
            "video": video,
            "track_id": track_id,
            "frame": frame_idx,
            "vehicle_score": round(score, 2),
        })

    @_resilient
    def write_attribute_sample(self, video: str, camera: str, track_id: int,
                               frame_idx: int, vehicle_crop: np.ndarray,
                               score: float, kind: str,
                               vehicle_class: int | None = None,
                               min_px: int = 48) -> None:
        """Queue one vehicle crop for human body-type and colour labelling.

        These attributes are *not* pseudo-labelled on purpose. An automatic
        labeller would give the classifier the answers of the very rule it is
        later compared against, turning that comparison into a measurement of
        imitation. The crop waits in ``attributes/`` until a human labels it
        with ``tools/label_attributes.py``.

        Passing vehicles are queued as well: they are still vehicles, so they
        are valid attribute samples even though they are negatives for the
        plate dataset.
        """
        if vehicle_crop.size == 0:
            return
        height, width = vehicle_crop.shape[:2]
        if min(height, width) < min_px:
            return
        stem = self._stem(video, track_id, frame_idx)
        img_dir = self.attr_images / video
        self._ensure(img_dir, self.attr_manifest.parent)
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), vehicle_crop)
        self._log({
            "stem": stem,
            "kind": kind,
            "camera": camera,
            "video": video,
            "track_id": track_id,
            "frame": frame_idx,
            "crop_hw": [height, width],
            "vehicle_score": round(score, 2),
            "vehicle_class": vehicle_class,
        }, path=self.attr_manifest)

    def close(self) -> None:
        return None
