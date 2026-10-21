"""Configuration model for the plate-dataset extractor.

Zones are polygons in normalized coordinates ([0, 1] relative to the frame) so
they stay valid across resolutions. This replaces the old single-rectangle
``roi_config.json``, which could not separate street traffic from vehicles
entering the premises: both occupy the same image band, so position alone is
not a valid discriminator.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from plate_dataset.attributes import DEFAULT_COLORS, DEFAULT_TYPES

Point = tuple[float, float]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}

# software/zones_config.json (config.py lives in software/tools/plate_dataset/)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "zones_config.json"


@dataclass
class CameraConfig:
    """Per-camera geometry and detection thresholds.

    ``street_zone`` is the polygon covering the public road. ``inside_zone`` is
    the polygon strictly beyond the gate (the premises apron/ramp). A vehicle
    whose ground-contact point reaches ``inside_zone`` has entered. ``tripwire``
    is an oriented segment separating the two, used to detect the crossing.
    """

    name: str
    street_zone: list[Point] = field(default_factory=list)
    inside_zone: list[Point] = field(default_factory=list)
    tripwire: list[Point] = field(default_factory=list)
    # Normalized boxes (x1, y1, x2, y2) blacked out before OCR: the recorder
    # burns a channel label and a timestamp into the frame, and PaddleOCR
    # happily reads them as plates ("Channel16").
    osd_zones: list[tuple[float, float, float, float]] = field(default_factory=list)
    vehicle_conf: float = 0.35
    min_area_pixels: int = 3000
    vehicle_classes: tuple[int, ...] = (2, 3, 5, 7)


@dataclass
class GlobalConfig:
    """Sampling, tracking and output thresholds shared by every camera."""

    # Pass 1 (sparse scan to locate activity windows).
    scan_stride: int = 12
    motion_downscale: float = 0.25
    motion_pixel_threshold: int = 25
    motion_threshold: float = 0.008
    segment_pad_frames: int = 45
    segment_merge_gap: int = 60

    # Pass 2 (dense processing of each window).
    dense_stride: int = 1
    grace_frames: int = 30
    min_track_hits: int = 3
    top_k_frames: int = 10

    # Plate pre-labelling (PaddleOCR).
    ocr_min_conf: float = 0.60
    min_plate_px: int = 40
    min_plate_sharpness: float = 15.0
    plate_y_min: float = 0.55
    plate_y_max: float = 1.00

    # Attribute labelling (body type + predominant colour). The vocabulary is
    # site-specific: it must match the vehicles actually registered in the
    # recinto, not the classes of a public dataset. These names become
    # directory names when the set is exported for training, so renaming a
    # class after labelling invalidates every label already written.
    vehicle_types: list[str] = field(default_factory=lambda: list(DEFAULT_TYPES))
    vehicle_colors: list[str] = field(default_factory=lambda: list(DEFAULT_COLORS))
    # Crops below this many pixels on the shortest edge carry too little shape
    # and paint to judge, so they are dropped instead of queued for labelling.
    attribute_min_px: int = 48

    # Output.
    save_negatives: bool = True
    model_name: str = "yolo26x.pt"
    half: bool = True  # request FP16 inference (maps to quantize=16 for .pt)
    device: str | int = 0
    tracker: str = "bytetrack.yaml"
    input_dir: str = "/mnt/nina/PARKING/ver/"
    output_dir: str = "/mnt/nina/Smart_Parking/dataset/"


def default_cameras() -> dict[str, CameraConfig]:
    """Starter zones derived from the two sampled access cameras.

    These are deliberately rough starting points. Use the trajectory viewer to
    refine them against real footage before trusting the output.
    """
    osd = [(0.00, 0.88, 0.24, 1.00), (0.68, 0.00, 1.00, 0.07)]
    return {
        "channel16": CameraConfig(
            name="channel16",
            street_zone=[(0.02, 0.08), (0.98, 0.08), (0.98, 0.50), (0.02, 0.50)],
            inside_zone=[(0.05, 0.53), (0.95, 0.53), (0.95, 1.00), (0.05, 1.00)],
            tripwire=[(0.02, 0.515), (0.98, 0.515)],
            osd_zones=list(osd),
        ),
        "channel6": CameraConfig(
            name="channel6",
            # Street traffic reaches y~0.40; keep a buffer below it so a passing
            # vehicle cannot touch inside_zone and be misread as an entry.
            street_zone=[(0.03, 0.04), (0.97, 0.04), (0.97, 0.42), (0.03, 0.42)],
            inside_zone=[(0.08, 0.48), (0.92, 0.48), (0.92, 1.00), (0.08, 1.00)],
            tripwire=[(0.05, 0.46), (0.95, 0.46)],
            osd_zones=list(osd),
        ),
    }


def _coerce_points(raw: list) -> list[Point]:
    return [(float(p[0]), float(p[1])) for p in raw]


def _coerce_boxes(raw: list) -> list[tuple[float, float, float, float]]:
    return [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in raw]


_POINT_RE = re.compile(r"\[\s*\n\s*(-?\d+(?:\.\d+)?),\s*\n\s*(-?\d+(?:\.\d+)?)\s*\n\s*\]")


def _compact_points(text: str) -> str:
    """Collapse ``[x, y]`` pairs onto a single line for hand-editing."""
    return _POINT_RE.sub(lambda m: f"[{m.group(1)}, {m.group(2)}]", text)


def write_template(path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Write a template config so the file can be edited by hand."""
    payload = {
        "cameras": {name: asdict(cam) for name, cam in default_cameras().items()},
        "global": asdict(GlobalConfig()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _compact_points(json.dumps(payload, indent=4))
    path.write_text(text + "\n", encoding="utf-8")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> tuple[dict[str, CameraConfig], GlobalConfig]:
    """Load camera zones and global settings, falling back to defaults."""
    if not Path(path).exists():
        write_template(path)
        return default_cameras(), GlobalConfig()

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    cameras: dict[str, CameraConfig] = {}
    for name, cam in data.get("cameras", {}).items():
        cameras[name] = CameraConfig(
            name=name,
            street_zone=_coerce_points(cam.get("street_zone", [])),
            inside_zone=_coerce_points(cam.get("inside_zone", [])),
            tripwire=_coerce_points(cam.get("tripwire", [])),
            osd_zones=_coerce_boxes(cam.get("osd_zones", [])),
            vehicle_conf=float(cam.get("vehicle_conf", 0.35)),
            min_area_pixels=int(cam.get("min_area_pixels", 3000)),
            vehicle_classes=tuple(cam.get("vehicle_classes", (2, 3, 5, 7))),
        )
    if not cameras:
        cameras = default_cameras()

    raw_global = data.get("global", {})
    known = {k: v for k, v in raw_global.items() if k in GlobalConfig.__dataclass_fields__}
    glob = GlobalConfig(**known)
    # An empty or missing vocabulary would leave the labeller with no keys.
    if not glob.vehicle_types:
        glob.vehicle_types = list(DEFAULT_TYPES)
    if not glob.vehicle_colors:
        glob.vehicle_colors = list(DEFAULT_COLORS)

    return cameras, glob


def infer_kwargs(glob: GlobalConfig) -> dict:
    """Extra predict/track kwargs for FP16.

    Current ultralytics deprecated ``half`` in favour of ``quantize=16``.
    TensorRT engines are already quantized, so ``half`` is turned off for them
    at load time; this only affects plain .pt weights.
    """
    return {"quantize": 16} if glob.half else {}


def resolve_weights(model_name: str) -> str:
    """Prefer a TensorRT engine next to the .pt weights when one exists.

    Export once (``tools/export_engine.py``) and every tool picks up the faster
    engine automatically; falling back to the .pt keeps things working on
    machines without TensorRT.
    """
    path = Path(model_name)
    engine = path.with_suffix(".engine")
    if path.suffix == ".pt" and engine.exists():
        return str(engine)
    return model_name


def list_videos(input_dir: Path) -> list[Path]:
    """Sorted list of video files in a directory (empty if it does not exist).

    Unreadable entries are skipped with a warning: a corrupt file on a flaky
    NTFS/exFAT volume must not abort the whole batch.
    """
    if not Path(input_dir).exists():
        return []
    try:
        entries = list(Path(input_dir).iterdir())
    except OSError as exc:
        print(f"[WARN] cannot list {input_dir}: {exc}", flush=True)
        return []
    found: list[Path] = []
    for path in entries:
        try:
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                found.append(path)
        except OSError as exc:
            print(f"[WARN] skipping unreadable entry {path.name}: {exc}", flush=True)
    return sorted(found)


def infer_camera(video_name: str, cameras: dict[str, CameraConfig]) -> CameraConfig:
    """Pick a camera by matching its key as a substring of the video filename.

    Falls back to the first configured camera when nothing matches.
    """
    lowered = video_name.lower()
    for name, cam in cameras.items():
        key = name.lower()
        # "channel16" -> also match the "ch16" token used in recorder filenames
        if key in lowered or key.replace("channel", "ch") in lowered:
            return cam
    return next(iter(cameras.values()))
