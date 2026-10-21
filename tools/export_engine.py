"""Export YOLO weights to a TensorRT engine.

    uv run python tools/export_engine.py --model yolo26x.pt

Produces ``yolo26x.engine`` next to the weights. After that, every tool picks
it up automatically (see ``resolve_weights``), so nothing else needs changing.
Engines are GPU-specific: re-export if you move to another GPU or driver.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export YOLO weights to TensorRT.")
    parser.add_argument("--model", help="Weights to export (defaults to config model_name).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    parser.add_argument("--device", default="0", help="GPU id used to build the engine.")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16.")
    args = parser.parse_args(argv)

    _, glob = load_config(DEFAULT_CONFIG_PATH)
    model_name = args.model or glob.model_name
    if not Path(model_name).exists():
        print(f"[ERROR] weights not found: {model_name} (run from the software/ directory)")
        return

    print(f"Exporting {model_name} -> TensorRT (imgsz={args.imgsz}, half={not args.fp32})")
    # Requires onnx-ir>=1.0 (torch>=2.12's exporter uses onnx_ir.schemas).
    engine = YOLO(model_name).export(
        format="engine", imgsz=args.imgsz, half=not args.fp32, device=args.device,
    )
    print(f"Engine written: {engine}")
    print("The tools now use it automatically.")


if __name__ == "__main__":
    main()
