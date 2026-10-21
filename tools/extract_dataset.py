"""CLI entry point for the plate-dataset extractor.

Implementation lives in the ``plate_dataset`` package:

    python tools/extract_dataset.py --input /mnt/PARKING/ver/ \
        --output /mnt/Smart_Parking/dataset/ --camera channel16

The previous single-rectangle prototype is kept at
``tools/extract_dataset.py.orig`` for reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
