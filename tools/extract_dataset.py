# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""CLI entry point for the plate-dataset extractor.

Implementation lives in the ``plate_dataset`` package; this only forwards the
arguments.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
