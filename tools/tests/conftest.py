# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Make the tool scripts importable when pytest runs from ``software/``.

The extractor tools live under ``tools/`` and import each other as
``plate_dataset.*`` (see the ``sys.path`` shim at the top of every entry
script). Tests need the same shim.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
