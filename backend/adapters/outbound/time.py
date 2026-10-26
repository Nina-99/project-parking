# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Wall clock behind the Clock port."""
from __future__ import annotations

from datetime import UTC, datetime


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)
