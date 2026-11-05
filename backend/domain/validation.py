# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The decisions an operator can take on a pending gate event.

They live in the domain because they are policy, not transport: the API, the
application use case and the database all have to agree on the same set.
"""
from __future__ import annotations

DECISIONS = (
    "confirmed",       # the machine's read is right
    "plate_corrected",  # the machine's read was wrong and the operator fixed it
    "dismissed",       # not actionable: a false positive, a vehicle that left
    "session_closed",  # an unmatched exit resolved by closing its session
)

DECISION_LABELS = {
    "confirmed": "Confirmado",
    "plate_corrected": "Placa corregida",
    "dismissed": "Descartado",
    "session_closed": "Sesión cerrada",
}
