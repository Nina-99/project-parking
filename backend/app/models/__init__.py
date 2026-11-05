# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

from backend.app.models.authorized_vehicle import AuthorizedVehicle
from backend.app.models.camera import Camera
from backend.app.models.detection_event import DetectionEvent
from backend.app.models.parking_session import ParkingSession
from backend.app.models.user import User
from backend.app.models.vehicle import Vehicle

__all__ = [
    "AuthorizedVehicle",
    "Camera",
    "DetectionEvent",
    "ParkingSession",
    "User",
    "Vehicle",
]
