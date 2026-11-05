# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Geometry helpers for zone-based vehicle reasoning.

Everything works in pixel coordinates. ``ZoneModel`` pre-computes the pixel
polygons for a camera at a given resolution and exposes the predicates the
tracker needs.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

Point = Sequence[float]
Polygon = Sequence[Point]
Box = Sequence[float]


def denormalize(polygon: Polygon, width: int, height: int) -> np.ndarray:
    """Convert normalized [0, 1] points to an integer pixel polygon."""
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2).copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    return np.round(pts).astype(np.int32)


def bottom_center(box: Box) -> Point:
    """Ground-contact point of a bbox: centre of the bottom edge.

    A vehicle touches the road at the bottom of its box, so this point (not the
    centroid) is the physically meaningful one for crossing a tripwire in
    perspective.
    """
    x1, _, x2, y2 = box[:4]
    return (float((x1 + x2) / 2.0), float(y2))


def point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def side_of_line(point: Point, line: Sequence[Point]) -> int:
    """Sign of the cross product of ``point`` relative to oriented ``a -> b``.

    Returns -1, 0 or 1. Sign 0 means the point lies on the line.
    """
    (ax, ay), (bx, by) = line[0], line[1]
    px, py = float(point[0]), float(point[1])
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    return int(np.sign(cross))


def polygon_centroid(polygon: np.ndarray) -> Point:
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))


def clamp_box(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))
    return x1, y1, x2, y2


def crop(image: np.ndarray, box: Box) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    return image[y1:y2, x1:x2]


class ZoneModel:
    """Pixel-space zones for one camera at one resolution."""

    def __init__(self, street_zone: Polygon, inside_zone: Polygon, tripwire: Polygon,
                 width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.street = denormalize(street_zone, width, height)
        self.inside = denormalize(inside_zone, width, height)
        self.tripwire = denormalize(tripwire, width, height)
        self.street_side = side_of_line(polygon_centroid(self.street), self.tripwire)
        self.inside_side = side_of_line(polygon_centroid(self.inside), self.tripwire)

    def in_street(self, point: Point) -> bool:
        return point_in_polygon(point, self.street)

    def in_inside(self, point: Point) -> bool:
        return point_in_polygon(point, self.inside)

    def side(self, point: Point) -> int:
        return side_of_line(point, self.tripwire)
