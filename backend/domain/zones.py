# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Where the access is, and whether a vehicle reached it.

The worker used to accept every vehicle the detector confirmed anywhere in the
frame. At a gate whose field of view includes the street, that turns passing
traffic into entries: the measured hour produced 129 sessions against 7 vehicles
that actually crossed. A session is a business fact, so the runtime needs the
same rule the offline extractor already used -- the vehicle counts when its
ground-contact point reaches the apron beyond the gate.

Geometry only: no OpenCV, no database. The polygons are normalised to the frame,
exactly as ``zones_config.json`` stores them, and converted to pixels here so a
resolution change does not need a new configuration file.
"""
from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]
Box = tuple[int, int, int, int]

# Samples a vehicle must stay on the far side of the line before the crossing
# counts. A vehicle driving past the threshold can graze it once; one that
# crosses stays on the other side.
DEFAULT_CROSSING_HITS = 2


def bottom_centre(box: Box) -> Point:
    """The point where the vehicle meets the ground, not the box's middle.

    The middle of the box is the windscreen on a car facing the camera; the
    bottom edge is where the tyres are, and that is what crosses a line drawn
    on the pavement.
    """
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray casting. Points on the boundary are not guaranteed either way."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing:
                inside = not inside
        j = i
    return inside


def side_of_line(point: Point, line: list[Point]) -> int:
    """+1 or -1 depending on which side of the segment the point falls."""
    if len(line) < 2:
        return 0
    (x1, y1), (x2, y2) = line[0], line[1]
    cross = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def centroid(polygon: list[Point]) -> Point:
    """Mean of the vertices: enough to say which side a zone sits on."""
    if not polygon:
        return (0.0, 0.0)
    return (sum(p[0] for p in polygon) / len(polygon),
            sum(p[1] for p in polygon) / len(polygon))


@dataclass(frozen=True)
class ZoneGate:
    """The access of one camera at one resolution."""

    inside: tuple[Point, ...]
    tripwire: tuple[Point, ...]
    width: int
    height: int

    @classmethod
    def from_normalised(cls, inside: list[Point], tripwire: list[Point],
                        width: int, height: int) -> ZoneGate:
        return cls(inside=tuple(inside), tripwire=tuple(tripwire),
                   width=width, height=height)

    def to_pixel(self, point: Point) -> Point:
        return (point[0] * self.width, point[1] * self.height)

    def in_inside(self, box: Box) -> bool:
        return point_in_polygon(bottom_centre(box),
                                [self.to_pixel(p) for p in self.inside])

    def side(self, box: Box) -> int:
        return side_of_line(bottom_centre(box),
                            [self.to_pixel(p) for p in self.tripwire])

    @property
    def inside_side(self) -> int:
        """Which side of the line the apron is on. +1 or -1."""
        return side_of_line(centroid([self.to_pixel(p) for p in self.inside]),
                            [self.to_pixel(p) for p in self.tripwire])
