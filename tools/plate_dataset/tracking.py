# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Per-track trajectory state and entry/pass classification.

The extractor reasons over a track's whole trajectory, never a single frame.
An entering vehicle crosses the tripwire toward the premises and moves mostly
in depth (vertical displacement dominant, scale growing); a passing vehicle
keeps a horizontal heading and never reaches the inside zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from plate_dataset.geometry import ZoneModel, bottom_center


@dataclass
class FrameCandidate:
    """A vehicle crop that might contain a legible plate.

    ``origin`` is (x1, y1, frame_width, frame_height): the crop's offset inside
    the full frame plus the frame size, needed to map frame-fixed regions (the
    burnt-in OSD) into crop-local coordinates.
    """

    frame_idx: int
    crop: np.ndarray
    area: float
    score: float
    origin: tuple[int, int, int, int] = (0, 0, 0, 0)
    vehicle_class: int | None = None


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    last_frame: int
    last_seen: int
    hits: int = 1
    frames_since_seen: int = 0
    points: list[tuple[int, float, float, float]] = field(default_factory=list)
    sides: list[int] = field(default_factory=list)
    visited_street: bool = False
    visited_inside: bool = False
    crossing: str | None = None  # "entry" (street -> inside) or "exit" (inside -> street)
    zone_state: int = 0  # 0 unknown, 1 last seen in street, 2 last seen inside
    best_frames: list[FrameCandidate] = field(default_factory=list)

    def update(self, frame_idx: int, box: np.ndarray, zones: ZoneModel, top_k: int,
               vehicle_crop: np.ndarray, score: float,
               origin: tuple[int, int, int, int] = (0, 0, 0, 0),
               vehicle_class: int | None = None) -> None:
        cx, cy = bottom_center(box)
        area = float((box[2] - box[0]) * (box[3] - box[1]))

        self.last_frame = frame_idx
        self.last_seen = frame_idx
        self.frames_since_seen = 0
        self.hits += 1
        self.points.append((frame_idx, cx, cy, area))
        self.sides.append(zones.side((cx, cy)))

        in_street = zones.in_street((cx, cy))
        in_inside = zones.in_inside((cx, cy))
        if in_street:
            self.visited_street = True
        if in_inside:
            self.visited_inside = True
        self._update_crossing(in_street, in_inside)

        self._consider_candidate(frame_idx, vehicle_crop, area, score, top_k, origin,
                                 vehicle_class)

    def _update_crossing(self, in_street: bool, in_inside: bool) -> None:
        """Record the first zone-to-zone transition and its direction.

        Direction is derived from the *polygons* (street vs inside), not from
        the tripwire: the tripwire only separates them if the user happens to
        draw it between the two zones, which is easy to get wrong.
        """
        if self.crossing is None:
            if in_street and self.zone_state == 2:
                self.crossing = "exit"
            elif in_inside and self.zone_state == 1:
                self.crossing = "entry"
        if in_street:
            self.zone_state = 1
        elif in_inside:
            self.zone_state = 2

    def _consider_candidate(self, frame_idx: int, vehicle_crop: np.ndarray,
                            area: float, score: float, top_k: int,
                            origin: tuple[int, int, int, int],
                            vehicle_class: int | None = None) -> None:
        if vehicle_crop.size == 0:
            return
        self.best_frames.append(
            FrameCandidate(frame_idx, vehicle_crop, area, score, origin, vehicle_class)
        )
        self.best_frames.sort(key=lambda c: c.score, reverse=True)
        del self.best_frames[top_k:]

    def largest_candidate(self) -> FrameCandidate | None:
        """The crop showing the most of the vehicle, or ``None`` when there is none.

        ``best_frames`` is ranked for *plate* legibility, which favours a sharp,
        slightly distant view. Body type and paint colour read better from the
        largest complete view of the car, so attribute samples are taken from
        the widest crop rather than the highest-scoring one.
        """
        if not self.best_frames:
            return None
        return max(self.best_frames, key=lambda c: c.area)

    def classify(self, zones: ZoneModel, min_hits: int) -> str:
        """Return ``entry``, ``exit``, ``pass`` or ``unknown`` for this track.

        Direction comes from the tripwire crossing order, so a vehicle leaving
        the premises is not mistaken for one arriving (both touch inside_zone).
        A track that reaches the inside zone without ever crossing the tripwire
        is a manoeuvre inside the premises, not a gate event.
        """
        if self.hits < min_hits:
            return "unknown"
        if self.crossing is not None:
            return self.crossing
        if not self.visited_inside:
            return "pass"
        return "unknown"
