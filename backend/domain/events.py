# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Turning frames into gate events.

One frame is not one vehicle. The gate samples every half second, so a car
crossing it appears in several consecutive frames; persisting each frame creates
several vehicles and several parking sessions for one car. That inflates the
occupancy count -- the one number the operator trusts -- and every statistic
derived from it.

This module groups detections into tracks by spatial overlap and emits **one**
event per track, after the track has been seen often enough to be real. It is
pure: no frames, no models, no database.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from backend.domain.vehicle import VehicleAttributes
from backend.domain.zones import DEFAULT_CROSSING_HITS, ZoneGate

Box = tuple[int, int, int, int]

IOU_MATCH_THRESHOLD = 0.3
CONFIRM_AFTER_FRAMES = 3
MAX_AGE_SECONDS = 10.0

def intersection_over_union(a: Box, b: Box) -> float:
    """Overlap of two boxes, 0.0 when they do not touch."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


@dataclass(frozen=True)
class GateDetection:
    """One vehicle found in one frame."""

    box: Box
    attributes: VehicleAttributes
    detection_confidence: float = 0.0


@dataclass(frozen=True)
class GateEvent:
    """One vehicle at a gate, assembled from the frames it appeared in.

    ``box`` and ``attributes`` come from the largest view of the vehicle, not
    from the last frame: the crop that reaches the operator should be the one
    the classifiers had the most pixels to work with.
    """

    camera_id: str
    track_id: int
    box: Box
    observed_at: datetime
    attributes: VehicleAttributes
    detection_confidence: float = 0.0
    # "entry" or "exit" when the crossing was seen, None when the camera has no
    # geometry: without a line there is no direction, only a vehicle in frame.
    direction: str | None = None


def _area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class _Track:
    track_id: int
    box: Box
    attributes: VehicleAttributes
    first_seen: datetime
    last_seen: datetime
    hits: int = 1
    emitted: bool = False
    inside_at_start: bool | None = None
    inside_now: bool = False
    flip_hits: int = 0
    vx: float = 0.0
    vy: float = 0.0
    best_box: Box | None = None
    best_attributes: VehicleAttributes | None = None
    best_area: int = 0
    best_detection_confidence: float = 0.0


class TrackAggregator:
    """Groups frame detections into tracks and emits one event per track.

    Matching is greedy IoU against the tracks still alive. That is enough at a
    gate, where a vehicle moves a few pixels between samples and two vehicles
    rarely overlap. It is not a general multi-object tracker and does not pretend
    to be one: a track that disappears for longer than ``max_age_seconds`` is
    forgotten, and a later appearance starts a new event.

    The aggregator is stateful per camera. One instance per camera, in one
    process: the state is the reason the worker must not fan out frames of the
    same camera across processes.

    With a ``gate``, a track is only emitted when it entered or left the apron
    polygon, and the side it starts on decides the direction. Using the polygon
    and not the tripwire is deliberate: a line drawn across the frame can sit
    behind the traffic lane, and then street traffic already counts as inside.
    """

    def __init__(
        self,
        *,
        match_iou: float = IOU_MATCH_THRESHOLD,
        confirm_after: int = CONFIRM_AFTER_FRAMES,
        max_age_seconds: float = MAX_AGE_SECONDS,
        gate: ZoneGate | None = None,
        crossing_hits: int = DEFAULT_CROSSING_HITS,
    ) -> None:
        if confirm_after < 1:
            raise ValueError("confirm_after must be at least 1")
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        if crossing_hits < 1:
            raise ValueError("crossing_hits must be at least 1")
        self.match_iou = match_iou
        self.confirm_after = confirm_after
        self.max_age_seconds = max_age_seconds
        self.gate = gate
        self.crossing_hits = crossing_hits
        self._tracks: dict[str, list[_Track]] = {}
        self._next_id = 1

    def update(
        self,
        camera_id: str,
        detections: list[GateDetection],
        now: datetime,
    ) -> list[GateEvent]:
        """Feed one frame's detections; return the events confirmed by it."""
        alive = [
            track
            for track in self._tracks.get(camera_id, [])
            if (now - track.last_seen).total_seconds() <= self.max_age_seconds
        ]

        matched: set[int] = set()
        for detection in detections:
            track = self._best_match(alive, detection.box, matched)
            if track is None:
                track = _Track(
                    track_id=self._next_id,
                    box=detection.box,
                    attributes=detection.attributes,
                    first_seen=now,
                    last_seen=now,
                    best_box=detection.box,
                    best_attributes=detection.attributes,
                    best_area=_area(detection.box),
                    best_detection_confidence=detection.detection_confidence,
                )
                self._next_id += 1
                alive.append(track)
            else:
                previous = track.box
                track.box = detection.box
                track.attributes = detection.attributes
                track.last_seen = now
                track.hits += 1
                # Constant-velocity estimate, smoothed, so the next match is
                # tested against where this vehicle is going and not only where
                # it was. Without it, a second car arriving on the same spot is
                # absorbed by the first one's track.
                dx = (detection.box[0] + detection.box[2]
                      - previous[0] - previous[2]) / 2.0
                dy = (detection.box[1] + detection.box[3]
                      - previous[1] - previous[3]) / 2.0
                track.vx = 0.5 * track.vx + 0.5 * dx
                track.vy = 0.5 * track.vy + 0.5 * dy
                if _area(detection.box) > track.best_area:
                    track.best_box = detection.box
                    track.best_attributes = detection.attributes
                    track.best_area = _area(detection.box)
                    track.best_detection_confidence = detection.detection_confidence
            if self.gate is not None:
                inside = self.gate.in_inside(detection.box)
                if track.inside_at_start is None:
                    track.inside_at_start = inside
                track.inside_now = inside
                if inside != track.inside_at_start:
                    track.flip_hits += 1
            matched.add(track.track_id)

        self._tracks[camera_id] = alive
        return self._confirmed(alive, camera_id, now)
    def _best_match(
        self,
        tracks: list[_Track],
        box: Box,
        already_matched: set[int],
    ) -> _Track | None:
        best: _Track | None = None
        best_score = self.match_iou
        for track in tracks:
            if track.track_id in already_matched:
                continue
            # Score against the current box and against the predicted one: a
            # vehicle that keeps moving is found where it is going to be, and a
            # newcomer standing on top of it no longer looks like a match.
            score = max(intersection_over_union(track.box, box),
                        intersection_over_union(self._predict(track), box))
            if score >= best_score:
                best, best_score = track, score
        return best

    @staticmethod
    def _predict(track: _Track) -> Box:
        dx, dy = int(round(track.vx)), int(round(track.vy))
        x1, y1, x2, y2 = track.box
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def _confirmed(
        self,
        tracks: list[_Track],
        camera_id: str,
        now: datetime,
    ) -> list[GateEvent]:
        events: list[GateEvent] = []
        for track in tracks:
            if track.emitted:
                continue
            if self.gate is None:
                if track.hits < self.confirm_after:
                    continue
                direction = None
            else:
                # The crossing of the line is the event, and it happens at the
                # gate: waiting for the vehicle to leave the whole polygon puts
                # the timestamp twenty seconds late. The polygon still has a say
                # -- a vehicle that never reached the apron only drove past.
                if track.flip_hits < self.crossing_hits:
                    continue
                direction = "entry" if track.inside_now else "exit"
            track.emitted = True
            if os.environ.get("PARKING_DEBUG_TRACKS"):
                print(f"[track] {camera_id} id={track.track_id} "
                      f"inside={track.inside_at_start}->{track.inside_now} "
                      f"flips={track.flip_hits} "
                      f"hits={track.hits} dir={direction}", flush=True)
            events.append(
                GateEvent(
                    camera_id=camera_id,
                    track_id=track.track_id,
                    box=track.best_box or track.box,
                    observed_at=now,
                    attributes=track.best_attributes or track.attributes,
                    detection_confidence=track.best_detection_confidence,
                    direction=direction,
                )
            )
        return events
