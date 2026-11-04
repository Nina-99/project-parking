# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Mark every vehicle crossing by hand, on the source video (H1.5).

    uv run python tools/label_events_gui.py \
        --video /mnt/nina/PARKING/C.OCCIDENTE_ch16_main_20260330100012_20260330110012.mp4 \
        --output ~/Smart_Parking_dataset/truth/ch16_2026033010.csv

Reads the recording directly, not the RTSP replay. The ground truth then lives
in seconds of video and stays valid for any later replay; taken from the live
stream it would be tied to the wall clock of one afternoon.

Only crossings toward the premises count. A vehicle that passes along the street
is not an event, and marking it would make the system look blind. When in doubt,
let the video run one more second and see where the vehicle went.

Controls
    i / 1        a vehicle entered
    o / 2        a vehicle left
    space        pause / resume
    - / +        slower / faster (0.5x, 1x, 1.5x, 2x, 3x, 4x)
    z            undo the last mark
    , / .        back / forward one frame
    b / f        back / forward five seconds
    q / Esc      quit and save

When the recording ends the window stays on the last frame, marked END in the
header, so a vehicle leaving in the final seconds can still be marked.

Marks are written as they happen, so a crash costs the last key and not the
afternoon. Re-running against an existing CSV resumes it; ``--force`` starts
over.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from event_truth import Crossing, read_truth  # noqa: E402

MAX_VIEW_W = 1100
MAX_VIEW_H = 720
HEADER = ["video_seconds", "direction"]
SPEEDS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="CSV of crossings in video seconds. Resumed if it exists.")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start at this second of the video (default: 0).")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Starting playback speed (default: 1.0). Change it while "
                             "running with - and +.")
    parser.add_argument("--force", action="store_true",
                        help="Start from an empty CSV instead of resuming.")
    return parser


class Session:
    """The marks plus the file they are appended to."""

    def __init__(self, path: Path, force: bool) -> None:
        self.path = path
        self.marks: list[Crossing] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force and path.stat().st_size:
            self.marks = read_truth(path)
        self._handle = path.open("a", encoding="utf-8", newline="")
        if self._handle.tell() == 0:
            csv.writer(self._handle).writerow(HEADER)
            self._handle.flush()

    def add(self, crossing: Crossing) -> None:
        self.marks.append(crossing)
        csv.writer(self._handle).writerow(
            [f"{crossing.video_seconds:.3f}", crossing.direction])
        self._handle.flush()

    def undo(self) -> Crossing | None:
        if not self.marks:
            return None
        removed = self.marks.pop()
        self._rewrite()
        return removed

    def _rewrite(self) -> None:
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for mark in sorted(self.marks, key=lambda item: item.video_seconds):
                writer.writerow([f"{mark.video_seconds:.3f}", mark.direction])

    def close(self) -> None:
        self._handle.close()


def _next_speed(current: float, *, faster: bool) -> float:
    """Step through the speed ladder, staying inside it."""
    index = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - current))
    index += 1 if faster else -1
    return SPEEDS[max(0, min(len(SPEEDS) - 1, index))]


def draw(frame, seconds: float, marks: list[Crossing], paused: bool, speed: float,
         ended: bool = False):
    view_h, view_w = frame.shape[:2]
    scale = min(MAX_VIEW_W / view_w, MAX_VIEW_H / view_h, 1.0)
    canvas = frame if scale >= 1.0 else cv2.resize(
        frame, (int(view_w * scale), int(view_h * scale)), interpolation=cv2.INTER_AREA)
    entries = sum(1 for mark in marks if mark.direction == "entry")
    exits = len(marks) - entries
    state = "END - [q] to quit" if ended else ("PAUSED" if paused else f"{speed}x")
    header = f"{seconds:8.2f}s  entries={entries}  exits={exits}  {state}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26),
                  (30, 30, 150) if ended else (30, 30, 30), -1)
    cv2.putText(canvas, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    for offset, mark in enumerate(marks[-5:]):
        cv2.putText(canvas, f"{mark.video_seconds:.1f}s {mark.direction}",
                    (8, canvas.shape[0] - 8 - 16 * offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 80), 1)
    return canvas


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    begin = args.start
    if args.start:
        capture.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
        # Trust the container, not the request: seeking to a position this codec
        # cannot reach silently leaves the reader at the start, and then every
        # timestamp would be offset by the value we asked for.
        begin = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 or args.start

    session = Session(args.output, args.force)
    window = "label access events"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    print(f"[truth] {args.video.name} at {args.speed}x -> {args.output}", flush=True)

    paused = False
    frame = None
    seconds = begin
    frame_index = 0
    speed = args.speed
    play_started = time.perf_counter()
    play_seconds = begin

    def refresh(new_index: int) -> None:
        """Seek to a frame and show it, paused or not."""
        nonlocal frame, seconds, frame_index
        new_index = max(0, new_index)
        capture.set(cv2.CAP_PROP_POS_FRAMES, new_index)
        ok, advanced = capture.read()
        if ok:
            frame = advanced
            frame_index = new_index + 1
            seconds = begin + new_index / fps

    ended = False

    try:
        while True:
            if not paused and not ended:
                ok, advanced = capture.read()
                if not ok:
                    # Hold on the last frame instead of quitting. The vehicle
                    # that leaves in the final seconds is exactly the one the
                    # reviewer is about to mark, and closing the window loses it.
                    ended = True
                    paused = True
                else:
                    frame = advanced
                    seconds = begin + frame_index / fps
                    frame_index += 1
            if frame is None:
                break

            cv2.imshow(window, draw(frame, seconds, session.marks, paused, speed, ended))
            key = cv2.waitKey(1 if not paused else 30) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" ") and not ended:
                paused = not paused
                play_started = time.perf_counter()
                play_seconds = seconds
                continue
            if key in (ord("i"), ord("1")):
                session.add(Crossing(seconds, "entry"))
            elif key in (ord("o"), ord("2")):
                session.add(Crossing(seconds, "exit"))
            elif key == ord("z"):
                session.undo()
            elif key in (ord("+"), ord("="), ord("-"), ord("_")):
                speed = _next_speed(speed, faster=key in (ord("+"), ord("=")))
                play_started = time.perf_counter()
                play_seconds = seconds
            elif key == ord(","):
                refresh(frame_index - 2)
            elif key == ord("."):
                refresh(frame_index)
            elif key == ord("b"):
                refresh(int((max(begin, seconds - 5.0) - begin) * fps))
            elif key == ord("f"):
                refresh(int((seconds + 5.0 - begin) * fps))

            if not paused:
                target = play_started + (seconds - play_seconds) / max(speed, 0.1)
                remaining = target - time.perf_counter()
                if remaining > 0:
                    time.sleep(min(remaining, 0.05))
    finally:
        capture.release()
        session.close()
        cv2.destroyWindow(window)

    entries = sum(1 for mark in session.marks if mark.direction == "entry")
    print(f"[truth] {len(session.marks)} crossings "
          f"({entries} entry, {len(session.marks) - entries} exit)")
    print(f"[truth] wrote {args.output}")


if __name__ == "__main__":
    main()
