"""
RTSP stream capturer — runs as a separate process per camera.

Captures frames from an RTSP stream, applies frame skipping,
and dispatches Celery tasks for detection.

Usage:
    python rtsp-capture/capture.py --camera-id gate-1 --rtsp-url rtsp://...
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force RTSP over TCP. OpenCV's FFmpeg backend negotiates UDP by default, and
# mediamtx logs "RTP packets are too big (1460 > 1440), remuxing them into
# smaller ones" on this footage: fragmented HEVC over UDP loses pieces silently,
# which shows up as reads that fail with no error and frames that never reach the
# inference. TCP costs a little latency and removes the failure mode. The receive
# timeout keeps a dead peer from blocking a read forever.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;5000000")


@dataclass
class ReconnectPolicy:
    """When a bad read means "reconnect" and how long to wait before trying.

    A live RTSP stream returns a failed read for reasons that are not a
    disconnection: an empty buffer between keyframes, a lost packet, the decoder
    waiting for its first reference frame. Reconnecting on the first failure
    throws away a healthy stream and costs seconds of footage, so failures have
    to be consecutive to count.

    The delay grows with each attempt and caps, so a camera that is off overnight
    is retried every half minute instead of in a tight loop.
    """

    tolerance: int = 30
    schedule: tuple[int, ...] = (2, 4, 8, 16, 30)
    failures: int = 0
    attempts: int = 0
    warned: bool = field(default=False)

    def failed(self) -> bool:
        """Record a failed read; True when the stream should be reopened."""
        self.failures += 1
        return self.failures > self.tolerance

    def recovered(self) -> None:
        """A good frame: forget everything, the stream is healthy again."""
        self.failures = 0
        self.attempts = 0
        self.warned = False

    def delay(self) -> int:
        """Seconds to wait before the next attempt."""
        index = min(self.attempts, len(self.schedule) - 1)
        self.attempts += 1
        return self.schedule[index]

    def should_warn(self) -> bool:
        """True once per streak, so a hiccup does not flood the log."""
        if self.warned:
            return False
        self.warned = True
        return True


class RTSPCapturer:
    """
    Manages an RTSP camera stream with reconnection and frame sampling.

    Attributes:
        camera_id: Human-readable camera name (e.g., 'gate-1', 'gate-2').
        rtsp_url: Full RTSP URL (rtsp://user:pass@ip:554/stream).
        sample_interval: Process one frame every N seconds.
        gate: ``entry`` or ``exit``: only an entry opens a parking session.
        motion_threshold: Mean absolute difference, over a 160x90 grey copy of
            the frame, above which the frame is worth an inference. A parked
            vehicle produces none, so the worker is not asked about it.
    """

    MOTION_SIZE = (160, 90)

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        sample_interval: float = 0.5,
        gate: str = "entry",
        motion_threshold: float = 8.0,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.sample_interval = sample_interval
        self.gate = gate
        self.motion_threshold = motion_threshold
        self.reconnect = ReconnectPolicy()
        self._dispatch_thread: threading.Thread | None = None
        self.dropped_frames = 0
        self._previous: np.ndarray | None = None
        self._running = False
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        """Start the capture loop (blocking)."""
        self._running = True
        frame_count = 0
        last_process_time = 0.0

        print(f"[{self.camera_id}] Connecting to {self.rtsp_url} ...")
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        # A camera that is not up yet is normal at boot, so the first connection
        # retries with the same backoff instead of giving up.
        while self._running and not self._cap.isOpened():
            self.reconnect.failed()
            self._reconnect()
        if not self._running:
            return

        # Set buffer size to minimum for lower latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[{self.camera_id}] Connected. Sampling every {self.sample_interval}s")

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if not self.reconnect.failed():
                    # Tolerated: a dropped packet or a buffer gap, not a
                    # disconnection. The next read usually recovers.
                    if self.reconnect.should_warn():
                        print(f"[{self.camera_id}] read failed, still connected — "
                              "tolerating up to "
                              f"{self.reconnect.tolerance} in a row")
                    continue
                print(f"[{self.camera_id}] {self.reconnect.failures} reads failed "
                      "— reopening the stream...")
                self._reconnect()
                continue

            self.reconnect.recovered()
            frame_count += 1
            now = time.time()

            # Frame skipping: only process every sample_interval seconds
            if now - last_process_time >= self.sample_interval:
                self._process_frame(frame)
                last_process_time = now

        self._cleanup()

    def _has_motion(self, frame: cv2.typing.MatLike) -> bool:
        """True when the frame differs enough from the previous sample.

        Inference is the expensive part of the chain, so a frame where nothing
        moves never reaches it. The comparison is against the *previous sampled*
        frame, not the previous decoded one, which keeps the threshold meaningful
        at the sampling rate.
        """
        small = cv2.cvtColor(
            cv2.resize(frame, self.MOTION_SIZE), cv2.COLOR_BGR2GRAY)
        if self._previous is None:
            self._previous = small
            return False
        delta = float(np.abs(
            small.astype(np.int16) - self._previous.astype(np.int16)).mean())
        self._previous = small
        return delta >= self.motion_threshold

    def _process_frame(self, frame: cv2.typing.MatLike) -> None:
        """Queue a frame for detection when something moved in it."""
        if not self._has_motion(frame):
            return
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            print(f"[{self.camera_id}] could not encode the frame")
            return
        self._dispatch(encoded.tobytes())

    def _dispatch(self, frame_bytes: bytes) -> None:
        """Publish the frame without ever blocking the sampling loop.

        Celery's producer retries until the broker answers: with Redis down it was
        measured blocking for 107 seconds, which stops the camera from sampling
        for close to two minutes. Setting a short socket timeout did not fix it, so
        the publish runs on its own thread and the loop keeps reading.

        Only one publish is in flight at a time. While the broker is unhealthy the
        thread stays stuck; queueing another per frame would spawn hundreds of
        them. Frames are transient, so dropping them is the correct trade.
        """
        if self._dispatch_thread is not None and self._dispatch_thread.is_alive():
            self.dropped_frames += 1
            return

        def publish() -> None:
            # The Celery app must be loaded before the task is called, or delay()
            # resolves against the default app and dials amqp://localhost.
            from backend.worker.celery_app import celery_app  # noqa: F401
            from backend.worker.tasks.detection import process_frame

            try:
                process_frame.delay(self.camera_id, frame_bytes, self.gate)
                print(f"[{self.camera_id}] frame queued for detection", flush=True)
            except Exception as error:  # broker down, queue full, task missing
                print(f"[{self.camera_id}] could not queue the frame: {error}",
                      flush=True)

        self._dispatch_thread = threading.Thread(target=publish, daemon=True)
        self._dispatch_thread.start()

    def _reconnect(self) -> None:
        """Reopen the stream after a delay that grows with the attempt count."""
        self._cleanup()
        # The next frame comes from a different moment; comparing against the old
        # one would report motion that never happened.
        self._previous = None
        delay = self.reconnect.delay()
        print(f"[{self.camera_id}] reconnecting in {delay}s "
              f"(attempt {self.reconnect.attempts})")
        time.sleep(delay)
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.reconnect.recovered()
            print(f"[{self.camera_id}] reconnected")
        else:
            print(f"[{self.camera_id}] could not open the stream; will retry")

    def _cleanup(self) -> None:
        """Release the video capture."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None

    def stop(self) -> None:
        """Signal the capture loop to stop."""
        self._running = False


def main() -> None:
    parser = argparse.ArgumentParser(description="RTSP Camera Capture")
    parser.add_argument("--camera-id", required=True, help="Camera identifier")
    parser.add_argument("--rtsp-url", required=True, help="RTSP stream URL")
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Frame sampling interval in seconds (default: 0.5)",
    )
    parser.add_argument("--gate", choices=("entry", "exit"), default="entry",
                        help="Only an entry opens a parking session.")
    parser.add_argument("--motion-threshold", type=float, default=8.0,
                        help="Mean grey difference above which a frame is inferred.")
    args = parser.parse_args()

    capturer = RTSPCapturer(
        camera_id=args.camera_id,
        rtsp_url=args.rtsp_url,
        sample_interval=args.interval,
        gate=args.gate,
        motion_threshold=args.motion_threshold,
    )

    def _signal_handler(_sig, _frame) -> None:  # type: ignore[no-untyped-def]
        print(f"\n[{args.camera_id}] Shutting down...")
        capturer.stop()
        # A publish stuck on a dead broker leaves a daemon thread inside a socket
        # call. Normal interpreter shutdown then aborts with "FATAL: exception not
        # rethrown" and dumps core, so exit without finalising. Buffers are
        # flushed first; nothing else is left to write.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("\n=== Parking RTSP Capturer ===")
    print(f"Camera:  {args.camera_id}")
    print(f"Stream:  {args.rtsp_url}")
    print(f"Rate:    every {args.interval}s")
    print(f"Gate:    {args.gate}")
    print("============================\n")

    capturer.start()


if __name__ == "__main__":
    main()
