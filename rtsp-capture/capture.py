"""
RTSP stream capturer — runs as a separate process per camera.

Captures frames from an RTSP stream, applies frame skipping,
and dispatches Celery tasks for detection.

Usage:
    python rtsp-capture/capture.py --camera-id gate-1 --rtsp-url rtsp://...
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


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
        self._previous: np.ndarray | None = None
        self._running = False
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        """Start the capture loop (blocking)."""
        self._running = True
        frame_count = 0
        last_process_time = 0.0

        print(f"[{self.camera_id}] Connecting to {self.rtsp_url} ...")
        self._cap = cv2.VideoCapture(self.rtsp_url)

        if not self._cap.isOpened():
            print(f"[{self.camera_id}] ERROR: Could not open stream")
            return

        # Set buffer size to minimum for lower latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[{self.camera_id}] Connected. Sampling every {self.sample_interval}s")

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                print(f"[{self.camera_id}] Lost frame — reconnecting in 2s...")
                self._reconnect()
                continue

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
        from backend.worker.tasks.detection import process_frame

        try:
            process_frame.delay(self.camera_id, encoded.tobytes(), self.gate)
        except Exception as error:  # broker down, queue full, task not imported
            # A broker blip must not stop the capture loop: the road keeps moving
            # and the next frame can still make it.
            print(f"[{self.camera_id}] could not queue the frame: {error}")
            return
        print(f"[{self.camera_id}] frame queued for detection")

    def _reconnect(self) -> None:
        """Attempt to reconnect to the RTSP stream."""
        self._cleanup()
        # The next frame comes from a different moment; comparing against the old
        # one would report motion that never happened.
        self._previous = None
        time.sleep(2)
        self._cap = cv2.VideoCapture(self.rtsp_url)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[{self.camera_id}] Reconnected successfully")
        else:
            print(f"[{self.camera_id}] Reconnection failed — retrying...")

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
        sys.exit(0)

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
