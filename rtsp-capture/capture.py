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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.app.core.config import settings


class RTSPCapturer:
    """
    Manages an RTSP camera stream with reconnection and frame sampling.

    Attributes:
        camera_id: Human-readable camera name (e.g., 'gate-1', 'gate-2').
        rtsp_url: Full RTSP URL (rtsp://user:pass@ip:554/stream).
        sample_interval: Process one frame every N seconds.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        sample_interval: float = 0.5,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.sample_interval = sample_interval
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

    def _process_frame(self, frame: cv2.typing.MatLike) -> None:
        """
        Process a captured frame.

        TODO:
        1. Motion detection (skip if no vehicle)
        2. Dispatch to Celery for YOLO inference
        3. Handle entry vs exit based on gate position
        """
        # from backend.worker.tasks.detection import process_frame
        # process_frame.delay(self.camera_id, frame.tobytes())
        _ = frame  # Placeholder
        print(f"[{self.camera_id}] Frame captured — detection not yet wired")

    def _reconnect(self) -> None:
        """Attempt to reconnect to the RTSP stream."""
        self._cleanup()
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
    args = parser.parse_args()

    capturer = RTSPCapturer(
        camera_id=args.camera_id,
        rtsp_url=args.rtsp_url,
        sample_interval=args.interval,
    )

    def _signal_handler(sig, frame) -> None:  # type: ignore[no-untyped-def]
        print(f"\n[{args.camera_id}] Shutting down...")
        capturer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print(f"\n=== Parking RTSP Capturer ===")
    print(f"Camera:  {args.camera_id}")
    print(f"Stream:  {args.rtsp_url}")
    print(f"Rate:    every {args.interval}s")
    print(f"============================\n")

    capturer.start()


if __name__ == "__main__":
    main()
