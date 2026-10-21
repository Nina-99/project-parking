"""When a failed read means "reconnect", decided without touching a camera."""

from __future__ import annotations

import sys
from pathlib import Path

RTSP_DIR = Path(__file__).resolve().parents[2] / "rtsp-capture"
if str(RTSP_DIR) not in sys.path:
    sys.path.insert(0, str(RTSP_DIR))

from capture import ReconnectPolicy  # noqa: E402


def test_a_single_bad_read_is_tolerated() -> None:
    policy = ReconnectPolicy(tolerance=3)

    assert policy.failed() is False
    assert policy.failed() is False


def test_enough_consecutive_failures_ask_for_a_reconnect() -> None:
    policy = ReconnectPolicy(tolerance=3)

    assert [policy.failed() for _ in range(4)] == [False, False, False, True]


def test_one_good_frame_forgets_the_whole_streak() -> None:
    """A hiccup every few frames must never accumulate into a reconnect."""
    policy = ReconnectPolicy(tolerance=3)
    for _ in range(10):
        policy.failed()
        policy.failed()
        policy.recovered()

    assert policy.failures == 0
    assert policy.attempts == 0


def test_the_delay_grows_and_then_caps() -> None:
    policy = ReconnectPolicy(schedule=(2, 4, 8))

    assert [policy.delay() for _ in range(5)] == [2, 4, 8, 8, 8]


def test_the_warning_is_printed_once_per_streak() -> None:
    policy = ReconnectPolicy()

    assert policy.should_warn() is True
    assert policy.should_warn() is False
    policy.recovered()
    assert policy.should_warn() is True
