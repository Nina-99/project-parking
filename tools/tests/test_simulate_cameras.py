"""The replay harness: discovery, pairing and command construction."""

from __future__ import annotations

from pathlib import Path

from simulate_cameras import (
    cameras_env,
    common_hours,
    ffmpeg_command,
    index_videos,
    parse_cameras,
    parse_chunk,
    stream_urls,
)

CH16 = "C.OCCIDENTE_ch16_main_20260330100012_20260330110012.mp4"
CH6 = "XVR_ch6_main_20260330100004_20260330110004.mp4"
CH16_LATE = "C.OCCIDENTE_ch16_main_20260330110012_20260330120012.mp4"
CH6_OTHER = "XVR_ch6_main_20260330140004_20260330150004.mp4"
JUNK = "not-a-recording.mp4"

CAMERAS = {"gate-1": "C.OCCIDENTE_ch16", "gate-2": "XVR_ch6"}


def _tree(root: Path, names: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"")
    return root


def test_a_chunk_name_yields_its_timestamps() -> None:
    assert parse_chunk(Path(CH16)) == ("20260330100012", "20260330110012")


def test_a_file_without_timestamps_is_ignored() -> None:
    assert parse_chunk(Path(JUNK)) is None


def test_chunks_are_grouped_by_the_start_hour(tmp_path: Path) -> None:
    index = index_videos(_tree(tmp_path, [CH16, CH6, JUNK]), CAMERAS)

    assert list(index["gate-1"]) == ["2026033010"]
    assert index["gate-1"]["2026033010"] == [tmp_path / CH16]
    assert index["gate-2"]["2026033010"] == [tmp_path / CH6]


def test_an_hour_recorded_by_only_one_camera_is_not_replayable(tmp_path: Path) -> None:
    index = index_videos(_tree(tmp_path, [CH16, CH6, CH6_OTHER]), CAMERAS)

    assert common_hours(index) == ["2026033010"]


def test_hours_are_paired_in_order(tmp_path: Path) -> None:
    index = index_videos(_tree(tmp_path, [CH16, CH16_LATE, CH6]), CAMERAS)

    assert common_hours(index) == ["2026033010"]


def test_a_split_hour_keeps_both_halves_in_order(tmp_path: Path) -> None:
    second = "C.OCCIDENTE_ch16_main_20260330101532_20260330110012.mp4"
    index = index_videos(_tree(tmp_path, [CH16, second, CH6]), CAMERAS)

    assert len(index["gate-1"]["2026033010"]) == 2


def test_the_ffmpeg_command_paces_the_stream_and_copies_it() -> None:
    command = ffmpeg_command(Path(CH16), "rtsp://127.0.0.1:8554/gate-1")

    assert "-re" in command, "without pacing the capture loop's sampling is meaningless"
    assert "-stream_loop" in command and "-1" in command
    assert command[command.index("-c") + 1] == "copy"
    assert command[-1] == "rtsp://127.0.0.1:8554/gate-1"


def test_the_env_value_matches_what_the_app_parses() -> None:
    urls = stream_urls(8554, ["gate-1", "gate-2"])

    assert cameras_env(urls) == (
        "gate-1=rtsp://127.0.0.1:8554/gate-1,gate-2=rtsp://127.0.0.1:8554/gate-2")


def test_camera_patterns_are_parsed_and_defaulted() -> None:
    assert parse_cameras([]) == CAMERAS
    assert parse_cameras(["cam-x=MyNVR_ch3"]) == {"cam-x": "MyNVR_ch3"}
