"""Replay recorded footage as if it were the live gate cameras.

    # see which hours both cameras recorded
    uv run python tools/simulate_cameras.py --list-hours

    # what would run, without starting anything
    uv run python tools/simulate_cameras.py --hour 2026033010 --dry-run

    # serve both cameras and print the commands for the rest of the chain
    uv run python tools/simulate_cameras.py --hour 2026033010

Why RTSP and not a file path: ``rtsp-capture/capture.py`` opens an RTSP URL with
OpenCV, so serving the footage over RTSP exercises the production path without
changing a line of it. A file source would need a simulation mode inside the
application, and then you would be testing the simulator.

Why ``-re``: without it ffmpeg pushes the file as fast as it can decode, and the
capture loop's "one frame every 0.5 s" stops meaning anything. With it, the stream
arrives at wall-clock speed, like a camera.

Both cameras are fed the *same* recorded hour, because entry and exit only relate
to each other if they are replayed together. Replaying different hours produces
entries with no matching exits and the session logic has nothing to correlate.

A note on clocks: the app timestamps events with the wall clock, not with the
timestamp in the filename. A window search during a replay is therefore relative
to "now" unless the container's clock is offset (``libfaketime``, or setting the
container date). The harness prints the recorded hour so the operator knows which
video time "now" corresponds to.

mediamtx is not vendored -- it is a 56 MB binary. Point ``--mediamtx`` at it, or
fetch it once::

    curl -sL -o m.tar.gz https://github.com/bluenviron/mediamtx/releases/download/v1.21.2/mediamtx_v1.21.2_linux_amd64.tar.gz
    tar xzf m.tar.gz mediamtx && install -m755 mediamtx ~/.local/bin/mediamtx
"""

from __future__ import annotations

import argparse
import collections
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CHUNK_RE = re.compile(r"_(?P<start>\d{14})_(?P<end>\d{14})\.mp4$")

DEFAULT_CAMERAS: dict[str, str] = {
    "gate-1": "C.OCCIDENTE_ch16",
    "gate-2": "XVR_ch6",
}

MEDIAMTX_CANDIDATES = (
    "mediamtx",
    "~/.local/bin/mediamtx",
    "/usr/local/bin/mediamtx",
    "/tmp/opencode/sim/mediamtx",
)


def parse_chunk(path: Path) -> tuple[str, str] | None:
    """(start, end) timestamps of a recorded chunk, or None if unnamed."""
    match = CHUNK_RE.search(path.name)
    if match is None:
        return None
    return match.group("start"), match.group("end")


def index_videos(root: Path, cameras: dict[str, str]) -> dict[str, dict[str, list[Path]]]:
    """Per camera, the recorded chunks grouped by their start hour.

    A list per hour, not a single file: the recorder can split an hour when it
    restarts, and both halves should be replayed in order.
    """
    index: dict[str, dict[str, list[Path]]] = {name: collections.defaultdict(list)
                                              for name in cameras}
    for path in sorted(root.rglob("*.mp4")):
        stamps = parse_chunk(path)
        if stamps is None:
            continue
        for name, pattern in cameras.items():
            if pattern in path.name:
                index[name][stamps[0][:10]].append(path)
    return {name: dict(hours) for name, hours in index.items()}


def common_hours(index: dict[str, dict[str, list[Path]]]) -> list[str]:
    """Hours recorded by every camera, oldest first."""
    if not index:
        return []
    shared: set[str] | None = None
    for hours in index.values():
        shared = set(hours) if shared is None else shared & set(hours)
    return sorted(shared or set())


def stream_urls(port: int, cameras: list[str]) -> dict[str, str]:
    return {name: f"rtsp://127.0.0.1:{port}/{name}" for name in cameras}


def cameras_env(urls: dict[str, str]) -> str:
    """The ``RTSP_CAMERAS`` value the app parses: ``name=url,name=url``."""
    return ",".join(f"{name}={url}" for name, url in urls.items())


MEDIAMTX_CONFIG = """# Minimal server for replaying recorded footage.
# paths/all_others is what makes an arbitrary stream name publishable: with no
# config file mediamtx defines no paths at all and rejects every publisher with
# "path is not configured".
logLevel: info
rtsp: yes
paths:
  all_others:
"""


def write_mediamtx_config(work_dir: Path) -> Path:
    """A minimal server config, in a directory of its own.

    mediamtx writes its self-signed certificates next to its working directory,
    so it must not run inside the repository.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    config = work_dir / "mediamtx.yml"
    config.write_text(MEDIAMTX_CONFIG, encoding="utf-8")
    return config


def ffmpeg_command(path: Path, url: str) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-re",                      # pace at wall-clock speed
        "-stream_loop", "-1",       # the hour repeats forever
        "-i", str(path),
        "-c", "copy",               # no re-encoding: the stream keeps NVR quality
        "-f", "rtsp", url,
    ]


def find_mediamtx(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    for candidate in MEDIAMTX_CANDIDATES:
        expanded = Path(candidate).expanduser()
        found = shutil.which(candidate) or (str(expanded) if expanded.exists() else None)
        if found:
            return Path(found)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay footage as live cameras.")
    parser.add_argument("--videos", type=Path, default=Path("/mnt/nina/PARKING"))
    parser.add_argument("--camera", action="append", default=[],
                        metavar="NAME=PATTERN",
                        help="Repeatable. Defaults to gate-1=C.OCCIDENTE_ch16 and "
                             "gate-2=XVR_ch6.")
    parser.add_argument("--hour", help="Recorded hour to replay, as YYYYMMDDHH.")
    parser.add_argument("--rtsp-port", type=int, default=8554)
    parser.add_argument("--mediamtx", type=Path)
    parser.add_argument("--work-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "smart-parking-sim",
                        help="Where mediamtx keeps its config and certificates.")
    parser.add_argument("--list-hours", action="store_true",
                        help="Print the hours both cameras recorded, then exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without starting processes.")
    parser.add_argument("--with-capture", action="store_true",
                        help="Also run rtsp-capture/capture.py per camera.")
    parser.add_argument("--sample-interval", type=float, default=0.5,
                        help="Frame sampling passed to capture.py.")
    return parser


def parse_cameras(values: list[str]) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_CAMERAS)
    cameras: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"[sim] --camera expects NAME=PATTERN, got {value!r}")
        name, pattern = value.split("=", 1)
        cameras[name.strip()] = pattern.strip()
    return cameras


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cameras = parse_cameras(args.camera)

    index = index_videos(args.videos, cameras)
    hours = common_hours(index)

    if args.list_hours:
        for name, recorded in index.items():
            print(f"[sim] {name} ({cameras[name]}): {len(recorded)} hours")
        print(f"[sim] hours recorded by every camera: {len(hours)}")
        for hour in hours[:20]:
            print(f"  {hour}")
        if len(hours) > 20:
            print(f"  ... and {len(hours) - 20} more")
        return

    if not hours:
        raise SystemExit(f"[sim] no hour is recorded by all cameras under {args.videos}")

    hour = args.hour or hours[0]
    if hour not in hours:
        raise SystemExit(
            f"[sim] {hour} is not recorded by every camera. "
            f"Available (first 5): {', '.join(hours[:5])}")

    urls = stream_urls(args.rtsp_port, list(cameras))
    plan = {name: [ffmpeg_command(path, urls[name]) for path in index[name][hour]]
            for name in cameras}

    print(f"[sim] replaying hour {hour} from {args.videos}")
    for name in cameras:
        chunks = index[name][hour]
        print(f"[sim] {name}: {len(chunks)} chunk(s), {len(plan[name])} publisher(s)")
    print("\n[sim] put this in .env:\n")
    print(f"RTSP_CAMERAS={cameras_env(urls)}\n")

    if args.dry_run:
        for commands in plan.values():
            for command in commands:
                print("[sim] would run:", " ".join(command))
        return

    binary = find_mediamtx(args.mediamtx)
    if binary is None:
        raise SystemExit(
            "[sim] mediamtx not found. Pass --mediamtx PATH or install it; see the "
            "module docstring for the one-line download.")

    processes: list[subprocess.Popen] = []
    try:
        config = write_mediamtx_config(args.work_dir)
        processes.append(subprocess.Popen([str(binary), str(config)], cwd=args.work_dir))
        print(f"[sim] mediamtx: {binary}  (config {config})")
        time.sleep(2)  # let the server bind before publishers connect

        publishers: dict[str, list[subprocess.Popen]] = {}
        for name, commands in plan.items():
            publishers[name] = [subprocess.Popen(command) for command in commands]
        time.sleep(3)  # a publisher that cannot connect dies within a second

        dead = {name: procs for name, procs in publishers.items()
                if any(proc.poll() is not None for proc in procs)}
        if dead:
            raise SystemExit(
                f"[sim] publisher failed for {', '.join(dead)} -- ffmpeg reported the "
                "reason above. Nothing is streaming, so the harness is not left running.")
        processes.extend(proc for procs in publishers.values() for proc in procs)
        for name in cameras:
            print(f"[sim] publishing {name} at {urls[name]}")

        capture = Path(__file__).resolve().parent.parent / "rtsp-capture" / "capture.py"
        for name, url in urls.items():
            print(f"\n[sim] capture command for {name}:\n"
                  f"  uv run python {capture} --camera-id {name} --rtsp-url {url}")
        if args.with_capture:
            for name, url in urls.items():
                processes.append(subprocess.Popen([
                    sys.executable, str(capture), "--camera-id", name,
                    "--rtsp-url", url, "--sample-interval", str(args.sample_interval),
                ]))

        print("\n[sim] running. Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[sim] stopping")
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("[sim] stopped")


def _exit_cleanly(_signum: int, _frame: object) -> None:
    """Turn SIGTERM into the same path as Ctrl+C, so the finally block runs.

    Without this, killing the harness with `timeout` or `kill` leaves mediamtx and
    two ffmpeg publishers running in the background, holding ports 8554 and 1935.
    """
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _exit_cleanly)
    main()
