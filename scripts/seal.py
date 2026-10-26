#!/usr/bin/env python3
"""Insert SPDX copyright headers into tracked source files.

    python scripts/seal.py --check           # exit 1 if a file is missing the notice
    python scripts/seal.py --write           # add it, idempotently
    python scripts/seal.py --check --skip-dirty

The script never touches JSON, lockfiles, generated files, binaries or the
legal files themselves. Those are covered by metadata and by LICENSE/NOTICE.
"""
from __future__ import annotations

import argparse
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "COPYRIGHT.toml").read_text(encoding="utf-8"))
OWNER = str(CONFIG["owner"])
YEAR = int(CONFIG["year"])
SPDX = str(CONFIG["spdx"])

EXCLUDED_NAMES = {
    "LICENSE",
    "NOTICE",
    "AUTHORS.md",
    "CITATION.cff",
    "COPYRIGHT.toml",
    ".gitignore",
    ".dockerignore",
}
EXCLUDED_PREFIXES = ("LICENSES/",)
SKIP_SUFFIXES = {
    ".json",
    ".lock",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".ico",
    ".pdf",
    ".docx",
    ".pt",
    ".onnx",
    ".engine",
    ".mako",
    ".html",
    ".coverage",
    ".txt",
}


def style_for(path: Path) -> tuple[str, str | None] | None:
    if path.name.startswith("Dockerfile"):
        return ("#", None)
    if path.name == ".env.example":
        return ("#", None)
    suffix = path.suffix.lower()
    if suffix in {".py", ".sh", ".toml", ".yml", ".yaml", ".ini", ".conf"}:
        return ("#", None)
    if suffix in {".ts", ".tsx", ".js", ".jsx"}:
        return ("//", None)
    if suffix == ".css":
        return ("/*", "*/")
    if suffix == ".md":
        return ("<!--", "-->")
    return None


def render_header(open_token: str, _close_token: str | None = None) -> str:
    lines = [
        f"SPDX-FileCopyrightText: {YEAR} {OWNER}",
        f"SPDX-License-Identifier: {SPDX}",
    ]
    if open_token == "#":
        return "\n".join(f"# {line}" for line in lines) + "\n"
    if open_token == "//":
        return "\n".join(f"// {line}" for line in lines) + "\n"
    if open_token == "/*":
        return "/* " + lines[0] + "\n * " + lines[1] + " */\n"
    if open_token == "<!--":
        return "<!--\n" + "\n".join(lines) + "\n-->\n"
    raise ValueError(f"unknown comment style {open_token!r}")


def insert(text: str, header: str) -> str:
    lines = text.splitlines(keepends=True)
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
        if len(lines) > 1 and "coding" in lines[1] and lines[1].lstrip().startswith("#"):
            index = 2
    prefix = "".join(lines[:index])
    rest = "".join(lines[index:])
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    return f"{prefix}{header}\n{rest}"


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"])
    return [ROOT / item for item in output.decode().split("\0") if item]


def dirty_files() -> set[str]:
    changed: set[str] = set()
    for args in (
        ["diff", "--name-only", "-z"],
        ["diff", "--cached", "--name-only", "-z"],
    ):
        output = subprocess.check_output(["git", "-C", str(ROOT), *args])
        changed.update(item for item in output.decode().split("\0") if item)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Add SPDX copyright headers")
    parser.add_argument("--write", action="store_true", help="write the headers")
    parser.add_argument("--check", action="store_true", help="report missing headers")
    parser.add_argument(
        "--skip-dirty",
        action="store_true",
        help="do not touch files with uncommitted changes",
    )
    args = parser.parse_args()
    if not args.write and not args.check:
        parser.error("use --write or --check")

    skipped = dirty_files() if args.skip_dirty else set()
    missing: list[str] = []
    written = 0

    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative in skipped:
            continue
        if path.name in EXCLUDED_NAMES or relative.startswith(EXCLUDED_PREFIXES):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        style = style_for(path)
        if style is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "SPDX-FileCopyrightText" in text[:600]:
            continue

        missing.append(relative)
        if args.write:
            path.write_text(insert(text, render_header(*style)), encoding="utf-8")
            written += 1

    if args.check:
        if missing:
            print("Archivos sin header SPDX:")
            for relative in missing:
                print(f"  {relative}")
            print(f"\n{len(missing)} archivo(s) sin sellar.")
            return 1
        print("Todos los archivos tienen header SPDX.")
        return 0

    print(f"Headers escritos: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
