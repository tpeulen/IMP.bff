#!/usr/bin/env python3
"""Invoke the configured SWIG executable with Python thread support enabled."""

from __future__ import annotations

import argparse
import subprocess


def swig_command(executable: str, arguments: list[str]) -> list[str]:
    """Add thread support only to Python generation, retaining argv verbatim."""
    if "-python" in arguments and "-version" not in arguments:
        arguments = ["-threads", *(arg for arg in arguments if arg != "-threads")]
    return [executable, *arguments]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swig", required=True, help="Configured real SWIG executable")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    arguments = options.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    return subprocess.call(swig_command(options.swig, arguments))


if __name__ == "__main__":
    raise SystemExit(main())
