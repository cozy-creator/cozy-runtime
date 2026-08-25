#!/usr/bin/env python3
"""Time + memory benchmarks for the CLI and the author import. Real runs, no harness."""

from __future__ import annotations

import statistics
import subprocess
import sys
import time

RUNS = 20


# Peak RSS is measured in a FRESH helper process per case: ru_maxrss over children is
# cumulative, so measuring in-line would report the largest case for every later case.
HELPER = (
    "import resource,subprocess,sys;"
    "subprocess.run(sys.argv[1:],capture_output=True,check=True);"
    "print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)"
)


def timed(argv: list[str]) -> tuple[list[float], int]:
    """Wall-clock ms per run, plus that command's peak RSS in KiB."""
    samples: list[float] = []
    for _ in range(RUNS):
        start = time.perf_counter()
        subprocess.run(argv, capture_output=True, check=True)
        samples.append((time.perf_counter() - start) * 1000)
    probe = subprocess.run(
        [sys.executable, "-c", HELPER, *argv], capture_output=True, text=True, check=True
    )
    return samples, int(probe.stdout.strip())


def report(label: str, samples: list[float], rss_kib: int) -> None:
    print(
        f"{label:<34} min {min(samples):6.1f} ms  median {statistics.median(samples):6.1f} ms  "
        f"max {max(samples):6.1f} ms  peak RSS {rss_kib / 1024:5.1f} MiB"
    )


def main() -> int:
    py = sys.executable
    cases = [
        ("cozy-runtime version (cold start)", ["cozy-runtime", "version"]),
        ("cozy-runtime --help", ["cozy-runtime", "--help"]),
        ("import cozy_runtime.author", [py, "-c", "import cozy_runtime.author"]),
        ("import cozy_runtime.cli.main", [py, "-c", "import cozy_runtime.cli.main"]),
        ("bare python interpreter", [py, "-c", "pass"]),
    ]
    print(f"{RUNS} runs each; peak RSS measured per case in a fresh helper process")
    for label, argv in cases:
        report(label, *timed(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
