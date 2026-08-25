#!/usr/bin/env python
"""Run the adversarial derivation corpus and print the scoreboard.

Not a test suite (#160): it spawns the real harness over real torch endpoint modules in a
real bounded sandbox, one child process per case, and prints what happened. The corpus is
the ONE place cr-004 is deliberately exhaustive — every other surface stays minimal.

    uv run --python corpus/.venv/bin/python scripts/derive-corpus.py [--only NAME]

A case FAILS the corpus when its verdict is `silent-wrong` (a derivation that disagrees
with reality), `untyped-error` (an escape the harness did not name), or simply not the
verdict the case declares. A refusal that names the wrong construct fails too.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MODULES = (
    "corpus.cases.baseline",
    "corpus.cases.devices",
    "corpus.cases.dynamics",
    "corpus.cases.mutation",
    "corpus.cases.opaque",
    "corpus.cases.lifecycle",
)
#: Two workers, never `-n auto` — this box is shared.
WORKERS = 2

FAIL = ("silent-wrong", "untyped-error")


def cases() -> list[tuple[str, Any]]:
    sys.path.insert(0, str(ROOT))
    found: list[tuple[str, Any]] = []
    for name in MODULES:
        module = importlib.import_module(name)
        found.extend((name, case) for case in module.CASES)
    return found


def run_one(module: str, name: str) -> dict[str, object]:
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, "-m", "corpus.run_case", module, name],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=300,
    )
    line = proc.stdout.strip().splitlines()[-1:] or [""]
    try:
        parsed: dict[str, object] = json.loads(line[0])
        return parsed
    except json.JSONDecodeError:
        return {
            "name": name,
            "verdict": "untyped-error",
            "named": f"exit {proc.returncode}",
            "detail": (proc.stderr.strip()[-400:] or proc.stdout.strip()[-400:]),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    found = [(m, c) for m, c in cases() if args.only in c.name]
    started = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda pair: run_one(pair[0], pair[1].name), found))

    width = max((len(c.name) for _, c in found), default=4)
    bad: list[str] = []
    print(f"{'case':<{width}}  {'verdict':<18} {'expected':<18} evidence")
    print("-" * (width + 78))
    for (_, case), got in zip(found, results, strict=True):
        verdict = str(got.get("verdict", "?"))
        named = str(got.get("named", ""))
        ok = verdict == case.expect and verdict not in FAIL
        if ok and case.names and case.names not in named:
            ok, verdict = False, f"{verdict}(misnamed)"
        evidence = (
            f"{got.get('destinations', '?')} dest, {got.get('derived_tensors', 0)} derived, "
            f"{str(got.get('digest', ''))[:19]}"
            if verdict == "derived-correct"
            else f"named {named!r}"
        )
        mark = " " if ok else "!"
        print(f"{mark}{case.name:<{width - 1}}  {verdict:<18} {case.expect:<18} {evidence}")
        if args.verbose or not ok:
            for key in ("detail", "divergence", "ground_truth_error", "traceback", "note"):
                if got.get(key) or (key == "note" and case.note):
                    print(f"      {key}: {got.get(key) or case.note}")
        if not ok:
            bad.append(case.name)

    elapsed = time.perf_counter() - started
    counts: dict[str, int] = {}
    for got in results:
        counts[str(got.get("verdict"))] = counts.get(str(got.get("verdict")), 0) + 1
    print("-" * (width + 78))
    print(f"{len(found)} cases in {elapsed:.1f}s — " + ", ".join(
        f"{v}: {n}" for v, n in sorted(counts.items())
    ))
    if bad:
        print(f"\nFAILED: {', '.join(bad)}")
        return 1
    print("\nno silent wrong derivations; every case ended on a named verdict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
