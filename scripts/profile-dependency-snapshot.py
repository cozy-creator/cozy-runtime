"""Measure local snapshot file I/O; run pinned before/after source on the same fixture."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

from cozy_runtime.internal import _installed_dependencies as dependencies


def io_counts() -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (
            row.split(":", 1) for row in Path("/proc/self/io").read_text().splitlines()
        )
    }


def venv(root: Path) -> Path:
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "bin/python").write_text("#!/bin/sh\nexit 77\n")
    (root / "pyvenv.cfg").write_text("version_info = 3.12\n")
    site = root / "lib/python3.12/site-packages"
    site.mkdir(parents=True, exist_ok=True)
    return site


def fixture(source: Path) -> None:
    if source.exists():
        return
    site = venv(source)
    info = site / "profile_dependency-1.0.dist-info"
    files = {
        "profile_dependency.py": b"def main():\n    return 0\n",
        f"{info.name}/METADATA": (
            b"Metadata-Version: 2.4\nName: profile-dependency\nVersion: 1.0\n"
        ),
        f"{info.name}/entry_points.txt": (
            b"[console_scripts]\nprofile-dependency = profile_dependency:main\n"
        ),
        "../../../bin/profile-dependency": b"#!/unavailable/source/python\n",
    }
    block = bytes(range(256)) * 4096
    files.update({f"data/part-{number:03}.bin": block for number in range(64)})
    rows = []
    for relative, body in files.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
        rows.append((relative, "sha256=" + digest, str(len(body))))
    with (info / "RECORD").open("w", newline="") as output:
        csv.writer(output).writerows([*rows, (f"{info.name}/RECORD", "", "")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--warm-source", action="store_true")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    fixture(args.source)
    venv(args.target)
    if args.warm_source:
        for path in args.source.rglob("*"):
            if path.is_file():
                with path.open("rb") as source:
                    while source.read(1 << 20):
                        pass
    before = io_counts()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.monotonic()
    if args.cache is None:
        manifest = dependencies.snapshot(args.source / "bin/python", args.target, set())
    else:
        manifest = dependencies.snapshot(
            args.source / "bin/python", args.target, set(), cache=args.cache
        )
    dependencies.verify_files(args.target, manifest)
    elapsed = time.monotonic() - started
    after = io_counts()
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    value = json.loads(manifest)
    print(
        json.dumps(
            {
                "elapsed_s": elapsed,
                "cpu_s": (
                    usage_after.ru_utime
                    + usage_after.ru_stime
                    - usage_before.ru_utime
                    - usage_before.ru_stime
                ),
                "process_io": {key: after[key] - before[key] for key in before},
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "captured_bytes": sum(row["length"] for row in value["files"]),
                "files": len(value["files"]),
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "source": str(args.source),
                "target": str(args.target),
                "warm_source": args.warm_source,
                "cache": str(args.cache) if args.cache is not None else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
