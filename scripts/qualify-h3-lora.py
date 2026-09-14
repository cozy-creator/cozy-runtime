#!/usr/bin/env python3
"""Qualify H3 through ordinary default-home `cozy run --lora` on an owned rental.

This never rents, installs a tool, restarts a daemon, or releases a machine. An operator
selects an already-prepared rental and records its expected GPU degree explicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

BASE = "paul/minimax-h3#sha256:d64f250c556b28889fb0c900643bf2028fa92cad619d8cc4b716d6145ca7d275"
PDD = (
    "paul/minimax-h3-turbo-lora#"
    "sha256:9921c17df2db67e8350769c490249b43659603ad780a67fc713d847a2ff194b0"
)
SPATIAL = (
    "paul/minimax-h3-spatial-physics-lora#"
    "sha256:7c5e4d22443d9537ba5befd79f133604e9caf68ed06ddcf645af4476bf60a742"
)
WUSHU = (
    "paul/minimax-h3-wushu-action-v7-lora#"
    "sha256:f8c8150d3ab44314b1d54085ec1bdf23c2d4b198edc160a98222bcde1d863cf7"
)
CASES = {
    "baseline": (),
    "zero": ((SPATIAL, "0"),),
    "spatial": ((SPATIAL, "0.6"),),
    "none_after": (),
    "wushu": ((WUSHU, "0.6"),),
    "spatial_repeat": ((SPATIAL, "0.6"),),
    "stack": ((SPATIAL, "0.3"), (WUSHU, "0.6")),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def verify(result_path: Path, events_path: Path, degree: int, active: bool) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    if result.get("status") != "completed" or not result.get("result", {}).get("video"):
        raise RuntimeError(f"generation did not complete: {result_path}")
    saved = {}
    for row in result.get("saved", []):
        path = Path(row["path"])
        if digest(path) != row["digest"]:
            raise RuntimeError(f"saved bytes differ from receipt: {path}")
        saved[row["output"]] = {"path": str(path), "digest": row["digest"]}
    if not {"video", "continuation_frame"} <= saved.keys():
        raise RuntimeError("video and continuation bytes were not retained")
    events = []
    logs = [*sorted(events_path.parent.glob("events.partial-*.jsonl")), events_path]
    for log in logs:
        for line in log.read_text().splitlines():
            # Reattachment may only replay the terminal result; previous runtime
            # observations remain evidence for this same idempotent command.
            with contextlib.suppress(json.JSONDecodeError):
                event = json.loads(line)
                if isinstance(event, dict):
                    events.append(event)
    facts = [
        row["fields"]
        for event in events
        for row in walk(event)
        if row.get("name") == "runtime model execution" and "fields" in row
    ]
    if not facts or int(facts[-1]["degree"]) != degree:
        raise RuntimeError(
            "actual Runtime GPU degree is absent or differs from the requested proof"
        )
    actual = facts[-1]
    if active and (actual["matmul_precision"] != "highest" or int(actual["lora_calls"]) <= 0):
        raise RuntimeError("nonzero LoRA did not execute under the FP32 precision contract")
    requests = sorted({event["request_id"] for event in events if event.get("request_id")})
    return {
        "execution": result.get("execution"),
        "queued": result.get("queued"),
        "machine": result.get("machine"),
        "request_ids": requests,
        "saved": saved,
        "runtime": actual,
    }


def retained_result(path: Path) -> bool:
    try:
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or document.get("status") != "completed":
            return False
        saved = {row["output"]: row for row in document.get("saved", [])}
        return all(
            name in saved
            and Path(saved[name]["path"]).is_file()
            and digest(Path(saved[name]["path"])) == saved[name]["digest"]
            for name in ("video", "continuation_frame")
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def preserve_partial_logs(root: Path) -> None:
    number = 1
    while any(root.glob(f"*.partial-{number:03d}.*")):
        number += 1
    for filename in ("result.json", "events.jsonl", "process.json"):
        path = root / filename
        if path.exists():
            path.rename(root / f"{path.stem}.partial-{number:03d}{path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cozy", default="cozy")
    parser.add_argument("--rental", required=True)
    parser.add_argument(
        "--degree",
        type=int,
        choices=(1, 2, 4),
        required=True,
        help="assert observed Runtime degree; machine width/placement selects devices",
    )
    parser.add_argument(
        "--key-prefix",
        required=True,
        help="unique cohort identity; do not reuse after changing software",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--task", action="append", choices=("fl2va", "ref2va", "fl2va_turbo", "ref2va_turbo")
    )
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--duration", type=int, choices=range(5, 16), default=5)
    parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument(
        "--prompt",
        default=(
            "A martial artist performs a fluid sequence of precise wushu movements in a sunlit "
            "courtyard. Natural body mechanics, stable anatomy, a steady camera, realistic "
            "lighting and movement."
        ),
    )
    args = parser.parse_args()
    default_home = Path.home() / ".cozy"
    if (
        Path(os.environ.get("COZY_HOME", str(default_home))).expanduser().resolve()
        != default_home.resolve()
    ):
        parser.error("qualification requires the user's default ~/.cozy")
    tasks = args.task or ["fl2va", "ref2va"]
    cases = args.case or list(CASES)
    if any(task.startswith("ref2va") for task in tasks) and not (
        args.reference and args.reference.is_file()
    ):
        parser.error("REF2VA requires --reference pointing to an actual input image")
    binary = shutil.which(args.cozy)
    if binary is None:
        parser.error("cozy executable is unavailable")
    args.out.mkdir(parents=True, exist_ok=True)
    cli_digest = digest(Path(binary))
    reports = {}
    for task in tasks:
        parameter = "base_model" if task.endswith("_turbo") else "model"
        component = task.removesuffix("_turbo") + "_dit"
        for case in cases:
            if digest(Path(binary)) != cli_digest:
                raise RuntimeError("cozy binary changed during qualification")
            label = f"{task}-{case}-d{args.degree}"
            root = args.out / label
            root.mkdir(exist_ok=True)
            payload = {"prompt": args.prompt, "seed": args.seed, "duration_s": args.duration}
            if not task.endswith("_turbo"):
                payload["steps"] = 30
            input_path = root / "input.json"
            argv = [
                binary,
                "run",
                f"paul/minimax-h3/{task}",
                f"model.{parameter}={BASE}",
            ]
            if task.endswith("_turbo"):
                argv.append(f"model.turbo_lora={PDD}")
            for ref, scale in CASES[case]:
                argv.append(f"--lora={parameter}:{component}={ref},{scale}")
            if task.startswith("ref2va"):
                argv.append(f"--asset={args.reference.resolve()}")
            argv.extend(
                [
                    f"--in={input_path.resolve()}",
                    f"--rental={args.rental}",
                    f"--idempotency-key={args.key_prefix}-{label}",
                    "--await",
                    "--json",
                    f"--out={(root / 'outputs').resolve()}",
                ]
            )
            command = {
                "argv": argv,
                "payload": payload,
                "expected_degree": args.degree,
                "cli_sha256": digest(Path(binary)),
                "reference_sha256": digest(args.reference) if task.startswith("ref2va") else None,
            }
            command_path = root / "command.json"
            if command_path.exists() and json.loads(command_path.read_text()) != command:
                raise RuntimeError(f"refusing to overwrite another qualification: {command_path}")
            write(input_path, payload)
            write(command_path, command)
            stdout, stderr = root / "result.json", root / "events.jsonl"
            if not retained_result(stdout):
                preserve_partial_logs(root)
                print(f"Starting/reattaching {label} on {args.rental}", flush=True)
                started = time.monotonic()
                with stdout.open("x") as output, stderr.open("x") as events:
                    completed = subprocess.run(argv, stdout=output, stderr=events, check=False)
                write(
                    root / "process.json",
                    {"exit_code": completed.returncode, "wall_seconds": time.monotonic() - started},
                )
                if completed.returncode:
                    raise RuntimeError(f"cozy failed; retained logs: {root}")
            report = verify(stdout, stderr, args.degree, bool(CASES[case] and case != "zero"))
            report["attention_verification"] = (
                "Retain this placement's Runtime boot attention line or triage plan.attention "
                "map; compare actual per-component kernels across every parity run. "
                "There is no global kernel pin: H3 audio_vae uses FP32/D256 attention, "
                "which BF16 FA3 cannot serve. This script does not prove the kernel map."
            )
            reports[label] = report
            write(root / "verification.json", report)
            print(f"Verified {label}: {report['execution']}", flush=True)
        for control, reference in (
            ("zero", "baseline"),
            ("none_after", "baseline"),
            ("spatial_repeat", "spatial"),
        ):
            key, other = f"{task}-{control}-d{args.degree}", f"{task}-{reference}-d{args.degree}"
            if key in reports and other in reports:
                for output in ("video", "continuation_frame"):
                    if (
                        reports[key]["saved"][output]["digest"]
                        != reports[other]["saved"][output]["digest"]
                    ):
                        raise RuntimeError(f"byte parity failed: {key} versus {other}, {output}")
    write(args.out / "verification.json", reports)


if __name__ == "__main__":
    main()
