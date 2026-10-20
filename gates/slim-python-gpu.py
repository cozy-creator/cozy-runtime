# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.4,<0.31"]
# ///
"""Run Torch placement and TensorFS reader checks on an explicitly rented GPU."""

import os
import json
import re
import signal
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from io import BytesIO
from pathlib import Path
from typing import Annotated

from cozy_runtime.author import AssetBound, FileAsset, Outputs, Tree

FAILURE_BOUNDARY = b"# SLIM_REQUEST_FAILURE_PROOF\n"
FAILURE_MARKER = "PASS actual failed-request slack release with warm weights and cache preserved"


def physical_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("the CUDA proof did not report a physical UUID")
    normalized = value.lower().removeprefix("gpu-").replace("-", "")
    if re.fullmatch(r"[0-9a-f]{32}", normalized) is None:
        raise RuntimeError(f"the CUDA proof reported an invalid physical UUID: {value!r}")
    return normalized


def verify_failed_request(accounted: str, device_output: str) -> None:
    if accounted.count(FAILURE_MARKER) != 1:
        raise RuntimeError("the three failed-request cleanup cases did not finish")
    reports = []
    for line in accounted.split(FAILURE_MARKER, 1)[1].splitlines():
        try:
            report = json.loads(line)
        except ValueError:
            continue
        if isinstance(report, dict) and set(report) == {"cases", "gpu"}:
            reports.append(report)
    if len(reports) != 1:
        raise RuntimeError("failed-request cleanup must report one three-case result")
    cases = reports[0]["cases"]
    if (not isinstance(cases, list) or len(cases) != 3
            or any(not isinstance(case, dict) for case in cases)
            or [case.get("kind") for case in cases] != ["returned", "escaping", "success"]):
        raise RuntimeError("failed-request cleanup omitted its returned/escaping/success cases")
    expected = [physical_uuid(line.removeprefix("SLIM_DEVICE_UUID "))
                for line in device_output.splitlines() if line.startswith("SLIM_DEVICE_UUID ")]
    if len(expected) != 1 or physical_uuid(reports[0]["gpu"]) != expected[0]:
        raise RuntimeError("failed-request cleanup used a different physical GPU")


def main(
    *,
    wheel: Annotated[FileAsset, AssetBound(max_bytes=67108864)],
    runtime_wheel: Annotated[FileAsset, AssetBound(max_bytes=67108864)],
    tests: Annotated[FileAsset, AssetBound(max_bytes=131072)],
    graph_tests: Annotated[FileAsset, AssetBound(max_bytes=131072)],
    accounting: Annotated[FileAsset, AssetBound(max_bytes=131072)],
    conftest: Annotated[FileAsset, AssetBound(max_bytes=131072)],
    read_engine: Annotated[FileAsset, AssetBound(max_bytes=131072)],
    out: Outputs,
) -> Tree:
    root = Path(tempfile.mkdtemp(prefix="slim-python-gpu-"))

    def supplied_wheel(asset: FileAsset, project: str) -> Path:
        contents = asset.read_bytes()
        with zipfile.ZipFile(BytesIO(contents)) as archive:
            info = next(
                name.rsplit("/", 1)[0]
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            metadata = Parser().parsestr(archive.read(f"{info}/METADATA").decode())
            tags = Parser().parsestr(archive.read(f"{info}/WHEEL").decode())
            assert metadata["Name"] == project
            assert tags["Root-Is-Purelib"] == "false", "supply the native SDK wheel"
            artifact = root / f"{project.replace('-', '_')}-{metadata['Version']}-{tags['Tag']}.whl"
        artifact.write_bytes(contents)
        return artifact

    artifact = supplied_wheel(wheel, "tensorfs")
    runtime_artifact = supplied_wheel(runtime_wheel, "cozy-runtime")
    (root / "test_slim_varena_gpu.py").write_bytes(tests.read_bytes())
    (root / "test_graph_cache_gpu.py").write_bytes(graph_tests.read_bytes())
    accounting_source = accounting.read_bytes()
    failure_requested = FAILURE_BOUNDARY in accounting_source
    (root / "cuda_cache_accounting.py").write_bytes(accounting_source)
    (root / "conftest.py").write_bytes(conftest.read_bytes())
    (root / "cuda_read_engine.py").write_bytes(read_engine.read_bytes())
    logs = root / "logs"
    logs.mkdir()
    environment_vars = os.environ.copy()
    # Keep the assigned CUDA device; do not inherit an outer Python source overlay.
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment_vars.pop(key, None)

    def run(name: str, command: list[str]) -> str:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=environment_vars,
        )
        try:
            text, _ = process.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            text, _ = process.communicate()
            print(text, flush=True)
            raise RuntimeError(f"{name} exceeded the test-harness bound") from None
        (logs / f"{name}.log").write_text(text)
        print(text, flush=True)
        if process.returncode:
            raise RuntimeError(f"{name} failed with exit {process.returncode}:\n{text[-4000:]}")
        return text

    environment = root / "venv"
    run(
        "environment",
        ["uv", "venv", "--python", sys.executable, "--system-site-packages", str(environment)],
    )
    python = str(environment / "bin/python")
    run(
        "install",
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--reinstall-package",
            "tensorfs",
            "--reinstall-package",
            "cozy-runtime",
            "--torch-backend",
            "cu130",
            str(artifact),
            str(runtime_artifact),
            "pytest",
            "numpy>=2.0",
            "torch==2.14.0",
        ],
    )
    device_code = (
        "import sys, cozy_runtime, tensorfs, torch; "
        "assert cozy_runtime.__file__.startswith(sys.prefix); "
        "assert tensorfs.__file__.startswith(sys.prefix); "
        "assert torch.cuda.device_count() == 1; "
        "print(torch.cuda.get_device_properties(0))"
    )
    if failure_requested:
        device_code += "; print('SLIM_DEVICE_UUID ' + str(torch.cuda.get_device_properties(0).uuid))"
    device_output = run(
        "device",
        [
            python,
            "-c",
            device_code,
        ],
    )
    accounted = run("cache-accounting", [python, "cuda_cache_accounting.py"])
    if "CUDA cache accounting: owned growth, real leaks, symmetric shrink" not in accounted:
        raise RuntimeError("the CUDA cache accounting checks did not finish")
    if failure_requested:
        verify_failed_request(accounted, device_output)
    result = run(
        "tests",
        [
            python,
            "-m",
            "pytest",
            "--real-gpu",
            "-q",
            "-p",
            "no:cacheprovider",
            "test_slim_varena_gpu.py",
            "test_graph_cache_gpu.py",
        ],
    )
    counts = {
        kind: int(count)
        for count, kind in re.findall(r"(\d+) (passed|failed|skipped|xfailed|xpassed)", result)
    }
    if counts != {"passed": 8}:
        raise RuntimeError(f"expected eight executed tests (seven CUDA and one CPU), got {counts}")
    reader = run("read-engine", [python, "cuda_read_engine.py"])
    if "CUDA arrays: chunk-spanning reads, prebuilt decode, raw parts" not in reader:
        raise RuntimeError("the native CUDA reader did not finish its real consumer checks")
    return out.save_tree(logs)
