#!/usr/bin/env python
"""Focused live proof for the trusted th-046 environment proof seat."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from cozy_runtime.internal import canonical
from cozy_runtime.internal.environment import WheelInventoryEntry
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

ROOT = Path(__file__).resolve().parent.parent
REQUEST_FORMAT = "cozy.runtime.EnvironmentProofRequest/1"


def main() -> None:
    helpers = runpy.run_path(str(ROOT / "scripts" / "environment-materializer-live.py"))
    build_wheel = cast(Callable[..., Path], helpers["build_wheel"])
    inventory = cast(Callable[..., WheelInventoryEntry], helpers["inventory"])
    writable_cleanup = cast(Callable[[Path], None], helpers["writable_cleanup"])
    root = Path(tempfile.mkdtemp(prefix="environment-proof-live-"))
    try:
        inputs = root / "inputs"
        inputs.mkdir()
        project_path = build_wheel(inputs, "proof-project", "1.0.0", "proof_project")
        resolved_path = build_wheel(inputs, "proof-resolved", "2.0.0", "proof_resolved")
        base_path = build_wheel(inputs, "proof-base", "3.0.0", "proof_base")
        project = inventory(project_path, "proof-project", "1.0.0", "project")
        resolved = inventory(resolved_path, "proof-resolved", "2.0.0", "bundle")
        base = inventory(base_path, "proof-base", "3.0.0", "wheelhouse")

        libc_name, libc_version = platform.libc_ver()
        target = {
            "accelerator_abi": "none",
            "accelerator_backend": "none",
            "libc": f"{libc_name}{libc_version}",
            "os_arch": "linux/amd64",
            "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        }
        wheelhouse = canonical.write(
            {
                "format": "WheelhouseManifest/1",
                "platform_target": target,
                "wheels": [_wheel(base)],
            }
        )
        resolved_set = canonical.write(
            {
                "format": "ResolvedWheelSet/1",
                "lock_digest": _digest(b"proof-lock"),
                "platform_target": target,
                "wheelhouse_manifest_digest": _digest(wheelhouse),
                "wheels": [_wheel(resolved)],
            }
        )
        descriptor = canonical.write({"format": "proof.EndpointDescriptor/1"})
        evaluated = canonical.write({"format": "proof.EvaluatedConfig/1"})
        bundle = canonical.write(
            {
                "descriptor": _ref(descriptor),
                "endpoint_release_id": "proof-release",
                "evaluated_config": _ref(evaluated),
                "format": "tensorhub.endpoint_bundle/1",
                "project_wheel": _wheel(project),
                "resolved_wheel_set": _ref(resolved_set),
                "wheelhouse_manifest": _ref(wheelhouse),
            }
        )
        environment_spec = documents.canonical_bytes(
            pb.EndpointEnvironmentSpec(
                platform_target=pb.PlatformTarget(**target),
                endpoint_bundle_digest=documents.digest_of(bundle),
                project_wheel_digest=documents.raw(project.digest),
                wheelhouse_manifest_digest=documents.digest_of(wheelhouse),
            )
        )
        paths = {
            "endpoint_environment_spec": _write(inputs / "environment.json", environment_spec),
            "endpoint_bundle": _write(inputs / "bundle.json", bundle),
            "resolved_wheel_set": _write(inputs / "resolved.json", resolved_set),
            "wheelhouse_manifest": _write(inputs / "wheelhouse.json", wheelhouse),
        }
        uv = Path(shutil.which("uv") or "")
        proof = Path(sys.executable).parent / "cozy-environment-proof"
        assert uv.is_absolute() and proof.is_file()
        base_request: dict[str, Any] = {
            **{name: str(path) for name, path in paths.items()},
            "format": REQUEST_FORMAT,
            "python": str(Path(sys.executable).resolve()),
            "uv": str(uv.resolve()),
            "wheel_files": {
                entry.digest: str(path.resolve())
                for entry, path in (
                    (project, project_path),
                    (resolved, resolved_path),
                    (base, base_path),
                )
            },
        }

        receipts: list[bytes] = []
        observations: list[dict[str, Any]] = []
        for name in ("cold", "rebuild"):
            request = {**base_request, "environment_root": str(root / f"environment-{name}")}
            request_path = root / f"request-{name}.json"
            request_path.write_text(json.dumps(request, indent=2))
            receipt = root / f"receipt-{name}.json"
            result = _run(proof, request_path, receipt)
            assert result.returncode == 0, result.stderr
            observation = json.loads(result.stdout)
            assert set(observation) == {"digest", "generation", "length", "reused"}
            assert observation["reused"] is False
            body = receipt.read_bytes()
            assert len(body) == observation["length"]
            assert _digest(body) == observation["digest"]
            receipts.append(body)
            observations.append(observation)
        assert receipts[0] == receipts[1]
        for observation in observations:
            python = Path(str(observation["generation"])) / "venv" / "bin" / "python"
            imported = subprocess.run(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import proof_base,proof_project,proof_resolved; print('installed')",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            assert imported.stdout.strip() == "installed"

        missing = dict(base_request)
        missing["environment_root"] = str(root / "environment-missing")
        missing["wheel_files"] = {
            digest: path
            for digest, path in cast(dict[str, str], base_request["wheel_files"]).items()
            if digest != resolved.digest
        }
        missing_path = root / "request-missing.json"
        missing_path.write_text(json.dumps(missing))
        missing_receipt = root / "receipt-missing.json"
        refused = _run(proof, missing_path, missing_receipt)
        assert refused.returncode == 6 and "wheel_grant_set_mismatch" in refused.stderr
        assert not missing_receipt.exists() and not (root / "environment-missing").exists()

        forbidden = dict(base_request)
        forbidden["environment_root"] = str(root / "environment-forbidden")
        forbidden["expected_receipt"] = str(root / "receipt-cold.json")
        forbidden_path = root / "request-forbidden.json"
        forbidden_path.write_text(json.dumps(forbidden))
        forbidden_receipt = root / "receipt-forbidden.json"
        refused = _run(proof, forbidden_path, forbidden_receipt)
        assert refused.returncode == 3 and "environment_proof_request_invalid" in refused.stderr
        assert not forbidden_receipt.exists() and not (root / "environment-forbidden").exists()

        print(
            "environment proof live: cold + independent rebuild produced identical exact "
            "receipts; isolated imports passed; missing wheel and expected-receipt input "
            f"refused; receipt={observations[0]['digest']}"
        )
    finally:
        writable_cleanup(root)


def _run(proof: Path, request: Path, receipt: Path) -> subprocess.CompletedProcess[str]:
    env = {
        "HOME": os.environ["HOME"],
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "PATH": os.environ["PATH"],
        "PIP_NO_INDEX": "1",
        "UV_OFFLINE": "1",
    }
    return subprocess.run(
        [str(proof), str(request), str(receipt)],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _wheel(entry: WheelInventoryEntry) -> dict[str, Any]:
    return {
        "digest": entry.digest,
        "distribution": entry.distribution,
        "filename": entry.filename,
        "length": entry.length,
        "tags": list(entry.tags),
        "version": entry.version,
    }


def _ref(raw: bytes) -> dict[str, Any]:
    return {"digest": _digest(raw), "length": len(raw)}


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    return path.resolve()


if __name__ == "__main__":
    main()
