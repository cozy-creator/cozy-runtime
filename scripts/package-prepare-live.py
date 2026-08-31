#!/usr/bin/env python3
"""Weightless Marco-Polo package preparation through exact local wheel facts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
import sys
import tempfile
from pathlib import Path

from cozy_runtime.internal import canonical, descriptor, discovery, overlay
from cozy_runtime.internal.worker import package_prepare
from cozy_runtime.internal.worker.session import read_placement_set
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cozy-package-prepare-") as raw:
        root = Path(raw)
        wheels = root / "built"
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(wheels), "examples/marco-polo"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        wheel = next(wheels.glob("*.whl"))
        digest = hashlib.sha256(wheel.read_bytes()).digest()
        cache = root / "artifacts"
        view = cache / "sets" / "marco"
        view.mkdir(parents=True)
        local_wheel = view / wheel.name
        local_wheel.write_bytes(wheel.read_bytes())
        base = root / "WheelhouseManifest.json"
        base.write_bytes(
            canonical.write(
                {
                    "base_distributions": [
                        {
                            "distribution": "cozy-runtime",
                            "version": importlib.metadata.version("cozy-runtime"),
                        },
                        {
                            "distribution": "msgspec",
                            "version": importlib.metadata.version("msgspec"),
                        },
                    ],
                    "base_import_roots": [
                        {"providers": ["cozy-runtime"], "root": "cozy_runtime"},
                        {"providers": ["msgspec"], "root": "msgspec"},
                    ],
                    "compatibility_profile": {
                        "accelerator_build": "cpu",
                        "os_cpu": "linux-x86",
                        "python_abi": "cp312",
                        "torch_release": "2.13.0",
                    },
                    "format": "WheelhouseManifest/1",
                    "minimum_cuda_version": "none",
                    "minimum_driver_version": "none",
                    "platform_target": {
                        "accelerator_abi": "none",
                        "accelerator_backend": "none",
                        "libc": "glibc2.39",
                        "os_arch": "linux/amd64",
                        "python_abi": "cp312",
                    },
                    "wheels": [
                        {
                            "digest": "sha256:" + "0" * 64,
                            "distribution": "cozy-runtime",
                            "filename": "cozy_runtime-0.0.9-py3-none-any.whl",
                            "import_roots": ["cozy_runtime"],
                            "length": 1,
                            "tags": ["py3-none-any"],
                            "version": importlib.metadata.version("cozy-runtime"),
                        }
                    ],
                }
            )
        )
        release_digest = "sha256:" + hashlib.sha256(b"marco release").hexdigest()
        delegation = canonical.write(
            {
                "expires_at_unix": 2_000_000_000,
                "format": "cozy.worker.v1.DownloadDelegation/1",
                "models": [],
                "packages": [
                    {
                        "package": "paul/marco-polo",
                        "release": "1.0.3",
                        "release_digest": release_digest,
                    }
                ],
                "rental_id": "pr-marco",
                "worker_boot_id": "boot-marco",
                "worker_id": "worker-marco",
                "worker_tls_certificate_digest": "sha256:" + (b"t" * 32).hex(),
            }
        )
        request = pb.PreparePackageSetRequest(
            download_delegation=delegation,
            environment_root=str(root / "environments"),
            files=[
                pb.LocalDownloadFile(
                    digest=digest,
                    filename=wheel.name,
                    kind=pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
                    length=wheel.stat().st_size,
                    path=str(local_wheel),
                )
            ],
        )

        def describe(installed: overlay.InstalledOverlay, distribution: str) -> bytes:
            import_root = installed.import_root
            sys.path.append(str(import_root))
            try:
                return descriptor.canonical_bytes(
                    descriptor.build(discovery.discover_distribution(distribution))
                )
            finally:
                sys.path.remove(str(import_root))

        result = package_prepare.prepare_package_set(
            request,
            artifact_cache=cache,
            environment_root=root / "environments",
            python=Path(sys.executable),
            describe=describe,
            verified=lambda _digest, _length, _path: None,
            base_manifest=base,
        )
        value = documents.read(result.placement_set.placement_set_canonical_bytes, pb.PlacementSet)
        placement = value["placements"][0]
        assert placement["package"]["release_digest"] == release_digest
        assert placement["models"] == []
        assert [row["name"] for row in placement["entrypoints"]] == ["marco"]
        assert placement["entrypoints"][0]["slots"] == []
        assert result.placement_set.placement_set_digest == documents.digest_of(
            result.placement_set.placement_set_canonical_bytes
        )
        assert len(read_placement_set(result.placement_set)) == 1
        print(
            "package preparation: pure Marco wheel -> overlay -> derived descriptor -> "
            "weightless PlacementSet passed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
