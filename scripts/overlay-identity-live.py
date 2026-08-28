#!/usr/bin/env python3
# mypy: follow_imports=skip
"""cr-045/cr-049 cold, warm, corruption, stat, restart, and sharing proof."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cozy_runtime.internal import canonical, overlay
from cozy_runtime.internal.worker import acquire
from cozy_runtime.protocol import worker_pb2 as pb


def digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def build_wheel(
    root: Path,
    distribution: str,
    package: str,
    value: str,
    *,
    owner: str,
    tag: str = "py3-none-any",
    requires_python: str = "",
    requires_dist: tuple[str, ...] = (),
) -> tuple[overlay.Wheel, Path]:
    filename = f"{distribution.replace('-', '_')}-1.0.0-{tag}.whl"
    wheel_root = root / f"{distribution}-{value}-{tag.replace('.', '_')}"
    wheel_root.mkdir()
    path = wheel_root / filename
    info = f"{distribution.replace('-', '_')}-1.0.0.dist-info"
    entries = {
        f"{package}/__init__.py": f"VALUE = {value!r}\n".encode(),
        f"{info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 1.0.0\n"
            + (f"Requires-Python: {requires_python}\n" if requires_python else "")
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires_dist)
        ).encode(),
        f"{info}/WHEEL": f"Wheel-Version: 1.0\nTag: {tag}\n".encode(),
    }
    record = []
    for name, raw in sorted(entries.items()):
        encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={encoded},{len(raw)}")
    record.append(f"{info}/RECORD,,")
    entries[f"{info}/RECORD"] = ("\n".join(record) + "\n").encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(entries.items()):
            archive.writestr(name, raw)
    raw = path.read_bytes()
    return (
        overlay.Wheel(
            distribution=distribution,
            version="1.0.0",
            filename=filename,
            digest=digest(raw),
            length=len(raw),
            tags=(tag,),
            owner=owner,
            import_roots=(package,),
        ),
        path,
    )


def base_manifest(path: Path) -> str:
    raw = canonical.write(
        {
            "base_distributions": [
                {"distribution": "base-control", "version": "1.0.0"},
                {"distribution": "numpy", "version": "2.0.0"},
            ],
            "base_import_roots": [
                {"providers": ["base-control"], "root": "base_control"},
                {"providers": ["numpy"], "root": "numpy"},
            ],
            "format": "WheelhouseManifest/2",
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
                    "distribution": "base-control",
                    "filename": "base_control-1.0.0-py3-none-any.whl",
                    "import_roots": ["base_control"],
                    "length": 1,
                    "tags": ["py3-none-any"],
                    "version": "1.0.0",
                }
            ],
        }
    )
    path.write_bytes(raw)
    path.chmod(0o444)
    return digest(raw)


def source(
    spec: str,
    family: str,
    project: tuple[overlay.Wheel, Path],
    custom: tuple[overlay.Wheel, Path],
    *,
    verified: bool = True,
) -> overlay.Input:
    rows = (project, custom)
    return overlay.Input(
        environment_spec_digest=digest(spec.encode()),
        base_family_digest=family,
        project_wheel_digest=project[0].digest,
        wheels=tuple(row[0] for row in rows),
        grants=tuple(
            overlay.BlobGrant(
                wheel.digest,
                wheel.length,
                path,
                overlay.VerifiedFile.record(path, wheel.digest, wheel.length) if verified else None,
            )
            for wheel, path in rows
        ),
    )


def acquirer(
    root: Path,
    grant: acquire.StandingGrant,
    verified: acquire.VerifiedArtifactCache,
) -> acquire.Acquirer:
    return acquire.Acquirer(
        grant,
        cache_root=root / "cache",
        environment_root=root / "environment",
        plans_root=root / "plans",
        documents_root=root / "documents",
        object_sets_root=root / "sets",
        artifact_store=root / "store",
        python=Path(sys.executable),
        verified_cache=verified,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cozy-overlay-identity-") as raw:
        root = Path(raw)
        wheels = root / "wheels"
        wheels.mkdir()
        manifest = root / "WheelhouseManifest.json"
        family = base_manifest(manifest)
        project = build_wheel(wheels, "proof-project", "proof_project", "stable", owner="project")
        custom_one = build_wheel(wheels, "proof-custom", "proof_custom", "one", owner="custom")
        custom_two = build_wheel(wheels, "proof-custom", "proof_custom", "two", owner="custom")
        environment = root / "installed"

        # Every arriving wheel already has fetch_exact's verified-file receipt. Overlay
        # inspection must not hash it again, even while extracting a new wheel pool.
        with patch.object(overlay, "_hash_file", side_effect=AssertionError("double hash")):
            cold_started = time.perf_counter_ns()
            cold = overlay.materialize(
                source("spec-one", family, project, custom_one),
                environment,
                python=Path(sys.executable),
                base_manifest=manifest,
            )
            cold_ns = time.perf_counter_ns() - cold_started
            warm_started = time.perf_counter_ns()
            warm = overlay.materialize(
                source("spec-two", family, project, custom_one),
                environment,
                python=Path(sys.executable),
                base_manifest=manifest,
            )
            warm_ns = time.perf_counter_ns() - warm_started
            changed = overlay.materialize(
                source("spec-three", family, project, custom_two),
                environment,
                python=Path(sys.executable),
                base_manifest=manifest,
            )

        assert cold.receipt_digest != warm.receipt_digest
        assert cold.content_digest == warm.content_digest
        assert cold.generation == warm.generation and warm.reused
        assert changed.content_digest != cold.content_digest
        cold_project = cold.import_root / "proof_project" / "__init__.py"
        changed_project = changed.import_root / "proof_project" / "__init__.py"
        cold_custom = cold.import_root / "proof_custom" / "__init__.py"
        changed_custom = changed.import_root / "proof_custom" / "__init__.py"
        assert cold_project.stat().st_ino == changed_project.stat().st_ino
        assert cold_custom.stat().st_ino != changed_custom.stat().st_ino

        corrupt_path = root / "corrupt.whl"
        corrupt = bytearray(custom_one[1].read_bytes())
        corrupt[len(corrupt) // 2] ^= 1
        corrupt_path.write_bytes(corrupt)
        corrupt_source = source("corrupt", family, project, custom_one, verified=False)
        corrupt_source = replace(
            corrupt_source,
            grants=(
                corrupt_source.grants[0],
                replace(corrupt_source.grants[1], path=corrupt_path),
            ),
        )
        try:
            overlay.materialize(
                corrupt_source,
                root / "corrupt-installed",
                python=Path(sys.executable),
                base_manifest=manifest,
            )
        except overlay.OverlayRefusal as exc:
            assert exc.code == "overlay_wheel_digest_mismatch"
        else:
            raise AssertionError("corrupt first arrival was admitted")

        abi3_project = build_wheel(
            wheels,
            "proof-abi3",
            "proof_abi3",
            "abi3",
            owner="project",
            tag="cp310-abi3-manylinux_2_17_x86_64",
            requires_python=">=3.10",
        )
        abi3 = overlay.materialize(
            source("abi3", family, abi3_project, custom_one),
            root / "abi3-installed",
            python=Path(sys.executable),
            base_manifest=manifest,
        )
        assert (abi3.import_root / "proof_abi3" / "__init__.py").is_file()

        incompatible_cases = {
            "overlay_wheel_platform_incompatible": build_wheel(
                wheels,
                "proof-new-manylinux",
                "proof_new_manylinux",
                "new-manylinux",
                owner="project",
                tag="cp312-cp312-manylinux_2_40_x86_64",
            ),
            "overlay_python_incompatible": build_wheel(
                wheels,
                "proof-new-python",
                "proof_new_python",
                "new-python",
                owner="project",
                requires_python=">=3.13",
            ),
            "overlay_dependency_base_conflict": build_wheel(
                wheels,
                "proof-numpy-pin",
                "proof_numpy_pin",
                "numpy-pin",
                owner="project",
                requires_dist=("numpy==1.0.0",),
            ),
            "overlay_dependency_missing": build_wheel(
                wheels,
                "proof-marker",
                "proof_marker",
                "marker",
                owner="project",
                requires_dist=("missing-package; python_version >= '3.12'",),
            ),
        }
        for code, bad_project in incompatible_cases.items():
            try:
                overlay.materialize(
                    source(code, family, bad_project, custom_one),
                    root / f"red-{code}",
                    python=Path(sys.executable),
                    base_manifest=manifest,
                )
            except overlay.OverlayRefusal as exc:
                assert exc.code == code, (exc.code, code)
            else:
                raise AssertionError(f"{code} was admitted")

        false_marker = build_wheel(
            wheels,
            "proof-false-marker",
            "proof_false_marker",
            "false-marker",
            owner="project",
            requires_dist=("missing-package; python_version < '3.12'",),
        )
        overlay.materialize(
            source("false-marker", family, false_marker, custom_one),
            root / "false-marker-installed",
            python=Path(sys.executable),
            base_manifest=manifest,
        )

        cache_root = root / "acquisition"
        cache = cache_root / "cache"
        cache.mkdir(parents=True)
        cached = cache / project[0].digest.removeprefix("sha256:")
        cached.write_bytes(project[1].read_bytes())
        cached.chmod(0o400)
        grant = acquire.StandingGrant(
            grant_id="proof",
            revision=1,
            subjects={project[0].digest: project[0].length},
            locations={project[0].digest: "https://artifacts.invalid/project.whl"},
            expires_at_unix=int(time.time()) + 60,
            base_url="https://artifacts.invalid",
        )
        verified = acquire.VerifiedArtifactCache()
        first = acquirer(cache_root, grant, verified)
        first.fetch(project[0].digest, project[0].length)
        assert (first.verified_bytes, first.trusted_reuse_bytes) == (project[0].length, 0)
        second = acquirer(cache_root, grant, verified)
        second.fetch(project[0].digest, project[0].length)
        assert (second.verified_bytes, second.trusted_reuse_bytes) == (0, project[0].length)

        stat_before = cached.stat()
        os.utime(cached, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns + 1))
        changed_stat = acquirer(cache_root, grant, verified)
        changed_stat.fetch(project[0].digest, project[0].length)
        assert changed_stat.verified_bytes == project[0].length

        restarted = acquirer(cache_root, grant, acquire.VerifiedArtifactCache())
        restarted.fetch(project[0].digest, project[0].length)
        assert restarted.verified_bytes == project[0].length

        damaged = bytearray(cached.read_bytes())
        damaged[0] ^= 1
        cached.chmod(0o600)
        cached.write_bytes(damaged)
        cached.chmod(0o400)
        try:
            acquirer(cache_root, grant, verified).fetch(project[0].digest, project[0].length)
        except acquire.AcquisitionRefusal as exc:
            assert exc.code == "local_artifact_digest_mismatch"
            assert exc.kind == pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH
        else:
            raise AssertionError("changed-stat corrupt cache bytes were trusted")

        print(
            json.dumps(
                {
                    "cold_ns": cold_ns,
                    "warm_ns": warm_ns,
                    "verified_bytes": first.verified_bytes,
                    "trusted_reuse_bytes": second.trusted_reuse_bytes,
                    "same_wheels_different_spec_one_content": True,
                    "one_changed_wheel_shared_project_inode": cold_project.stat().st_ino,
                    "changed_stat_rehashed": True,
                    "restart_rehashed": True,
                    "corrupt_first_arrival_refused": True,
                    "abi3_older_floor_admitted": True,
                    "manylinux_newer_floor_refused": True,
                    "requires_python_refused": True,
                    "active_marker_missing_dependency_refused": True,
                    "inactive_marker_ignored": True,
                    "base_version_conflict_refused": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
