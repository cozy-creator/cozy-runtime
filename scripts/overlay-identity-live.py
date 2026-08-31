#!/usr/bin/env python3
# mypy: follow_imports=skip
"""cr-045/cr-049 cold, warm, corruption, stat, restart, and sharing proof."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cozy_runtime.internal import (
    canonical,
    child_env,
    jit_cache,
    overlay,
)
from cozy_runtime.internal.worker import acquire, child, downloads
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
    root_is_pure: bool = True,
    extra_entries: dict[str, bytes] | None = None,
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
        f"{info}/WHEEL": (
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: {str(root_is_pure).lower()}\nTag: {tag}\n"
        ).encode(),
    }
    entries.update(extra_entries or {})
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


def compile_shared(root: Path, name: str, source: str, *linker_flags: str) -> bytes:
    """Test harness only: manufacture a prebuilt input before Runtime sees the wheel."""

    root.mkdir()
    source_path = root / f"{name}.c"
    library_path = root / f"{name}.so"
    source_path.write_text(source)
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(source_path), "-o", str(library_path), *linker_flags],
        stdin=subprocess.DEVNULL,
        check=True,
        timeout=30,
    )
    return library_path.read_bytes()


def base_manifest(path: Path) -> str:
    raw = canonical.write(
        {
            "base_distributions": [
                {"distribution": "base-control", "version": "1.0.0"},
                {"distribution": "numpy", "version": "2.0.0"},
                {"distribution": "torch", "version": "2.13.0+cpu"},
            ],
            "base_import_roots": [
                {"providers": ["base-control"], "root": "base_control"},
                {"providers": ["numpy"], "root": "numpy"},
                {"providers": ["torch"], "root": "torch"},
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
        environment_digest=digest(spec.encode()),
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
    plan: downloads.DownloadPlan,
    verified: acquire.VerifiedArtifactCache,
) -> acquire.Acquirer:
    return acquire.Acquirer(
        plan,
        cache_root=root / "cache",
        environment_root=root / "environment",
        selection_root=root / "selections",
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

        abi3_custom = build_wheel(
            wheels,
            "proof-abi3",
            "proof_abi3",
            "abi3",
            owner="custom",
            tag="cp310-abi3-manylinux_2_17_x86_64",
            requires_python=">=3.10",
            root_is_pure=False,
        )
        abi3 = overlay.materialize(
            source("abi3", family, project, abi3_custom),
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
                owner="custom",
                tag="cp312-cp312-manylinux_2_40_x86_64",
                root_is_pure=False,
            ),
            "overlay_python_incompatible": build_wheel(
                wheels,
                "proof-new-python",
                "proof_new_python",
                "new-python",
                owner="project",
                requires_python=">=3.15",
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
        for code, bad_wheel in incompatible_cases.items():
            bad_project = bad_wheel if bad_wheel[0].owner == "project" else project
            bad_custom = bad_wheel if bad_wheel[0].owner == "custom" else custom_one
            try:
                overlay.materialize(
                    source(code, family, bad_project, bad_custom),
                    root / f"red-{code}",
                    python=Path(sys.executable),
                    base_manifest=manifest,
                )
            except overlay.OverlayRefusal as exc:
                assert exc.code == code, (exc.code, code)
            else:
                raise AssertionError(f"{code} was admitted")

        native_project = build_wheel(
            wheels,
            "proof-native-project",
            "proof_native_project",
            "native-project",
            owner="project",
            extra_entries={"proof_native_project/_native.so": b"\x7fELFfake"},
        )
        build_input_custom = build_wheel(
            wheels,
            "proof-source-custom",
            "proof_source_custom",
            "source-custom",
            owner="custom",
            extra_entries={"proof_source_custom/kernels/op.cu": b"// source must not ship\n"},
        )
        native_reds = {
            "overlay_project_wheel_native": (native_project, custom_one),
            "overlay_native_build_input_forbidden": (project, build_input_custom),
        }
        for code, (bad_project, bad_custom) in native_reds.items():
            try:
                overlay.materialize(
                    source(code, family, bad_project, bad_custom),
                    root / f"red-{code}",
                    python=Path(sys.executable),
                    base_manifest=manifest,
                )
            except overlay.OverlayRefusal as exc:
                assert exc.code == code, (exc.code, code)
            else:
                raise AssertionError(f"{code} was admitted")

        tagged_project = replace(
            project[0],
            filename="proof_project-1.0.0-cp312-cp312-manylinux_2_17_x86_64.whl",
            tags=("cp312-cp312-manylinux_2_17_x86_64",),
        )
        try:
            overlay._verify_inventory(
                replace(
                    source("tagged-project", family, project, custom_one),
                    wheels=(tagged_project, custom_one[0]),
                ),
                overlay._read_base_manifest(manifest, family),
            )
        except overlay.OverlayRefusal as exc:
            assert exc.code == "overlay_project_wheel_not_pure", exc.code
        else:
            raise AssertionError("platform-specific project wheel was admitted")

        native_bytes = compile_shared(
            root / "native-refusal-build",
            "proof_native",
            "int proof_add(int a, int b) { return a + b;}\n",
        )
        native_custom = build_wheel(
            wheels,
            "proof-native-custom",
            "proof_native_custom",
            "native-custom",
            owner="custom",
            tag="cp312-cp312-manylinux_2_17_x86_64",
            root_is_pure=False,
            extra_entries={"proof_native_custom/_native.so": native_bytes},
        )
        try:
            overlay.materialize(
                source("native-custom", family, project, native_custom),
                root / "native-custom-refused",
                base_manifest=manifest,
            )
        except overlay.OverlayRefusal as exc:
            assert exc.code == "overlay_custom_wheel_native_unsupported", exc.code
        else:
            raise AssertionError("custom native wheel was admitted")

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

        supervision = child.ExecutorSupervision(
            root=root / "jit-supervision",
            python=sys.executable,
            base_env=(),
            cozy_home=root / "node-local",
        )
        try:
            assert supervision.use_overlay(
                str(cold.python), str(cold.import_root), cold.content_digest
            )
            jit_env = supervision._prepare_jit_cache()
            assert set(jit_env) == {name for name, _directory in child_env.JIT_CACHE_ENV}
            executor_scope = Path(next(iter(jit_env.values()))).parent
            relative = executor_scope.relative_to(root / "node-local" / "jit-cache")
            assert len(relative.parts) == 2
            assert relative.parts[1] == cold.content_digest.removeprefix("sha256:")
            assert all(Path(value).parent == executor_scope for value in jit_env.values())
            assert all(child_env.erased(name) for name in jit_env)
            assert not any(path.is_file() for path in executor_scope.rglob("*"))
            (executor_scope / "poison").write_text("owned only by this lifecycle")
        finally:
            supervision.close()
        assert not executor_scope.exists()

        tenant_a = jit_cache.scope(
            root / "tenant-cache",
            cold.content_digest,
            pod_scope="pod-a",
        )
        tenant_a_warm = jit_cache.scope(
            root / "tenant-cache",
            cold.content_digest,
            pod_scope="pod-a",
        )
        tenant_b = jit_cache.scope(
            root / "tenant-cache",
            cold.content_digest,
            pod_scope="pod-b",
        )
        try:
            assert tenant_a.root != tenant_b.root
            assert tenant_a.root == tenant_a_warm.root
            assert all(
                Path(value).is_relative_to(tenant_a.root) for value in tenant_a.environment.values()
            )
            assert all(
                Path(value).is_relative_to(tenant_b.root) for value in tenant_b.environment.values()
            )
            (tenant_a.root / "poison").write_text("tenant-a")
            assert (tenant_a_warm.root / "poison").read_text() == "tenant-a"
            assert not (tenant_b.root / "poison").exists()
            assert all(name in tenant_a.environment for name, _directory in child_env.JIT_CACHE_ENV)
        finally:
            jit_cache.remove_pod(root / "tenant-cache", "pod-a")
            jit_cache.remove_pod(root / "tenant-cache", "pod-b")

        cache_root = root / "acquisition"
        cache = cache_root / "cache"
        cache.mkdir(parents=True)
        cached = cache / project[0].digest.removeprefix("sha256:")
        cached.write_bytes(project[1].read_bytes())
        cached.chmod(0o400)
        expires = int(time.time()) + 60
        plan = downloads.DownloadPlan(
            server_time_unix=int(time.time()),
            expires_at_unix=int(time.time()) + 60,
            files={
                project[0].digest: downloads.DownloadFile(
                    project[0].digest,
                    project[0].length,
                    "wheel",
                    "https://r2.invalid/project.whl",
                    expires,
                )
            },
        )
        verified = acquire.VerifiedArtifactCache()
        first = acquirer(cache_root, plan, verified)
        first.fetch(project[0].digest, project[0].length)
        assert (first.verified_bytes, first.trusted_reuse_bytes) == (project[0].length, 0)
        second = acquirer(cache_root, plan, verified)
        second.fetch(project[0].digest, project[0].length)
        assert (second.verified_bytes, second.trusted_reuse_bytes) == (0, project[0].length)

        stat_before = cached.stat()
        os.utime(cached, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns + 1))
        changed_stat = acquirer(cache_root, plan, verified)
        changed_stat.fetch(project[0].digest, project[0].length)
        assert changed_stat.verified_bytes == project[0].length

        restarted = acquirer(cache_root, plan, acquire.VerifiedArtifactCache())
        restarted.fetch(project[0].digest, project[0].length)
        assert restarted.verified_bytes == project[0].length

        damaged = bytearray(cached.read_bytes())
        damaged[0] ^= 1
        cached.chmod(0o600)
        cached.write_bytes(damaged)
        cached.chmod(0o400)
        try:
            acquirer(cache_root, plan, verified).fetch(project[0].digest, project[0].length)
        except acquire.AcquisitionRefusal as exc:
            assert exc.code == "local_artifact_identity_mismatch"
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
                    "native_project_wheel_refused": True,
                    "custom_source_build_input_refused": True,
                    "platform_specific_project_wheel_refused": True,
                    "node_local_jit_cache_scoped": True,
                    "jit_cache_bytes_unpublished": True,
                    "jit_cache_worker_shutdown_removed": True,
                    "jit_cache_pod_scopes_separated": True,
                    "jit_cache_warm_reuse_within_pod": True,
                    "jit_cache_defaults_never_used": True,
                    "custom_native_wheel_refused": True,
                    "mutable_local_base_refused": True,
                    "source_path_injection_erased": True,
                    "killed_materialization_retry_atomic": True,
                    "base_version_conflict_refused": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
