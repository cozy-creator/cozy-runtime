#!/usr/bin/env python3
# mypy: follow_imports=skip
"""cr-045/cr-049 cold, warm, corruption, stat, restart, and sharing proof."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cozy_runtime.internal import (
    base_observation,
    canonical,
    child_env,
    jit_cache,
    package_environment,
    placement_materialization,
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
    version: str = "1.0.0",
) -> tuple[package_environment.Wheel, Path]:
    filename = f"{distribution.replace('-', '_')}-{version}-{tag}.whl"
    wheel_root = root / f"{distribution}-{value}-{tag.replace('.', '_')}"
    wheel_root.mkdir()
    path = wheel_root / filename
    info = f"{distribution.replace('-', '_')}-{version}.dist-info"
    entries = {
        f"{package}/__init__.py": f"VALUE = {value!r}\n".encode(),
        f"{info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n"
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
        package_environment.Wheel(
            distribution=distribution,
            version=version,
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


def fixture_base(
    *, runtime_version: str | None = None, uv_version: str | None = None
) -> base_observation.Observation:
    observed = base_observation.current()
    distributions = dict(observed.distributions)
    if runtime_version is not None:
        distributions["cozy-runtime"] = runtime_version
    return replace(
        observed,
        distributions=tuple(sorted(distributions.items())),
        uv_version=uv_version or observed.uv_version,
    )


def source(
    spec: str,
    project: tuple[package_environment.Wheel, Path],
    custom: tuple[package_environment.Wheel, Path],
    *,
    verified: bool = True,
) -> package_environment.Input:
    rows = (project, custom)
    return package_environment.Input(
        environment_digest=digest(spec.encode()),
        project_wheel_digest=project[0].digest,
        wheels=tuple(row[0] for row in rows),
        grants=tuple(
            package_environment.BlobGrant(
                wheel.digest,
                wheel.length,
                path,
                package_environment.VerifiedFile.record(path, wheel.digest, wheel.length)
                if verified
                else None,
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
    with tempfile.TemporaryDirectory(prefix="cozy-package-environment-") as raw:
        root = Path(raw)
        wheels = root / "wheels"
        wheels.mkdir()
        base = fixture_base()
        project = build_wheel(
            wheels,
            "proof-project",
            "proof_project",
            "stable",
            owner="project",
            requires_dist=("cozy-runtime>=0.0.19,<1.0.0",),
        )
        custom_one = build_wheel(wheels, "proof-custom", "proof_custom", "one", owner="custom")
        custom_two = build_wheel(wheels, "proof-custom", "proof_custom", "two", owner="custom")
        environment = root / "installed"

        # Every arriving wheel already has fetch_exact's verified-file receipt. Admission
        # must not hash it again before uv installs the exact wheel set.
        with patch.object(
            package_environment, "_hash_file", side_effect=AssertionError("double hash")
        ):
            cold_started = time.perf_counter_ns()
            cold = package_environment.materialize(
                source("spec-one", project, custom_one),
                environment,
                python=Path(sys.executable),
                base=base,
            )
            cold_ns = time.perf_counter_ns() - cold_started
            warm_started = time.perf_counter_ns()
            warm = package_environment.materialize(
                source("spec-two", project, custom_one),
                environment,
                python=Path(sys.executable),
                base=base,
            )
            warm_ns = time.perf_counter_ns() - warm_started
            changed = package_environment.materialize(
                source("spec-three", project, custom_two),
                environment,
                python=Path(sys.executable),
                base=base,
            )

        assert cold.receipt_digest != warm.receipt_digest
        assert cold.content_digest == warm.content_digest
        assert cold.generation == warm.generation and warm.reused
        assert changed.content_digest != cold.content_digest
        assert (cold.site_packages / "proof_project" / "__init__.py").is_file()
        assert (changed.site_packages / "proof_project" / "__init__.py").is_file()
        assert (cold.site_packages / "proof_custom" / "__init__.py").read_bytes() != (
            changed.site_packages / "proof_custom" / "__init__.py"
        ).read_bytes()

        ordinary_v2 = build_wheel(
            wheels,
            "proof-custom",
            "proof_custom",
            "version-two",
            owner="custom",
            version="2.0.0",
        )
        stack_project_one = build_wheel(
            wheels,
            "stack-project-one",
            "stack_project_one",
            "one",
            owner="project",
            requires_dist=("proof-custom==1.0.0",),
        )
        stack_project_two = build_wheel(
            wheels,
            "stack-project-two",
            "stack_project_two",
            "two",
            owner="project",
            requires_dist=("proof-custom==2.0.0",),
        )
        stacked_root = root / "stacked"
        with ThreadPoolExecutor(max_workers=2) as pool:
            stacked = tuple(
                future.result()
                for future in (
                    pool.submit(
                        package_environment.materialize,
                        source("ordinary-v1", stack_project_one, custom_one),
                        stacked_root,
                        python=Path(sys.executable),
                        base=base,
                    ),
                    pool.submit(
                        package_environment.materialize,
                        source("ordinary-v2", stack_project_two, ordinary_v2),
                        stacked_root,
                        python=Path(sys.executable),
                        base=base,
                    ),
                )
            )
        assert stacked[0].generation != stacked[1].generation
        assert not list(stacked_root.glob(".environment-staging-*"))
        for installed, value in zip(stacked, ("one", "version-two"), strict=True):
            result = subprocess.run(
                [
                    str(installed.python),
                    "-I",
                    "-c",
                    "import proof_custom; print(proof_custom.VALUE)",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            assert result.stdout.strip() == value

        protected_distribution = build_wheel(
            wheels,
            "cozy-runtime",
            "cozy_runtime",
            "shadow",
            owner="custom",
            version=base.distribution_versions["cozy-runtime"],
        )
        protected_root = build_wheel(
            wheels,
            "different-runtime-name",
            "cozy_runtime",
            "shadow",
            owner="custom",
        )
        msgspec_shadow = build_wheel(
            wheels,
            "msgspec",
            "msgspec",
            "shadow",
            owner="custom",
            version=base.distribution_versions["msgspec"],
        )

        if "torch" not in base.distribution_versions:
            second_torch = build_wheel(
                wheels,
                "torch",
                "torch",
                "second-family",
                owner="custom",
                version="2.13.0",
            )

            def wheel_fact(row: package_environment.Wheel) -> dict[str, object]:
                return {
                    "distribution": row.distribution,
                    "filename": row.filename,
                    "import_roots": list(row.import_roots),
                    "ref": {"digest": row.digest, "length": row.length},
                    "tags": list(row.tags),
                    "version": row.version,
                }

            environment_body = {"wheels": [wheel_fact(second_torch[0])]}
            environment_digest = digest(
                canonical.write(
                    {
                        **environment_body,
                        "format": placement_materialization.ENVIRONMENT_FORMAT,
                    }
                )
            )
            placement = SimpleNamespace(
                development=None,
                environment=environment_body,
                environment_digest=bytes.fromhex(environment_digest.removeprefix("sha256:")),
                package={
                    "package": "proof/proof-project",
                    "project_wheel": wheel_fact(project[0]),
                    "release": "1.0.0",
                    "release_digest": digest(b"release"),
                },
            )
            before_download = acquirer(
                root / "second-family",
                downloads.DownloadPlan(),
                acquire.VerifiedArtifactCache(),
            )
            with patch.object(
                before_download,
                "fetch",
                side_effect=AssertionError("protected family reached download"),
            ):
                try:
                    before_download._environment(placement)
                except acquire.AcquisitionRefusal as exc:
                    assert exc.code == "package_environment_second_protected_family", exc.code
                else:
                    raise AssertionError("a second protected family reached package download")
        for code, shadow in (
            ("package_environment_shadows_base_distribution", protected_distribution),
            ("package_environment_shadows_base_import", protected_root),
            ("package_environment_shadows_base_distribution", msgspec_shadow),
        ):
            try:
                package_environment.materialize(
                    source(code, project, shadow),
                    root / code,
                    base=base,
                )
            except package_environment.EnvironmentRefusal as exc:
                assert exc.code == code, (exc.code, code)
            else:
                raise AssertionError(f"{code} was admitted")
        opened = package_environment.open_existing_environment(cold.python, "sha256:" + "f" * 64)
        assert opened.python == cold.python
        assert opened.generation == cold.generation

        interrupted = environment / ".environment-staging-2147483647-interrupted"
        interrupted.mkdir()
        (interrupted / "partial").write_text("not a selectable venv")
        recovered = package_environment.materialize(
            source("spec-after-interruption", project, custom_two),
            environment,
            python=Path(sys.executable),
            base=base,
        )
        assert recovered.reused and recovered.generation == changed.generation
        assert not interrupted.exists()
        assert not list(environment.glob(".environment-staging-*"))
        assert recovered.generation.stat().st_mode & 0o222 == 0

        patch_base = fixture_base(runtime_version="0.0.21")
        compatible_patch = package_environment.materialize(
            source("spec-runtime-patch", project, custom_one),
            environment,
            python=Path(sys.executable),
            base=patch_base,
        )
        assert compatible_patch.reused
        assert compatible_patch.generation == cold.generation

        incompatible_base = fixture_base(runtime_version="1.0.0")
        try:
            package_environment.materialize(
                source("spec-runtime-major", project, custom_one),
                environment,
                python=Path(sys.executable),
                base=incompatible_base,
            )
        except package_environment.EnvironmentRefusal as exc:
            assert exc.code == "package_environment_dependency_base_conflict"
            assert "cozy-runtime<1.0.0,>=0.0.19" in exc.detail
        else:
            raise AssertionError("an incompatible Runtime major reused the package environment")

        wrong_uv_base = fixture_base(uv_version="0.0.0")
        try:
            package_environment.materialize(
                source("spec-wrong-uv", project, custom_one),
                environment,
                python=Path(sys.executable),
                base=wrong_uv_base,
            )
        except package_environment.EnvironmentRefusal as exc:
            assert exc.code == "package_environment_uv_changed"
            assert "observed=0.0.0" in exc.detail
        else:
            raise AssertionError("an unpinned uv executable materialized a package environment")

        corrupt_path = root / "corrupt.whl"
        corrupt = bytearray(custom_one[1].read_bytes())
        corrupt[len(corrupt) // 2] ^= 1
        corrupt_path.write_bytes(corrupt)
        corrupt_source = source("corrupt", project, custom_one, verified=False)
        corrupt_source = replace(
            corrupt_source,
            grants=(
                corrupt_source.grants[0],
                replace(corrupt_source.grants[1], path=corrupt_path),
            ),
        )
        try:
            package_environment.materialize(
                corrupt_source,
                root / "corrupt-installed",
                python=Path(sys.executable),
                base=base,
            )
        except package_environment.EnvironmentRefusal as exc:
            assert exc.code == "package_environment_wheel_digest_mismatch"
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
        abi3 = package_environment.materialize(
            source("abi3", project, abi3_custom),
            root / "abi3-installed",
            python=Path(sys.executable),
            base=base,
        )
        assert (abi3.site_packages / "proof_abi3" / "__init__.py").is_file()

        incompatible_cases = {
            "package_environment_wheel_platform_incompatible": build_wheel(
                wheels,
                "proof-new-manylinux",
                "proof_new_manylinux",
                "new-manylinux",
                owner="custom",
                tag="cp312-cp312-manylinux_2_40_x86_64",
                root_is_pure=False,
            ),
            "package_environment_python_incompatible": build_wheel(
                wheels,
                "proof-new-python",
                "proof_new_python",
                "new-python",
                owner="project",
                requires_python=">=3.15",
            ),
            "package_environment_dependency_base_conflict": build_wheel(
                wheels,
                "proof-runtime-pin",
                "proof_runtime_pin",
                "runtime-pin",
                owner="project",
                requires_dist=("cozy-runtime==1.0.0",),
            ),
            "package_environment_dependency_missing": build_wheel(
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
                package_environment.materialize(
                    source(code, bad_project, bad_custom),
                    root / f"red-{code}",
                    python=Path(sys.executable),
                    base=base,
                )
            except package_environment.EnvironmentRefusal as exc:
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
            "package_environment_project_wheel_native": (native_project, custom_one),
            "package_environment_native_build_input_forbidden": (project, build_input_custom),
        }
        for code, (bad_project, bad_custom) in native_reds.items():
            try:
                package_environment.materialize(
                    source(code, bad_project, bad_custom),
                    root / f"red-{code}",
                    python=Path(sys.executable),
                    base=base,
                )
            except package_environment.EnvironmentRefusal as exc:
                assert exc.code == code, (exc.code, code)
            else:
                raise AssertionError(f"{code} was admitted")

        tagged_project = replace(
            project[0],
            filename="proof_project-1.0.0-cp312-cp312-manylinux_2_17_x86_64.whl",
            tags=("cp312-cp312-manylinux_2_17_x86_64",),
        )
        try:
            package_environment._verify_inventory(
                replace(
                    source("tagged-project", project, custom_one),
                    wheels=(tagged_project, custom_one[0]),
                ),
                base,
            )
        except package_environment.EnvironmentRefusal as exc:
            assert exc.code == "package_environment_project_wheel_not_pure", exc.code
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
            package_environment.materialize(
                source("native-custom", project, native_custom),
                root / "native-custom-refused",
                base=base,
            )
        except package_environment.EnvironmentRefusal as exc:
            assert exc.code == "package_environment_custom_wheel_native_unsupported", exc.code
        else:
            raise AssertionError("custom native wheel was admitted")

        native_local = package_environment.materialize(
            source("native-custom-local", project, native_custom),
            root / "native-custom-local",
            base=base,
            allow_native_custom=True,
        )
        library = ctypes.CDLL(
            str(native_local.site_packages / "proof_native_custom" / "_native.so")
        )
        library.proof_add.argtypes = (ctypes.c_int, ctypes.c_int)
        library.proof_add.restype = ctypes.c_int
        assert library.proof_add(20, 22) == 42

        false_marker = build_wheel(
            wheels,
            "proof-false-marker",
            "proof_false_marker",
            "false-marker",
            owner="project",
            requires_dist=("missing-package; python_version < '3.12'",),
        )
        package_environment.materialize(
            source("false-marker", false_marker, custom_one),
            root / "false-marker-installed",
            python=Path(sys.executable),
            base=base,
        )

        supervision = child.ExecutorSupervision(
            root=root / "jit-supervision",
            python=sys.executable,
            base_env=(),
            cozy_home=root / "node-local",
        )
        try:
            assert supervision.use_environment(str(cold.python), cold.content_digest)
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
                    "actual_base_python_abi": base.python_abi,
                    "actual_base_uv_version": base.uv_version,
                    "actual_protected_distributions": len(base.distributions),
                    "cold_ns": cold_ns,
                    "warm_ns": warm_ns,
                    "verified_bytes": first.verified_bytes,
                    "trusted_reuse_bytes": second.trusted_reuse_bytes,
                    "same_wheels_different_spec_one_content": True,
                    "existing_uv_venv_opened_without_install": True,
                    "compatible_runtime_patch_reused": True,
                    "incompatible_runtime_major_refused": True,
                    "wrong_uv_version_refused": True,
                    "one_changed_wheel_has_distinct_environment": True,
                    "incompatible_ordinary_versions_isolated": True,
                    "incompatible_ordinary_versions_concurrent_on_one_worker": True,
                    "second_protected_family_refused_before_download": True,
                    "protected_distribution_shadow_refused": True,
                    "protected_import_shadow_refused": True,
                    "runtime_abi_msgspec_shadow_refused": True,
                    "interrupted_staging_removed_before_reuse": True,
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
                    "custom_native_wheel_refused": True,
                    "custom_native_wheel_local_loaded": True,
                    "base_version_conflict_refused": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
