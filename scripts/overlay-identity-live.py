#!/usr/bin/env python3
# mypy: follow_imports=skip
"""cr-045/cr-049 cold, warm, corruption, stat, restart, and sharing proof."""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cozy_runtime.internal import canonical, child_env, native_wheel, overlay
from cozy_runtime.internal.worker import acquire, child
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
            ],
            "base_import_roots": [
                {"providers": ["base-control"], "root": "base_control"},
                {"providers": ["numpy"], "root": "numpy"},
            ],
            "compatibility_profile": {
                "accelerator_build": "none",
                "os_cpu": "linux-x86",
                "python_abi": "cp312",
                "torch_release": "2.13.0",
            },
            "format": "WheelhouseManifest/3",
            "minimum_cuda_version": "12.6",
            "minimum_driver_version": "560.28.03",
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
            root / "native-valid-build",
            "proof_native",
            "int proof_add(int a, int b) { return a + b; }\n",
        )
        native_custom = build_wheel(
            wheels,
            "proof-native-custom",
            "proof_native_custom",
            "native-custom",
            owner="custom",
            tag="cp312-cp312-manylinux_2_17_x86_64",
            root_is_pure=False,
            extra_entries={
                "proof_native_custom/_native.so": native_bytes,
                "proof_native_custom/fixture.py": (
                    b"import ctypes\nfrom pathlib import Path\n"
                    b"def run():\n"
                    b"    lib = ctypes.CDLL(str(Path(__file__).with_name('_native.so')))\n"
                    b"    return {'value': lib.proof_add(1, 2)}\n"
                ),
                "proof_native_custom/no_kernel.py": (
                    b"def run():\n"
                    b"    raise RuntimeError('no kernel image is available for execution')\n"
                ),
            },
        )
        native_installed = overlay.materialize(
            source("native-custom", family, project, native_custom),
            root / "native-custom-installed",
            python=Path(sys.executable),
            base_manifest=manifest,
        )
        native_inspection = native_wheel.inspect_tree(
            native_installed.import_root, ("proof_native_custom/_native.so",)
        )
        native_host = native_wheel.qualify_host(native_inspection, gpu_sm=None)
        assert native_host["qualification"] == "static-host-compatible"
        operator_evidence = native_wheel.qualify_operator(
            python=Path(sys.executable),
            overlay=native_installed.import_root,
            fixture="proof_native_custom.fixture:run",
            expected_result_digest=canonical.digest({"value": 3}),
            environment={},
        )
        assert operator_evidence["result_digest"] == canonical.digest({"value": 3})
        try:
            native_wheel.qualify_operator(
                python=Path(sys.executable),
                overlay=native_installed.import_root,
                fixture="proof_native_custom.fixture:run",
                expected_result_digest=canonical.digest({"value": 4}),
                environment={},
            )
        except native_wheel.NativeWheelRefusal as exc:
            assert exc.code == "overlay_native_operator_numerical_mismatch", exc.code
        else:
            raise AssertionError("native operator numerical mismatch was admitted")
        try:
            native_wheel.qualify_operator(
                python=Path(sys.executable),
                overlay=native_installed.import_root,
                fixture="proof_native_custom.no_kernel:run",
                expected_result_digest=canonical.digest({"value": 3}),
                environment={},
            )
        except native_wheel.NativeWheelRefusal as exc:
            assert exc.code == "overlay_native_no_kernel_image", exc.code
        else:
            raise AssertionError("no-kernel-image operator failure was admitted")

        realization_evidence: list[bytes] = []
        proof_python = Path(sys.prefix) / "bin" / "python"
        assert proof_python.is_file()
        for realization_kind, realization_digit in (("oci", "a"), ("managed-local", "b")):
            request_path = root / f"native-request-{realization_kind}.json"
            evidence_path = root / f"native-evidence-{realization_kind}.json"
            request_path.write_bytes(
                canonical.write(
                    {
                        "base_realization_digest": "sha256:" + realization_digit * 64,
                        "base_realization_kind": realization_kind,
                        "base_worker_profile": "torch2.13.0-none-cp312-linux-x86",
                        "device_index": None,
                        "endpoint_environment_spec_digest": digest(b"native-custom"),
                        "environment_root": str(root / "native-custom-installed"),
                        "expected_result_digest": canonical.digest({"value": 3}),
                        "fixture": "proof_native_custom.fixture:run",
                        "format": "cozy.runtime.NativeWheelProofRequest/1",
                        "installed_environment_receipt_digest": native_installed.receipt_digest,
                        "python": str(proof_python),
                        "wheelhouse_manifest": str(manifest),
                    }
                )
            )
            cli_result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from cozy_runtime.cli.native_wheel_proof import main; "
                    "raise SystemExit(main())",
                    str(request_path),
                    str(evidence_path),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=60,
                env={
                    "COZY_HOME": str(root / f"node-{realization_kind}"),
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                },
            )
            assert cli_result.returncode == 0, cli_result.stderr.decode()
            realization_evidence.append(evidence_path.read_bytes())
        assert realization_evidence[0] != realization_evidence[1]
        oci_evidence = canonical.parse_canonical(realization_evidence[0])
        local_evidence = canonical.parse_canonical(realization_evidence[1])
        assert isinstance(oci_evidence, dict) and isinstance(local_evidence, dict)
        assert (
            oci_evidence["installed_environment_receipt_digest"] == native_installed.receipt_digest
        )
        assert (
            local_evidence["installed_environment_receipt_digest"]
            == native_installed.receipt_digest
        )
        assert oci_evidence["overlay_content_digest"] == local_evidence["overlay_content_digest"]

        mutable_manifest = root / "mutable-base.json"
        mutable_manifest.write_bytes(manifest.read_bytes() + b" ")
        try:
            overlay.open_generation(
                environment,
                receipt_digest=cold.receipt_digest,
                environment_spec_digest=digest(b"spec-one"),
                base_manifest=mutable_manifest,
                python=Path(sys.executable),
            )
        except overlay.OverlayRefusal as exc:
            assert exc.code == "base_family_mismatch", exc.code
        else:
            raise AssertionError("byte-different local base borrowed the admitted receipt")
        assert child_env.erased("PYTHONPATH") and child_env.erased("PYTHONHOME")

        killed_custom = build_wheel(
            wheels,
            "proof-killed-custom",
            "proof_killed_custom",
            "killed-custom",
            owner="custom",
            extra_entries={
                f"proof_killed_custom/module_{index:05d}.py": f"VALUE = {index}\n".encode()
                for index in range(5_000)
            },
        )
        killed_root = root / "killed-installed"
        killed_source = source("killed", family, project, killed_custom)
        process = multiprocessing.get_context("fork").Process(
            target=overlay.materialize,
            args=(killed_source, killed_root),
            kwargs={"python": Path(sys.executable), "base_manifest": manifest},
        )
        process.start()
        deadline = time.monotonic() + 10
        abandoned: list[Path] = []
        while time.monotonic() < deadline and process.is_alive():
            abandoned = list(killed_root.glob(".*-staging-*"))
            if abandoned:
                break
            time.sleep(0.001)
        assert abandoned and process.is_alive(), "materialization finished before kill arm"
        process.kill()
        process.join(10)
        assert process.exitcode is not None and process.exitcode < 0
        assert not list((killed_root / "receipts").glob("*.json"))
        retried = overlay.materialize(
            killed_source,
            killed_root,
            python=Path(sys.executable),
            base_manifest=manifest,
        )
        assert retried.generation.is_dir()
        assert not list(killed_root.glob(".*-staging-*"))

        hostile_bytes = compile_shared(
            root / "native-rpath-build",
            "proof_rpath",
            "int proof(void) { return 1; }\n",
            "-Wl,-rpath,/tmp/hostile",
        )
        hostile_custom = build_wheel(
            wheels,
            "proof-hostile-rpath",
            "proof_hostile_rpath",
            "hostile-rpath",
            owner="custom",
            tag="cp312-cp312-manylinux_2_17_x86_64",
            root_is_pure=False,
            extra_entries={"proof_hostile_rpath/_native.so": hostile_bytes},
        )
        structural_native_reds = [("overlay_native_rpath_forbidden", hostile_custom)]
        for index, library in enumerate(
            (
                "libtorch_cuda.so",
                "libcublas.so.12",
                "libcublasLt.so.12",
                "libcudart.so.12",
                "libcusparse.so.12",
                "libcudnn.so.9",
                "libnvidia-ml.so.1",
            )
        ):
            package = f"proof_library_shadow_{index}"
            structural_native_reds.append(
                (
                    "overlay_native_base_library_shadow",
                    build_wheel(
                        wheels,
                        f"proof-library-shadow-{index}",
                        package,
                        f"library-shadow-{index}",
                        owner="custom",
                        tag="cp312-cp312-manylinux_2_17_x86_64",
                        root_is_pure=False,
                        extra_entries={f"{package}/{library}": native_bytes},
                    ),
                )
            )
        for index, (code, bad_custom) in enumerate(structural_native_reds):
            try:
                overlay.materialize(
                    source(code, family, project, bad_custom),
                    root / f"red-{code}-{index}",
                    python=Path(sys.executable),
                    base_manifest=manifest,
                )
            except overlay.OverlayRefusal as exc:
                assert exc.code == code, (exc.code, code)
            else:
                raise AssertionError(f"{code} was admitted")

        missing_bytes = compile_shared(
            root / "native-symbol-build",
            "proof_symbol",
            "extern int cozy_symbol_that_does_not_exist(void); "
            "int proof(void) { return cozy_symbol_that_does_not_exist(); }\n",
        )
        missing_path = root / "missing-symbol.so"
        missing_path.write_bytes(missing_bytes)
        missing_inspection = native_wheel.inspect_tree(root, (missing_path.name,))
        try:
            native_wheel.qualify_host(missing_inspection, gpu_sm=None)
        except native_wheel.NativeWheelRefusal as exc:
            assert exc.code == "overlay_native_symbol_unavailable", exc.code
        else:
            raise AssertionError("unavailable native symbol was admitted")

        gpu_inspection = native_wheel.Inspection(
            (
                native_wheel.ELFMember(
                    "proof_gpu/_native.so",
                    "",
                    (),
                    (),
                    (),
                    (),
                    (".nv_fatbin",),
                    (80,),
                    (),
                ),
            )
        )
        try:
            native_wheel.qualify_host(gpu_inspection, gpu_sm=90)
        except native_wheel.NativeWheelRefusal as exc:
            assert exc.code == "overlay_native_gpu_unsupported", exc.code
        else:
            raise AssertionError("wrong-SM native wheel was admitted")

        with patch("cozy_runtime.internal.native_wheel.shutil.which", return_value=None):
            try:
                native_wheel._cuda_images(missing_path)
            except native_wheel.NativeWheelRefusal as exc:
                assert exc.code == "overlay_native_gpu_inspector_unavailable", exc.code
            else:
                raise AssertionError("CUDA image inspection passed without cuobjdump")

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
            scope = Path(next(iter(jit_env.values()))).parent
            assert scope.parent == root / "node-local" / "jit-cache"
            assert scope.name == cold.content_digest.removeprefix("sha256:")
            assert all(Path(value).parent == scope for value in jit_env.values())
            assert all(child_env.erased(name) for name in jit_env)
            assert not any(path.is_file() for path in scope.rglob("*"))
        finally:
            supervision.close()

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
                    "native_project_wheel_refused": True,
                    "custom_source_build_input_refused": True,
                    "platform_specific_project_wheel_refused": True,
                    "node_local_jit_cache_scoped": True,
                    "jit_cache_bytes_unpublished": True,
                    "prebuilt_native_custom_wheel_admitted": True,
                    "hostile_rpath_refused": True,
                    "base_library_shadow_refused": True,
                    "nvidia_base_library_shadows_refused": True,
                    "unavailable_native_symbol_refused": True,
                    "wrong_sm_refused": True,
                    "missing_cuobjdump_refused": True,
                    "native_operator_executed": True,
                    "native_operator_numerical_mismatch_refused": True,
                    "native_operator_no_kernel_image_refused": True,
                    "oci_local_same_overlay_path": True,
                    "realization_fingerprints_separate": True,
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
