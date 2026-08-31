#!/usr/bin/env python3
"""Weightless Marco-Polo package preparation through exact local wheel facts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cozy_runtime.internal import canonical, descriptor, discovery, overlay
from cozy_runtime.internal.worker import acquire, downloads, package_prepare
from cozy_runtime.internal.worker.plan import DeclaredBinding, JobBinding
from cozy_runtime.internal.worker.session import read_placement_set
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", type=Path)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="cozy-package-prepare-") as raw:
        root = Path(raw)
        if args.wheel_dir is None:
            wheels = root / "built"
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(wheels), "examples/marco-polo"],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        else:
            wheels = args.wheel_dir
        wheel = next(wheels.glob("marco_polo-*.whl"))
        digest = hashlib.sha256(wheel.read_bytes()).digest()
        cache = root / "artifacts"
        view = cache / "sets" / "marco"
        view.mkdir(parents=True)
        local_wheel = view / wheel.name
        local_wheel.write_bytes(wheel.read_bytes())
        redundant = view / (
            "msgspec-" + importlib.metadata.version("msgspec") + "-py3-none-any.whl"
        )
        redundant.write_bytes(b"base-owned carrier is deliberately not installed")
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
                        "release": "1.0.4",
                        "release_digest": release_digest,
                    }
                ],
                "rental_id": "pr-marco",
                "worker_boot_id": "boot-marco",
                "worker_id": "worker-marco",
                "worker_tls_certificate_digest": "sha256:" + (b"t" * 32).hex(),
            }
        )
        files = [
            pb.LocalDownloadFile(
                digest=digest,
                filename=wheel.name,
                kind=pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
                length=wheel.stat().st_size,
                path=str(local_wheel),
            ),
            pb.LocalDownloadFile(
                digest=hashlib.sha256(redundant.read_bytes()).digest(),
                filename=redundant.name,
                kind=pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL,
                length=redundant.stat().st_size,
                path=str(redundant),
            ),
        ]
        files.sort(key=lambda row: row.digest)
        verified: list[str] = []
        request = pb.PreparePackageSetRequest(
            download_delegation=delegation,
            environment_root=str(root / "environments"),
            files=files,
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
            artifact_store=root / "store",
            environment_root=root / "environments",
            python=Path(sys.executable),
            describe=describe,
            verified=lambda digest, _length, _path: verified.append(digest),
            job_plan_root=root / "job-plans",
            base_manifest=base,
        )
        value = documents.read(result.placement_set.placement_set_canonical_bytes, pb.PlacementSet)
        placement = value["placements"][0]
        assert placement["package"]["release_digest"] == release_digest
        assert placement["models"] == []
        assert placement["environment"]["wheels"] == []
        assert len(verified) == 2
        assert [row["name"] for row in placement["entrypoints"]] == ["marco"]
        assert placement["entrypoints"][0]["slots"] == []
        descriptor_path = cache / str(placement["package_descriptor"]["digest"]).removeprefix(
            "sha256:"
        )
        descriptor_body = descriptor.read_bytes(descriptor_path.read_bytes(), "Marco descriptor")
        job_id = descriptor.job_descriptor_id(descriptor_body, "marco_job")
        job = JobBinding.read(
            json.loads((root / "job-plans" / f"{job_id.removeprefix('sha256:')}.json").read_text())
        )
        assert job.job == "marco_job"
        assert job.build_id == release_digest
        assert Path(job.overlay).is_dir()
        assert Path(job.package_descriptor) == descriptor_path
        assert job.reclaim_on_terminal
        assert result.placement_set.placement_set_digest == documents.digest_of(
            result.placement_set.placement_set_canonical_bytes
        )
        assert len(read_placement_set(result.placement_set)) == 1
        private_source = hashlib.sha256(b"exact editable Marco source").digest()
        private_result = package_prepare.prepare_private_package(
            pb.PreparePrivatePackageRequest(
                operation_id="private-marco-01",
                package=pb.DevelopmentPackage(
                    package="local/marco-polo",
                    release="1.0.4",
                    source_digest=private_source,
                ),
                files=files,
                environment_root=str(root / "private-environments"),
            ),
            artifact_cache=cache,
            environment_root=root / "private-environments",
            python=Path(sys.executable),
            describe=describe,
            verified=lambda digest, _length, _path: verified.append(digest),
            job_plan_root=root / "private-job-plans",
            base_manifest=base,
        )
        private = documents.read(
            private_result.placement_set.placement_set_canonical_bytes, pb.PlacementSet
        )["placements"][0]
        assert private["development"]["source_digest"] == documents.spell(private_source)
        assert private["development"]["project_wheel"]["ref"]["digest"] == (
            "sha256:" + digest.hex()
        )
        assert private["environment"]["wheels"] == []
        assert len(read_placement_set(private_result.placement_set)) == 1
        modeled(root, base, cache, args.wheel_dir)
        print(
            "package preparation: published/private Marco and exact modeled TensorFS package passed"
        )
    return 0


def modeled(root: Path, base: Path, cache: Path, wheel_dir: Path | None) -> None:
    import tensorfs

    proof_support = ROOT / "proofs" / "internal"
    sys.path.insert(0, str(proof_support))
    try:
        from modeled_development_store import create
    finally:
        sys.path.remove(str(proof_support))

    if wheel_dir is None:
        wheels = root / "modeled-built"
        subprocess.run(
            [
                "uv",
                "build",
                "--wheel",
                "--out-dir",
                str(wheels),
                "proofs/fixtures/modeled-development-package",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    else:
        wheels = wheel_dir
    wheel = next(wheels.glob("modeled_development_package-*.whl"))
    wheel_raw = wheel.read_bytes()
    wheel_digest = hashlib.sha256(wheel_raw).digest()
    view = cache / "sets" / "modeled"
    view.mkdir(parents=True)
    local_wheel = view / wheel.name
    local_wheel.write_bytes(wheel_raw)
    (cache / wheel_digest.hex()).hardlink_to(local_wheel)

    source_root = root / "modeled-source"
    model = create(source_root, tensor=b"\x80\x3f" * 4)
    source = tensorfs.Store.open(str(source_root))
    manifest_digest = str(model["manifest_digest"])
    manifest = bytes(source.manifest(manifest_digest)["manifest"])
    manifest_path = view / f"manifest-{manifest_digest.removeprefix('sha256:')}.json"
    manifest_path.write_bytes(manifest)
    local_files = [
        pb.LocalDownloadFile(
            digest=wheel_digest,
            filename=wheel.name,
            kind=pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
            length=len(wheel_raw),
            path=str(local_wheel),
        ),
        pb.LocalDownloadFile(
            digest=documents.raw(manifest_digest),
            filename=manifest_path.name,
            kind=pb.LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST,
            length=len(manifest),
            path=str(manifest_path),
        ),
    ]
    for reached in source.walk(manifest_digest):
        digest = str(reached["id"])
        length = int(reached["length"])
        path = view / f"object-{digest.removeprefix('sha256:')}.bin"
        path.write_bytes(bytes(source.document(digest, max_bytes=length)))
        local_files.append(
            pb.LocalDownloadFile(
                digest=documents.raw(digest),
                filename=path.name,
                kind=pb.LOCAL_DOWNLOAD_KIND_MODEL_OBJECT,
                length=length,
                path=str(path),
            )
        )
    local_files.sort(key=lambda row: row.digest)

    release_digest = "sha256:" + hashlib.sha256(b"modeled release").hexdigest()
    delegation_doc: dict[str, Any] = {
        "expires_at_unix": 2_000_000_000,
        "format": "cozy.worker.v1.DownloadDelegation/1",
        "models": [
            {
                "manifest": manifest_digest,
                "model": "proof/tiny-core",
                "package": "proof/modeled-development-package",
                "release": "1.0.0",
                "slot": "render.models.model",
            }
        ],
        "packages": [
            {
                "package": "proof/modeled-development-package",
                "release": "1.0.0",
                "release_digest": release_digest,
            }
        ],
        "rental_id": "pr-modeled",
        "worker_boot_id": "boot-modeled",
        "worker_id": "worker-modeled",
        "worker_tls_certificate_digest": "sha256:" + (b"m" * 32).hex(),
    }
    delegation = canonical.write(delegation_doc)
    request = pb.PreparePackageSetRequest(
        download_delegation=delegation,
        environment_root=str(root / "environments"),
        files=local_files,
    )
    verified = acquire.VerifiedArtifactCache()

    def mark(digest: str, length: int, path: Path) -> None:
        verified.record(digest, length, path)

    def describe(installed: overlay.InstalledOverlay, distribution: str) -> bytes:
        sys.path.append(str(installed.import_root))
        try:
            return descriptor.canonical_bytes(
                descriptor.build(discovery.discover_distribution(distribution))
            )
        finally:
            sys.path.remove(str(installed.import_root))

    target_store = root / "modeled-target"
    result = package_prepare.prepare_package_set(
        request,
        artifact_cache=cache,
        artifact_store=target_store,
        environment_root=root / "environments",
        python=Path(sys.executable),
        describe=describe,
        verified=mark,
        base_manifest=base,
    )
    body = documents.read(result.placement_set.placement_set_canonical_bytes, pb.PlacementSet)
    value = body["placements"][0]
    assert value["models"] == [
        {
            "id": "model-0000",
            "manifest": {"digest": manifest_digest, "length": len(manifest)},
            "repo": "proof/tiny-core",
            "version": "1.0.0",
        }
    ]
    slot = value["entrypoints"][0]["slots"][0]
    assert slot["slot"] == "model"
    assert "model_construction_contract" not in slot
    assert slot["config"]["document"]["digest"] == model["config_digest"]

    placement = SimpleNamespace(
        environment_digest=documents.raw(str(value["environment_digest"])),
        package=value["package"],
        environment=value["environment"],
        models=tuple(value["models"]),
        package_descriptor=(
            str(value["package_descriptor"]["digest"]),
            int(value["package_descriptor"]["length"]),
        ),
        entrypoints=tuple(value["entrypoints"]),
    )
    acquired = acquire.Acquirer(
        downloads.DownloadPlan(),
        cache_root=cache,
        environment_root=root / "environments",
        selection_root=root / "environments" / "selections",
        artifact_store=target_store,
        python=Path(sys.executable),
        base_manifest=base,
        verified_cache=verified,
    ).acquire(placement)
    binding = next(iter(acquired.bindings.values()))
    assert binding.reference_snapshot == manifest_digest
    assert binding.config_digest == model["config_digest"]
    assert binding.components == ("core",)
    assert binding.expected_logical_model_digest == ""
    _execute_modeled(acquired, binding, root)

    wrong_slot = copy.deepcopy(delegation_doc)
    wrong_slot["models"][0]["slot"] = "render.models.other"
    _expect_modeled_refusal(
        root,
        base,
        cache,
        target_store,
        describe,
        wrong_slot,
        local_files,
        "package_prepare_model_selection_mismatch",
    )
    wrong_manifest = copy.deepcopy(delegation_doc)
    wrong_manifest["models"][0]["manifest"] = "sha256:" + "1" * 64
    _expect_modeled_refusal(
        root,
        base,
        cache,
        target_store,
        describe,
        wrong_manifest,
        local_files,
        "package_prepare_model_manifest_set_mismatch",
    )
    fake_manifest = "sha256:" + "2" * 64
    wrong_digest = copy.deepcopy(delegation_doc)
    wrong_digest["models"][0]["manifest"] = fake_manifest
    changed_files = [_copy_file(row) for row in local_files]
    for row in changed_files:
        if row.kind == pb.LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST:
            row.digest = documents.raw(fake_manifest)
    changed_files.sort(key=lambda row: row.digest)
    _expect_modeled_refusal(
        root,
        base,
        cache,
        root / "modeled-wrong-digest",
        describe,
        wrong_digest,
        changed_files,
        "package_prepare_model_refused",
    )
    foreign_raw = b"foreign model object"
    foreign_digest = hashlib.sha256(foreign_raw).digest()
    foreign = view / ("object-" + foreign_digest.hex() + ".bin")
    foreign.write_bytes(foreign_raw)
    extra_files = [_copy_file(row) for row in local_files]
    extra_files.append(
        pb.LocalDownloadFile(
            digest=foreign_digest,
            filename="object-" + foreign_digest.hex() + ".bin",
            kind=pb.LOCAL_DOWNLOAD_KIND_MODEL_OBJECT,
            length=foreign.stat().st_size,
            path=str(foreign),
        )
    )
    extra_files.sort(key=lambda row: row.digest)
    _expect_modeled_refusal(
        root,
        base,
        cache,
        root / "modeled-cross-object",
        describe,
        delegation_doc,
        extra_files,
        "package_prepare_model_object_set_mismatch",
    )


def _copy_file(row: pb.LocalDownloadFile) -> pb.LocalDownloadFile:
    copy_ = pb.LocalDownloadFile()
    copy_.CopyFrom(row)
    return copy_


def _execute_modeled(
    acquired: acquire.AcquiredPlacement, binding: DeclaredBinding, root: Path
) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    if not torch.cuda.is_available():
        return
    from cozy_runtime.internal import executor as runtime_executor

    class Sink:
        def send(self, _frame: object) -> None:
            return

    installed = acquired.installed
    assert installed is not None
    sys.path.append(str(installed.import_root))
    try:
        runtime_executor._capture_seal()
        executor = runtime_executor.Executor(Sink(), root / "direct-executor")  # type: ignore[arg-type]
        selected = binding
        reply = executor.prepare(
            {
                "binding": {
                    "application": selected.application,
                    "package_descriptor": selected.descriptor_path,
                    "development_project": "",
                    "model_class": selected.model_class,
                    "model_binding_path": selected.model_binding_path,
                    "model_parameter_name": selected.model_parameter_name,
                    "component": selected.components[0],
                    "components": list(selected.components),
                    "snapshots": dict(selected.snapshots),
                    "store": selected.store,
                    "config": selected.config,
                    "snapshot": selected.reference_snapshot,
                    "release": selected.release,
                    "variant": "",
                    "development": False,
                    "stamps": {},
                    "custody": "canonical",
                    "strict_keys": False,
                    "variants": [],
                    "objective": "latency",
                    "steps_basis": 20,
                },
                "budgets": {
                    "declared_weight_bytes": selected.model_weight_bytes,
                    "pinned": 0,
                    "resident": None,
                    "vram_floor": 0,
                },
                "expect_construction_digest": "",
            }
        )
        assert reply.get("ok"), reply
        request_id = "modeled-direct#1"
        prepared = executor.prepare_request(
            {"request_id": request_id, "entrypoint": "render", "payload": {"value": 41}}
        )
        assert prepared.get("ok"), prepared
        spool = root / "direct-spool"
        spool.mkdir()
        outcome = executor.invoke(
            {
                "request_id": request_id,
                "entrypoint": "render",
                "deadline_s": 60,
                "boot_warmup": False,
                "spool": str(spool),
                "placement": "all_resident",
                "headroom_bytes": 0,
                "inputs": {},
                "max_input_bytes": 64 << 20,
            }
        )
        assert outcome["outcome"]["terminal"] == "succeeded", outcome
        assert runtime_executor._read_result(spool) == {"value": 42}
    finally:
        sys.path.remove(str(installed.import_root))


def _expect_modeled_refusal(
    root: Path,
    base: Path,
    cache: Path,
    artifact_store: Path,
    describe: Callable[[overlay.InstalledOverlay, str], bytes],
    delegation: dict[str, Any],
    files: list[pb.LocalDownloadFile],
    want: str,
) -> None:
    request = pb.PreparePackageSetRequest(
        download_delegation=canonical.write(delegation),
        environment_root=str(root / "environments"),
        files=files,
    )
    try:
        package_prepare.prepare_package_set(
            request,
            artifact_cache=cache,
            artifact_store=artifact_store,
            environment_root=root / "environments",
            python=Path(sys.executable),
            describe=describe,
            verified=lambda _digest, _length, _path: None,
            base_manifest=base,
        )
    except package_prepare.PreparationRefusal as exc:
        assert exc.code == want, (exc.code, want, exc.detail)
        return
    raise AssertionError(f"{want} was admitted")


if __name__ == "__main__":
    raise SystemExit(main())
