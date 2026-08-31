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
import sysconfig
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from cozy_runtime.cli import private_preflight
from cozy_runtime.cli.io import CliError, Options
from cozy_runtime.internal import canonical, descriptor, discovery, overlay
from cozy_runtime.internal.worker import acquire, downloads, package_prepare
from cozy_runtime.internal.worker.plan import DeclaredBinding, JobBinding
from cozy_runtime.internal.worker.session import Worker, read_placement_set
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

ROOT = Path(__file__).resolve().parent.parent


def _describe_with_environment(
    installed: overlay.InstalledEnvironment, distribution: str
) -> bytes:
    code = (
        "import sys; sys.path.extend(sys.argv[2:]); "
        "from cozy_runtime.internal import descriptor,discovery; "
        "sys.stdout.buffer.write(descriptor.canonical_bytes("
        "descriptor.build(discovery.discover_distribution(sys.argv[1]))))"
    )
    result = subprocess.run(
        [
            str(installed.python),
            "-I",
            "-c",
            code,
            distribution,
            str(ROOT / "src"),
            sysconfig.get_path("purelib"),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-2000:])
    return result.stdout


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
                        "torch_release": "none",
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
        ]
        files.sort(key=lambda row: row.digest)
        verified: list[str] = []
        request = pb.PreparePackageSetRequest(
            download_delegation=delegation,
            environment_root=str(root / "environments"),
            files=files,
        )

        def describe(installed: overlay.InstalledEnvironment, distribution: str) -> bytes:
            return _describe_with_environment(installed, distribution)

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
        assert len(verified) == 1
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
        assert Path(job.python).is_file()
        assert Path(job.package_descriptor) == descriptor_path
        assert job.reclaim_on_terminal
        assert result.placement_set.placement_set_digest == documents.digest_of(
            result.placement_set.placement_set_canonical_bytes
        )
        assert len(read_placement_set(result.placement_set)) == 1
        private_source = hashlib.sha256(b"exact editable Marco source").digest()
        private_files = [row for row in files if row.kind == pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL]
        private_descriptor = descriptor.canonical_bytes(
            descriptor.build(discovery.discover(ROOT / "examples" / "marco-polo"))
        )
        revision = pb.PrivatePackageRevision(
            package="local/marco-polo",
            release="1.0.4",
            source_digest=private_source,
            package_descriptor=pb.Ref(
                digest=hashlib.sha256(private_descriptor).digest(),
                length=len(private_descriptor),
            ),
            files=[
                pb.PrivatePackageFileRef(
                    digest=row.digest,
                    filename=row.filename,
                    kind=row.kind,
                    length=row.length,
                )
                for row in private_files
            ],
        )
        _revision_raw, private_revision = documents.identity(revision)
        revision_path = root / "private-revision.json"
        revision_path.write_bytes(_revision_raw)
        descriptor_path = root / "private-descriptor.json"
        descriptor_path.write_bytes(private_descriptor)
        preflight_request = root / "private-preflight.json"
        preflight_request.write_text(
            json.dumps(
                {
                    "base_manifests": [
                        {
                            "digest": "sha256:" + hashlib.sha256(base.read_bytes()).hexdigest(),
                            "path": str(base),
                        }
                    ],
                    "descriptor": str(descriptor_path),
                    "revision": str(revision_path),
                    "wheels": [str(local_wheel)],
                },
                sort_keys=True,
            )
        )
        preflight = private_preflight.run([str(preflight_request)], Options())
        assert isinstance(preflight.document, dict)
        assert preflight.document["private_revision_digest"] == documents.spell(private_revision)
        assert len(preflight.document["compatible"]) == 1
        missing_source = json.loads(_revision_raw)
        missing_source.pop("source_digest")
        missing_source_path = root / "private-revision-missing-source.json"
        missing_source_path.write_bytes(canonical.write(missing_source))
        missing_request = json.loads(preflight_request.read_text())
        missing_request["revision"] = str(missing_source_path)
        missing_request_path = root / "private-preflight-missing-source.json"
        missing_request_path.write_text(json.dumps(missing_request, sort_keys=True))
        try:
            private_preflight.run([str(missing_request_path)], Options())
        except CliError as exc:
            assert exc.name == "private_revision_invalid", exc
        else:
            raise AssertionError("private preflight admitted an absent source digest")

        malformed_descriptor = b"{}"
        malformed_descriptor_path = root / "private-descriptor-malformed.json"
        malformed_descriptor_path.write_bytes(malformed_descriptor)
        malformed_revision = json.loads(_revision_raw)
        malformed_revision["package_descriptor"] = {
            "digest": "sha256:" + hashlib.sha256(malformed_descriptor).hexdigest(),
            "length": len(malformed_descriptor),
        }
        malformed_revision_path = root / "private-revision-malformed-descriptor.json"
        malformed_revision_path.write_bytes(canonical.write(malformed_revision))
        malformed_request = json.loads(preflight_request.read_text())
        malformed_request["descriptor"] = str(malformed_descriptor_path)
        malformed_request["revision"] = str(malformed_revision_path)
        malformed_request_path = root / "private-preflight-malformed-descriptor.json"
        malformed_request_path.write_text(json.dumps(malformed_request, sort_keys=True))
        try:
            private_preflight.run([str(malformed_request_path)], Options())
        except CliError as exc:
            assert exc.name == "private_revision_descriptor_invalid", exc
        else:
            raise AssertionError("private preflight admitted a malformed descriptor")
        private_result = package_prepare.prepare_private_package(
            pb.PreparePrivatePackageRequest(
                operation_id="private-marco-01",
                package=pb.DevelopmentPackage(
                    package="local/marco-polo",
                    release="1.0.4",
                    source_digest=private_source,
                    private_revision_digest=private_revision,
                ),
                files=private_files,
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
        assert private["development"]["private_revision_digest"] == documents.spell(
            private_revision
        )
        assert private["development"]["project_wheel"]["ref"]["digest"] == (
            "sha256:" + digest.hex()
        )
        assert private["environment"]["wheels"] == []
        assert private["entrypoints"]
        assert all(entrypoint["slots"] == [] for entrypoint in private["entrypoints"])
        assert private["models"] == []
        assert len(read_placement_set(private_result.placement_set)) == 1
        changed_refs = [pb.PrivatePackageFileRef() for _row in revision.files]
        for target, source_ref in zip(changed_refs, revision.files, strict=True):
            target.CopyFrom(source_ref)
        project_ref = next(
            row for row in changed_refs if row.kind == pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL
        )
        project_ref.digest = hashlib.sha256(b"different wheel bytes, same source").digest()
        changed_refs.sort(key=lambda row: row.digest)
        changed_revision_doc = pb.PrivatePackageRevision(
            package=revision.package,
            release=revision.release,
            source_digest=revision.source_digest,
            package_descriptor=revision.package_descriptor,
            files=changed_refs,
        )
        _changed_raw, changed_revision = documents.identity(changed_revision_doc)
        assert changed_revision != private_revision
        try:
            package_prepare.prepare_private_package(
                pb.PreparePrivatePackageRequest(
                    operation_id="private-marco-changed",
                    package=pb.DevelopmentPackage(
                        package="local/marco-polo",
                        release="1.0.4",
                        source_digest=private_source,
                        private_revision_digest=changed_revision,
                    ),
                    files=private_files,
                    environment_root=str(root / "private-changed-environments"),
                ),
                artifact_cache=cache,
                environment_root=root / "private-changed-environments",
                python=Path(sys.executable),
                describe=describe,
                verified=lambda digest, _length, _path: verified.append(digest),
                base_manifest=base,
            )
        except package_prepare.PreparationRefusal as exc:
            assert exc.code == "private_package_revision_mismatch", exc
        else:
            raise AssertionError("changed wheel bytes under one private revision were admitted")
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
        subprocess.run(
            [
                "uv",
                "build",
                "--wheel",
                "--out-dir",
                str(wheels),
                "proofs/fixtures/environment-resolved",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    else:
        wheels = wheel_dir
    wheel = next(wheels.glob("modeled_development_package-*.whl"))
    helper_wheel = next(wheels.glob("proof_environment_resolved-*.whl"))
    wheel_raw = wheel.read_bytes()
    wheel_digest = hashlib.sha256(wheel_raw).digest()
    view = cache / "sets" / "modeled"
    view.mkdir(parents=True)
    local_wheel = view / wheel.name
    local_wheel.write_bytes(wheel_raw)
    (cache / wheel_digest.hex()).hardlink_to(local_wheel)
    base_without_helper = base.read_bytes()
    base_with_helper = json.loads(base_without_helper)
    base_with_helper["base_distributions"].append(
        {"distribution": "proof-environment-resolved", "version": "2.0.0"}
    )
    base_with_helper["base_distributions"].sort(key=lambda row: row["distribution"])
    base_with_helper["base_import_roots"].append(
        {
            "providers": ["proof-environment-resolved"],
            "root": "proof_environment_resolved",
        }
    )
    base_with_helper["base_import_roots"].sort(key=lambda row: row["root"])
    _base, carried, _observed = overlay.preflight_overlay(
        local_wheel, (helper_wheel.resolve(),), base_without_helper
    )
    _base, reused, _observed = overlay.preflight_overlay(
        local_wheel, (helper_wheel.resolve(),), canonical.write(base_with_helper)
    )
    assert len(carried) == 2 and len(reused) == 1

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

    def describe(installed: overlay.InstalledEnvironment, distribution: str) -> bytes:
        return _describe_with_environment(installed, distribution)

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

    # The private transport prepares only code/descriptor/Environment. A later ordinary
    # PlacementSet joins one exact model without rebuilding or relabeling that code revision.
    private_source = hashlib.sha256(b"modeled private source").digest()
    descriptor_ref = value["package_descriptor"]
    private_files = [
        pb.LocalDownloadFile(
            digest=wheel_digest,
            filename=wheel.name,
            kind=pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
            length=len(wheel_raw),
            path=str(local_wheel),
        )
    ]
    private_revision_doc = pb.PrivatePackageRevision(
        package="proof/modeled-development-package",
        release="1.0.0",
        source_digest=private_source,
        package_descriptor=pb.Ref(
            digest=documents.raw(str(descriptor_ref["digest"])),
            length=int(descriptor_ref["length"]),
        ),
        files=[
            pb.PrivatePackageFileRef(
                digest=wheel_digest,
                filename=wheel.name,
                kind=pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
                length=len(wheel_raw),
            )
        ],
    )
    _private_revision_raw, private_revision = documents.identity(private_revision_doc)
    private_result = package_prepare.prepare_private_package(
        pb.PreparePrivatePackageRequest(
            operation_id="private-modeled-01",
            package=pb.DevelopmentPackage(
                package="proof/modeled-development-package",
                release="1.0.0",
                source_digest=private_source,
                private_revision_digest=private_revision,
            ),
            files=private_files,
            environment_root=str(root / "private-modeled-environments"),
        ),
        artifact_cache=cache,
        environment_root=root / "private-modeled-environments",
        python=Path(sys.executable),
        describe=describe,
        verified=mark,
        job_plan_root=root / "private-modeled-job-plans",
        base_manifest=base,
    )
    private_body = documents.read(
        private_result.placement_set.placement_set_canonical_bytes, pb.PlacementSet
    )["placements"][0]
    assert private_body["models"] == []
    assert private_body["entrypoints"] == []
    assert private_body["development"]["private_revision_digest"] == documents.spell(
        private_revision
    )
    private_delegation_doc = copy.deepcopy(delegation_doc)
    private_delegation_doc["packages"] = []
    private_delegation = canonical.write(private_delegation_doc)
    private_model_files = [
        row
        for row in local_files
        if row.kind
        in {
            pb.LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST,
            pb.LOCAL_DOWNLOAD_KIND_MODEL_OBJECT,
        }
    ]
    joined_result = package_prepare.prepare_private_placement(
        pb.PreparePrivatePlacementRequest(
            operation_id="private-modeled-01",
            private_revision_digest=private_revision,
            download_delegation=private_delegation,
            files=private_model_files,
        ),
        prepared=private_body,
        artifact_cache=cache,
        artifact_store=target_store,
        verified=mark,
    )
    parsed_dynamic = read_placement_set(joined_result.placement_set)[0]
    dynamic = parsed_dynamic
    assert parsed_dynamic["development"] == private_body["development"]
    assert parsed_dynamic["environment"] == private_body["environment"]
    assert parsed_dynamic["package_descriptor"] == private_body["package_descriptor"]
    assert parsed_dynamic["models"] == value["models"]
    assert parsed_dynamic["entrypoints"] == value["entrypoints"]

    changed_revision = bytearray(private_revision)
    changed_revision[0] ^= 1
    try:
        package_prepare.prepare_private_placement(
            pb.PreparePrivatePlacementRequest(
                operation_id="private-modeled-01",
                private_revision_digest=bytes(changed_revision),
                download_delegation=private_delegation,
                files=private_model_files,
            ),
            prepared=private_body,
            artifact_cache=cache,
            artifact_store=target_store,
            verified=mark,
        )
    except package_prepare.PreparationRefusal as exc:
        assert exc.code == "private_placement_identity_invalid"
    else:
        raise RuntimeError("changed private revision was joined to prepared code")

    same_revision_other_operation = cast(Any, object.__new__(Worker))
    same_revision_other_operation.preparation_lock = threading.Lock()
    same_revision_other_operation.private_operations = {
        ("private-modeled-02", documents.spell(private_revision)): SimpleNamespace(
            document=private_body
        )
    }
    same_revision_other_operation.verified_artifacts = verified
    same_revision_other_operation.options = SimpleNamespace(
        artifact_cache=cache, artifact_store=target_store
    )
    try:
        same_revision_other_operation.prepare_private_placement(
            pb.PreparePrivatePlacementRequest(
                operation_id="private-modeled-01",
                private_revision_digest=private_revision,
                download_delegation=private_delegation,
                files=private_model_files,
            )
        )
    except package_prepare.PreparationRefusal as exc:
        assert exc.code == "private_revision_unprepared"
    else:
        raise RuntimeError("operation A joined operation B's same-digest private core")

    wrong_package_doc = copy.deepcopy(private_delegation_doc)
    wrong_package_doc["models"][0]["package"] = "proof/another-package"
    try:
        package_prepare.prepare_private_placement(
            pb.PreparePrivatePlacementRequest(
                operation_id="private-modeled-01",
                private_revision_digest=private_revision,
                download_delegation=canonical.write(wrong_package_doc),
                files=private_model_files,
            ),
            prepared=private_body,
            artifact_cache=cache,
            artifact_store=target_store,
            verified=mark,
        )
    except package_prepare.PreparationRefusal as exc:
        assert exc.code == "package_prepare_model_selection_mismatch"
    else:
        raise RuntimeError("model binding for another package joined private code")

    def private_namespace(selected: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            document=selected,
            environment_digest=documents.raw(str(selected["environment_digest"])),
            package={},
            development=selected["development"],
            environment=selected["environment"],
            models=tuple(selected["models"]),
            package_descriptor=(
                str(selected["package_descriptor"]["digest"]),
                int(selected["package_descriptor"]["length"]),
            ),
            entrypoints=tuple(selected["entrypoints"]),
        )

    private_placement = private_namespace(parsed_dynamic)
    private_acquired = acquire.Acquirer(
        downloads.DownloadPlan(),
        cache_root=cache,
        environment_root=root / "private-modeled-environments",
        selection_root=root / "private-modeled-environments" / "selections",
        artifact_store=target_store,
        python=Path(sys.executable),
        base_manifest=base,
        verified_cache=verified,
    ).acquire(private_placement)
    private_binding = next(iter(private_acquired.bindings.values()))
    assert private_binding.package_revision_digest == documents.spell(private_revision)
    assert private_binding.development
    assert private_binding.reference_snapshot == manifest_digest
    absent = copy.deepcopy(dynamic)
    absent["models"] = []
    absent["bindings_digest"] = canonical.digest(
        {"entrypoints": absent["entrypoints"], "models": absent["models"]}
    )
    absent_bytes = canonical.write(
        {"format": "cozy.worker.v1.PlacementSet/1", "placements": [absent]}
    )
    try:
        read_placement_set(
            pb.DesiredPlacementSet(
                placement_set_digest=documents.digest_of(absent_bytes),
                placement_set_canonical_bytes=absent_bytes,
            )
        )
    except documents.DocumentError as exc:
        assert exc.code == "unknown_model", exc
    else:
        raise AssertionError("private modeled Placement admitted an absent model")

    wrong = copy.deepcopy(dynamic)
    wrong["models"][0]["manifest"]["digest"] = "sha256:" + "1" * 64
    wrong["bindings_digest"] = canonical.digest(
        {"entrypoints": wrong["entrypoints"], "models": wrong["models"]}
    )
    wrong_bytes = canonical.write(
        {"format": "cozy.worker.v1.PlacementSet/1", "placements": [wrong]}
    )
    wrong_selected = read_placement_set(
        pb.DesiredPlacementSet(
            placement_set_digest=documents.digest_of(wrong_bytes),
            placement_set_canonical_bytes=wrong_bytes,
        )
    )[0]
    try:
        acquire.Acquirer(
            downloads.DownloadPlan(),
            cache_root=cache,
            environment_root=root / "private-modeled-environments",
            selection_root=root / "private-modeled-environments" / "selections",
            artifact_store=target_store,
            python=Path(sys.executable),
            base_manifest=base,
            verified_cache=verified,
        ).acquire(private_namespace(wrong_selected))
    except acquire.AcquisitionRefusal as exc:
        assert exc.code in {"download_plan_missing", "selected_model_manifest_unavailable"}, exc
    else:
        raise AssertionError("private modeled Placement admitted a wrong model Manifest")

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
        pass


def _expect_modeled_refusal(
    root: Path,
    base: Path,
    cache: Path,
    artifact_store: Path,
    describe: Callable[[overlay.InstalledEnvironment, str], bytes],
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
