from __future__ import annotations

import dataclasses
import os
import subprocess
from pathlib import Path

import pytest

from cozy_runtime.internal import package_environment as environment
from test_unpublished_exact_dependencies import _material, _wheel


def test_conflicting_complete_closures_ignore_base_and_ambient_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = environment.observe_base()
    base = dataclasses.replace(
        base, distributions=(*base.distributions, ("huggingface-hub", "1.16.1"))
    )
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "ambient_sentinel.py").write_text("VALUE=999")
    monkeypatch.setenv("PYTHONPATH", str(poison))
    monkeypatch.setenv("UV_CONSTRAINT", str(poison / "missing-constraints"))
    root = tmp_path / "envs"
    held = []
    for version in ("1.23.0", "1.16.1"):
        wheel_root = tmp_path / version
        wheel_root.mkdir()
        dependency = _wheel(
            wheel_root, "huggingface_hub", (), f"VALUE={version!r}\n".encode(), version=version
        )
        helper = _wheel(
            wheel_root, "diffusers", (f"huggingface-hub=={version}",), b"import huggingface_hub\n"
        )
        project = _wheel(wheel_root, "model_project", ("diffusers==1.0",), b"import diffusers\n")
        source = _material(project, (helper, dependency), base)
        installed = environment.materialize(source, root, base=base, exact_dependencies=True)
        result = subprocess.run(
            [
                str(installed.python),
                "-I",
                "-c",
                "import huggingface_hub,importlib.util; print(huggingface_hub.VALUE); "
                "assert importlib.util.find_spec('ambient_sentinel') is None; "
                "assert importlib.util.find_spec('grpc') is None",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout.strip() == version
        assert (
            "include-system-site-packages = false"
            in (installed.generation / "pyvenv.cfg").read_text()
        )
        held.append((source, installed))
    monkeypatch.setattr(
        environment, "_install_with_uv", lambda *a, **k: pytest.fail("warm install")
    )
    for source, installed in held:
        replay = environment.materialize(source, root, base=base, exact_dependencies=True)
        assert replay.reused and replay.generation == installed.generation


def test_alternate_torch_dependency_is_admitted_in_private_wheel_closure(tmp_path: Path) -> None:
    base = environment.observe_base()
    torch = _wheel(tmp_path, "torch", (), b"VALUE=1\n")
    project = _wheel(tmp_path, "torch_user", ("torch==1.0",), b"import torch\n")
    source = _material(project, (torch,), base)
    installed = environment.materialize(
        source, tmp_path / "env", base=base, exact_dependencies=True
    )
    assert (installed.site_packages / "torch.py").is_file()


def test_private_direct_dependency_fetch_install_and_offline_reuse(tmp_path: Path) -> None:
    import hashlib

    from test_publish_derivation import _SimpleIndex

    dependency = _wheel(tmp_path, "private_dependency", (), b"VALUE=91\n")
    project = _wheel(
        tmp_path,
        "private_client",
        ("private-dependency==1.0",),
        b"from private_dependency import VALUE\n",
    )
    index = _SimpleIndex(tmp_path / "index", [dependency])
    raw = (
        f"private-dependency @ {index.url}private-dependency/{dependency.name} "
        f"--hash=sha256:{hashlib.sha256(dependency.read_bytes()).hexdigest()}\n"
    ).encode()
    lock = environment.read_dependency_requirements(raw)
    root = tmp_path / "environments"
    base = environment.observe_base()
    try:
        fetched = environment.fetch_locked_wheels(lock.rows, root, base)
    finally:
        index.close()
    # The local artifact cache is immutable and needs neither network nor uv on warm open.
    replay = environment.fetch_locked_wheels(lock.rows, root, base)
    assert replay[0].wheel == fetched[0].wheel
    source = _material(project, (fetched[0].path,), base)
    installed = environment.materialize(source, root, base=base, exact_dependencies=True)
    result = subprocess.run(
        [str(installed.python), "-I", "-c", "import private_client; print(private_client.VALUE)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "91"
    assert environment.materialize(source, root, base=base, exact_dependencies=True).reused


@pytest.mark.parametrize(
    "row",
    [
        "proof==1.0 --hash=sha256:" + "a" * 64,
        "proof @ https://user:secret@example/proof-1.0-py3-none-any.whl --hash=sha256:" + "a" * 64,
        "proof @ https://example/proof-1.0-py3-none-any.whl?token=secret --hash=sha256:" + "a" * 64,
        "proof @ https://example/proof-1.0-py3-none-any.whl ; python_version > '3' --hash=sha256:"
        + "a" * 64,
        "proof @ https://example/proof-1.0-py3-none-any.whl --hash=sha256:" + "z" * 64,
        "--index-url https://pypi.org/simple\n"
        "proof @ https://example/proof-1.0-py3-none-any.whl --hash=sha256:" + "a" * 64,
    ],
)
def test_private_dependency_transport_rejects_unselected_or_credential_rows(row: str) -> None:
    with pytest.raises(environment.EnvironmentRefusal):
        environment.read_dependency_requirements(row.encode())


def test_prepared_private_revision_reopens_without_dependency_carriers(tmp_path: Path) -> None:
    import hashlib
    import sys

    from cozy_runtime.internal import canonical
    from cozy_runtime.internal.worker import local_installations, package_prepare
    from cozy_runtime.protocol import documents
    from cozy_runtime.protocol import worker_pb2 as pb
    from test_publish_derivation import _SimpleIndex

    root = tmp_path / "environments"
    wheel_root = root / ".stage" / "prepare" / "wheels"
    wheel_root.mkdir(parents=True)
    dependency = _wheel(tmp_path, "private_dependency", (), b"VALUE=91\n")
    project = _wheel(
        wheel_root,
        "private_client",
        ("private-dependency==1.0",),
        b"from private_dependency import VALUE\n",
    )
    interface = canonical.write(
        {
            "format": "cozy.package.interface/1",
            "application": "private_client:app",
            "entrypoints": [],
            "jobs": [],
        }
    )
    index = _SimpleIndex(tmp_path / "index", [dependency])
    raw = (
        f"private-dependency @ {index.url}private-dependency/{dependency.name} "
        f"--hash=sha256:{hashlib.sha256(dependency.read_bytes()).hexdigest()}\n"
    ).encode()
    row = pb.LocalPackageWheel(
        filename=project.name,
        path=str(project),
        digest=documents.digest_of(project.read_bytes()),
        length=project.stat().st_size,
    )
    revision = pb.LocalPackageRevision(
        package="local/private-client",
        release="1.0",
        source_digest=b"s" * 32,
        package_interface=pb.Ref(digest=documents.digest_of(interface), length=len(interface)),
        files=[pb.LocalPackageFileRef(filename=row.filename, digest=row.digest, length=row.length)],
        dependency_requirements=pb.Ref(digest=documents.digest_of(raw), length=len(raw)),
    )
    _, revision_digest = documents.identity(revision)
    request = pb.PrepareLocalPackageRequest(
        operation_id="prepare",
        package=pb.DevelopmentPackage(
            package=revision.package,
            release=revision.release,
            source_digest=revision.source_digest,
            local_revision_digest=revision_digest,
        ),
        wheels=[row],
        install_root=str(root),
        dependency_requirements=raw,
    )
    base = environment.observe_base()
    held = []
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    def describe(installed: environment.InstalledEnvironment, name: str) -> bytes:
        result = subprocess.run(
            [
                str(installed.python),
                "-I",
                "-c",
                "import private_client; print(private_client.VALUE)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "91"
        return interface

    try:
        result = package_prepare.prepare_local_package(
            request,
            artifact_cache=artifacts,
            install_root=root,
            python=Path(sys.executable),
            describe=describe,
            verified=lambda *_: None,
            base=base,
            materialized=held.append,
        )
    finally:
        index.close()
    local_installations.record(root, request, result, held[0], base)
    # Warm reopen has neither uploaded carriers nor a live dependency source.
    project.unlink()
    warmed = local_installations.reopen(
        root,
        request,
        artifact_cache=artifacts,
        python=Path(sys.executable),
        base=base,
        job_plan_root=tmp_path / "plans",
    )
    assert warmed is not None and warmed[1].generation == held[0].generation
    changed = pb.PrepareLocalPackageRequest()
    changed.CopyFrom(request)
    changed.dependency_requirements += b"# changed exact input\n"
    with pytest.raises(package_prepare.PreparationRefusal, match="local_package_revision_mismatch"):
        local_installations.reopen(
            root,
            changed,
            artifact_cache=artifacts,
            python=Path(sys.executable),
            base=base,
            job_plan_root=tmp_path / "plans",
        )


def test_separate_package_roots_share_uv_payload_inodes_and_safe_cleanup(tmp_path: Path) -> None:
    dependency = _wheel(
        tmp_path,
        "shared_dependency",
        (),
        b"#" + os.urandom(4 << 20).hex().encode() + b"\nVALUE=1\n",
    )
    base = environment.observe_base()
    cache = tmp_path / "machine-dependencies"
    environments = []
    for name in ("first", "second"):
        project = _wheel(tmp_path, name, ("shared-dependency==1.0",), b"import shared_dependency\n")
        source = _material(project, (dependency,), base)
        environments.append(
            environment.materialize(
                source, tmp_path / name, base=base, exact_dependencies=True, dependency_cache=cache
            )
        )
    first, second = (item.site_packages / "shared_dependency.py" for item in environments)
    before = first.stat()
    after = second.stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    assert before.st_nlink >= 3  # Both packages plus uv's own shared cache.
    environment._remove_tree(environments[0].generation)
    remaining = second.stat()
    assert (remaining.st_ino, remaining.st_mode) == (after.st_ino, after.st_mode)
    assert second.read_bytes().startswith(b"#")
