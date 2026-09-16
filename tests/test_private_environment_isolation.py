from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import shutil
import stat
import subprocess
import sys
import zipfile
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
    held: list[environment.InstalledEnvironment] = []
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
        # Each operation receives a different grant path for identical bytes. uv
        # otherwise keys archive extraction by the input path, duplicating payloads.
        grant = tmp_path / (name + "-grant") / dependency.name
        grant.parent.mkdir()
        shutil.copyfile(dependency, grant)
        grant.chmod(0o444)
        source = _material(project, (grant,), base)
        environments.append(
            environment.materialize(
                source, tmp_path / name, base=base, exact_dependencies=True, dependency_cache=cache
            )
        )
    digest = hashlib.sha256(dependency.read_bytes()).hexdigest()
    archive = cache / ".stage" / digest / dependency.name
    original_grant = tmp_path / "first-grant" / dependency.name
    assert archive.stat().st_ino == original_grant.stat().st_ino
    assert stat.S_IMODE(original_grant.stat().st_mode) == 0o444
    first, second = (item.site_packages / "shared_dependency.py" for item in environments)
    before = first.stat()
    after = second.stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    assert before.st_nlink >= 3  # Both packages plus uv's own shared cache.
    environment._remove_tree(environments[0].generation)
    remaining = second.stat()
    assert (remaining.st_ino, remaining.st_mode) == (after.st_ino, after.st_mode)
    assert second.read_bytes().startswith(b"#")


@pytest.mark.parametrize("installer", ["locked", "wheel", "fetch"])
@pytest.mark.parametrize("snapshot_first", [False, True])
def test_installers_and_snapshots_share_private_cache_root(
    tmp_path: Path, installer: str, snapshot_first: bool
) -> None:
    from cozy_runtime.internal import _installed_dependencies as dependencies
    from test_installed_dependencies import _distribution, _venv
    from test_publish_derivation import _SimpleIndex

    cache = tmp_path / "local-packages" / "dependency-objects"
    source = tmp_path / "source"
    _distribution(_venv(source), "captured", {"captured.py": b"VALUE=73\n"})
    snapshots: list[tuple[Path, bytes]] = []

    def capture(target: Path) -> None:
        manifest = dependencies.snapshot(source / "bin/python", target, set(), cache=cache)
        snapshots.append((target, manifest))

    if snapshot_first:
        first = tmp_path / "first-snapshot"
        _venv(first)
        capture(first)
    wheel = _wheel(tmp_path, "cache_client", (), b"VALUE=91\n")
    index = _SimpleIndex(tmp_path / "index", [wheel])
    uv_path = shutil.which("uv")
    assert uv_path is not None
    uv, python = Path(uv_path), Path(sys.executable)
    target = tmp_path / "installed"
    raw = (
        f"cache-client @ {index.url}cache-client/{wheel.name} "
        f"--hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}\n"
    ).encode()
    rows = environment.read_locked_requirements(raw).rows
    try:
        if installer == "locked":
            environment._install_locked(
                uv,
                python,
                target,
                tmp_path / "requirements",
                (index.url,),
                rows,
                dependencies=capture,
                cache_root=cache,
            )
        else:
            if installer == "fetch":
                inspected = environment.fetch_locked_wheels(
                    rows, tmp_path / "fetch", environment.observe_base(), cache_root=cache
                )
            else:
                item, _ = environment._measure_inspected_wheel(
                    wheel, "custom", environment.observe_base()
                )
                inspected = [item]
            environment._install_with_uv(
                uv, python, target, inspected, dependencies=capture, cache_root=cache
            )
    finally:
        index.close()
    later = tmp_path / "later-snapshot"
    _venv(later)
    capture(later)
    for root in (cache, cache / "uv-cache"):
        info = root.lstat()
        assert stat.S_ISDIR(info.st_mode)
        assert stat.S_IMODE(info.st_mode) == 0o700
        assert info.st_uid == os.geteuid()
        assert root.resolve() == root
    payloads = []
    for path, manifest in snapshots:
        dependencies.verify_files(path, manifest)
        payloads.append(next(path.glob("lib/python*/site-packages/captured.py")))
    assert all(path.samefile(payloads[0]) for path in payloads)
    result = subprocess.run(
        [
            str(target / "bin/python"),
            "-I",
            "-c",
            "import cache_client, captured; "
            "assert (cache_client.VALUE, captured.VALUE) == (91, 73)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "include-system-site-packages = false" in (target / "pyvenv.cfg").read_text()


@pytest.mark.parametrize("kind", ["permissions", "symlink", "file", "relative"])
def test_dependency_cache_rejects_existing_unsafe_roots_without_repair(
    tmp_path: Path, kind: str
) -> None:
    cache = tmp_path / "cache"
    if kind == "permissions":
        cache.mkdir(mode=0o755)
    elif kind == "symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        cache.symlink_to(target, target_is_directory=True)
    elif kind == "file":
        cache.write_text("user-owned")
    else:
        cache = Path("relative-dependency-cache")
    before = cache.lstat() if kind != "relative" else None
    with pytest.raises(environment.EnvironmentRefusal, match="dependency_cache_invalid"):
        environment._install_with_uv(
            Path("/must-not-run"), Path(sys.executable), tmp_path / "staging", [], cache_root=cache
        )
    if before is not None:
        after = cache.lstat()
        assert (before.st_ino, before.st_mode, before.st_uid) == (
            after.st_ino,
            after.st_mode,
            after.st_uid,
        )
        assert not (cache / "uv-cache").exists()


def test_verified_wheel_hook_runs_in_private_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    poison = tmp_path / "ambient"
    poison.mkdir()
    (poison / "declared_hook.py").write_text("raise RuntimeError('ambient hook')\n")
    monkeypatch.setenv("PYTHONPATH", str(poison))
    hook = _wheel(
        tmp_path,
        "declared_hook",
        (),
        b"import builtins; builtins.cozy_declared_hook_origin = __file__\n",
        resources={"declared_hook.pth": b"import declared_hook\n"},
    )
    project = _wheel(tmp_path, "hook_client", ("declared-hook==1.0",), b"VALUE=47\n")
    base = environment.observe_base()
    source = _material(project, (hook,), base)
    assert not hasattr(builtins, "cozy_declared_hook_origin")
    installed = environment.materialize(
        source, tmp_path / "private", base=base, exact_dependencies=True
    )
    result = subprocess.run(
        [
            str(installed.python),
            "-I",
            "-c",
            "import builtins,sys; from pathlib import Path; import hook_client; "
            "assert Path(builtins.cozy_declared_hook_origin).is_relative_to(sys.prefix); "
            "assert hook_client.VALUE == 47",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not hasattr(builtins, "cozy_declared_hook_origin")
    assert (installed.site_packages / "declared_hook.pth").read_bytes() == b"import declared_hook\n"


@pytest.mark.parametrize("damage", ["changed", "unrecorded"])
def test_wheel_hooks_require_recorded_verified_bytes(tmp_path: Path, damage: str) -> None:
    wheel = _wheel(
        tmp_path,
        "declared_hook",
        (),
        b"VALUE=1\n",
        resources={"declared_hook.pth": b"import declared_hook\n"},
    )
    with zipfile.ZipFile(wheel) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if damage == "changed":
        members["declared_hook.pth"] = b"import altered_hook_\n"
    else:
        record = "declared_hook-1.0.dist-info/RECORD"
        members[record] = b"".join(
            line
            for line in members[record].splitlines(keepends=True)
            if not line.startswith(b"declared_hook.pth,")
        )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    wheel.write_bytes(output.getvalue())
    with pytest.raises(environment.EnvironmentRefusal, match="wheel_record_"):
        environment.measure_wheel(wheel, "custom", environment.observe_base())


@pytest.mark.parametrize("purelib", ["true", "false"])
@pytest.mark.parametrize("scheme", ["root", "platlib"])
def test_platform_shared_library_wheels_need_no_cpython_abi(
    tmp_path: Path, purelib: str, scheme: str
) -> None:
    tag = "py3-none-manylinux_2_17_x86_64"
    prefix = "" if scheme == "root" else "ffi_payload-1.0.data/platlib/"
    wheel = _wheel(
        tmp_path,
        "ffi_payload",
        (),
        b"VALUE=1\n",
        resources={
            prefix + "ffi_payload/libffi.so": b"\x7fELFverified platform library",
            "ffi_payload-1.0.data/data/share/doc/ffi_payload/LICENSE": b"license\n",
            "ffi_payload-1.0.dist-info/WHEEL": (
                f"Wheel-Version: 1.0\nRoot-Is-Purelib: {purelib}\nTag: {tag}\n"
            ).encode(),
        },
    )
    wheel = wheel.rename(wheel.with_name(f"ffi_payload-1.0-{tag}.whl"))
    item, _ = environment._measure_inspected_wheel(wheel, "custom", environment.observe_base())
    assert item.native_members == ("ffi_payload/libffi.so",)


@pytest.mark.parametrize("platform", ["any", "manylinux_2_17_aarch64"])
def test_shared_libraries_still_require_a_matching_concrete_platform(
    tmp_path: Path, platform: str
) -> None:
    tag = f"py3-none-{platform}"
    wheel = _wheel(
        tmp_path,
        "ffi_payload",
        (),
        b"VALUE=1\n",
        resources={
            "ffi_payload/libffi.so": b"\x7fELFverified platform library",
            "ffi_payload-1.0.dist-info/WHEEL": (
                f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n"
            ).encode(),
        },
    )
    wheel = wheel.rename(wheel.with_name(f"ffi_payload-1.0-{tag}.whl"))
    base = environment.observe_base()
    with pytest.raises(
        environment.EnvironmentRefusal, match=r"(tag_not_concrete|platform_incompatible)"
    ):
        item, _ = environment._measure_inspected_wheel(wheel, "custom", base)
        environment._require_compatible_tags(item.wheel, base)


def test_seeded_archive_reuses_exact_bytes_without_download_broker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    from cozy_runtime.internal import storage_admission

    wheel = _wheel(tmp_path, "seeded_dependency", (), b"VALUE=92\n")
    base = environment.observe_base()
    item, _ = environment._measure_inspected_wheel(wheel, "custom", base)
    cache = tmp_path / "machine-dependencies"
    environment._require_dependency_cache_root(cache)
    archive = environment._cache_wheel_archive(cache, item)
    assert archive.stat().st_ino != wheel.stat().st_ino  # Mutable source was copied.
    raw = (
        f"seeded-dependency @ https://invalid.example/{wheel.name} --hash={item.wheel.digest}\n"
    ).encode()
    rows = environment.read_locked_requirements(raw).rows
    # A real inherited channel makes this take the rental branch. No downloader
    # is serving it: any accidental fetch fails instead of escaping to the web.
    host, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with host, worker:
        worker.settimeout(0.05)
        host.settimeout(0.05)
        monkeypatch.setattr(storage_admission, "_channel", worker)
        fetched = environment.fetch_locked_wheels(rows, tmp_path / "other", base, cache_root=cache)
        assert fetched[0].path == archive
        with pytest.raises(TimeoutError):
            host.recv(1)
    assert fetched[0].wheel.digest == item.wheel.digest


@pytest.mark.parametrize("damage", ["contents", "symlink", "writable"])
def test_shared_archive_damage_refuses_without_replacing_it(tmp_path: Path, damage: str) -> None:
    wheel = _wheel(tmp_path, "seeded_dependency", (), b"VALUE=92\n")
    item, _ = environment._measure_inspected_wheel(wheel, "custom", environment.observe_base())
    cache = tmp_path / "machine-dependencies"
    environment._require_dependency_cache_root(cache)
    archive = environment._cache_wheel_archive(cache, item)
    if damage == "contents":
        archive.chmod(0o644)
        archive.write_bytes(b"changed")
        archive.chmod(0o444)
    elif damage == "symlink":
        archive.unlink()
        archive.symlink_to(wheel)
    else:
        archive.chmod(0o644)
    before = archive.lstat()
    with pytest.raises(environment.EnvironmentRefusal, match="wheel_invalid"):
        environment._cache_wheel_archive(cache, item)
    assert archive.lstat() == before
