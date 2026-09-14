"""Real file snapshots refuse mutable hooks and survive loss of their source venv."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

from cozy_runtime.internal import _installed_dependencies as dependencies
from cozy_runtime.internal import canonical, package_environment


def _venv(path: Path, *, executable: Path | None = None) -> Path:
    (path / "bin").mkdir(parents=True)
    (path / "pyvenv.cfg").write_text("version_info = 3.12\n")
    if executable is None:
        # Starting the source interpreter would leave observable evidence and fail.
        (path / "bin/python").write_text('#!/bin/sh\ntouch "${0%/*}/executed"\nexit 77\n')
        (path / "bin/python").chmod(0o755)
    else:
        (path / "bin/python").symlink_to(executable)
    site = path / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    return site


def _distribution(
    site: Path,
    name: str,
    members: dict[str, bytes],
    *,
    editable: bool = False,
    requirements: tuple[str, ...] = (),
) -> Path:
    info = site / f"{name}-1.0.dist-info"
    content = {
        f"{info.name}/METADATA": (
            f"Metadata-Version: 2.3\nName: {name}\nVersion: 1.0\n"
            + "".join(f"Requires-Dist: {row}\n" for row in requirements)
        ).encode(),
        **members,
    }
    if editable:
        content[f"{info.name}/direct_url.json"] = json.dumps(
            {"url": "file:///absent-source", "dir_info": {"editable": True}}
        ).encode()
    rows = []
    for relative, body in content.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
        rows.append((relative, "sha256=" + digest, str(len(body))))
    record = info / "RECORD"
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows([*rows, (f"{info.name}/RECORD", "", "")])
    return info


def test_snapshot_excludes_editables_and_hooks_and_survives_source_removal(tmp_path: Path) -> None:
    source, staging = tmp_path / "source", tmp_path / "staging"
    source_site = _venv(source)
    target_site = _venv(staging, executable=Path(sys.executable))
    _distribution(
        source_site,
        "dependency",
        {
            "dependency/data.bin": b"accepted immutable bytes",
            "dangerous.pth": b"import os; raise RuntimeError('mutable hook')\n",
            "dependency-1.0.dist-info/entry_points.txt": (
                b"[console_scripts]\nsnapshot-proof = sys:exit\n"
            ),
            "../../../bin/snapshot-proof": b"#!/absent/source/bin/python\n",
        },
    )
    _distribution(source_site, "captured", {"captured.pth": b"/absent-source\n"}, editable=True)
    (source_site / "unrecorded.py").write_text("raise RuntimeError('not a distribution member')")
    manifest = dependencies.snapshot(source / "bin/python", staging, {"captured"})
    assert not (source / "bin/executed").exists()
    assert not list(target_site.glob("*.pth"))
    assert not (target_site / "captured-1.0.dist-info").exists()
    assert not (target_site / "unrecorded.py").exists()
    assert (target_site / "dependency/data.bin").stat().st_ino != (
        source_site / "dependency/data.bin"
    ).stat().st_ino
    (source_site / "dependency/data.bin").write_bytes(b"source changed")
    dependencies.verify_files(staging, manifest)
    shutil.rmtree(source)
    final = tmp_path / "published"
    staging.rename(final)
    result = subprocess.run(
        [str(final / "bin/snapshot-proof"), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (final / "lib/python3.12/site-packages/dependency/data.bin").read_bytes() == (
        b"accepted immutable bytes"
    )


@pytest.mark.parametrize("failure", ["hash", "editable", "symlink", "collision", "overlap"])
def test_snapshot_refuses_unverifiable_dependency_files(tmp_path: Path, failure: str) -> None:
    source, staging = tmp_path / "source", tmp_path / "staging"
    source_site = _venv(source)
    target_site = _venv(staging)
    _distribution(
        source_site, "dependency", {"dependency.py": b"value=1\n"}, editable=failure == "editable"
    )
    if failure == "hash":
        (source_site / "dependency.py").write_bytes(b"value=2\n")
    elif failure == "symlink":
        (source_site / "dependency.py").unlink()
        (source_site / "dependency.py").symlink_to(__file__)
    elif failure == "collision":
        (target_site / "dependency.py").write_bytes(b"different distribution\n")
    elif failure == "overlap":
        _distribution(source_site, "other", {"dependency.py": b"value=1\n"})
    with pytest.raises(package_environment.EnvironmentRefusal) as caught:
        dependencies.snapshot(source / "bin/python", staging, set())
    assert (
        caught.value.code
        == {
            "hash": "package_environment_dependency_member_changed",
            "editable": "package_environment_dependency_editable",
            "symlink": "package_environment_dependency_member_unsupported",
            "collision": "package_environment_dependency_member_collision",
            "overlap": "package_environment_dependency_member_collision",
        }[failure]
    )
    assert not (source / "bin/executed").exists()


@pytest.mark.parametrize("changed", ["dependency.py", "../../../bin/dependency-command"])
def test_manifest_binds_bytes_and_detects_wheel_overwrite(tmp_path: Path, changed: str) -> None:
    source, staging = tmp_path / "source", tmp_path / "staging"
    source_site = _venv(source)
    target_site = _venv(staging)
    _distribution(
        source_site,
        "dependency",
        {
            "dependency.py": b"value=1\n",
            "dependency-1.0.dist-info/entry_points.txt": (
                b"[console_scripts]\ndependency-command = sys:exit\n"
            ),
        },
    )
    manifest = dependencies.snapshot(source / "bin/python", staging, set())
    (staging / dependencies.MANIFEST_NAME).write_bytes(manifest)
    ref = {"digest": "sha256:" + hashlib.sha256(manifest).hexdigest(), "length": len(manifest)}
    assert dependencies.read_manifest(staging, ref) == canonical.parse_canonical(manifest)
    (target_site / changed).write_bytes(b"value=2\n")
    with pytest.raises(package_environment.EnvironmentRefusal, match="member_collision"):
        dependencies.verify_files(staging, manifest)
    (staging / dependencies.MANIFEST_NAME).write_bytes(manifest.replace(b"1.0", b"2.0"))
    with pytest.raises(package_environment.EnvironmentRefusal, match="snapshot_invalid"):
        dependencies.read_manifest(staging, ref)


def test_failed_final_observation_publishes_no_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source_site = _venv(source)
    _distribution(source_site, "dependency", {"dependency.py": b"value=1\n"})
    wheel_root = tmp_path / "wheel"
    wheel_root.mkdir()
    _distribution(
        wheel_root,
        "captured",
        {
            "captured.py": b"value=2\n",
            "captured-1.0.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: snapshot-test\nRoot-Is-Purelib: true\n"
                b"Tag: py3-none-any\n"
            ),
        },
    )
    path = tmp_path / "captured-1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member in sorted(wheel_root.rglob("*")):
            if member.is_file():
                archive.write(member, str(member.relative_to(wheel_root)))
    base = package_environment.observe_base(Path(sys.executable))
    wheel, verified = package_environment.measure_wheel(path, "project", base)
    material = package_environment.Input(
        "sha256:" + "a" * 64,
        wheel.digest,
        (wheel,),
        (package_environment.BlobGrant(wheel.digest, wheel.length, path, verified),),
    )
    install_root = tmp_path / "environments"
    # This thin source deliberately lacks Runtime. The actual final observation must fail
    # before a receipt could admit it, even though its recorded dependency copied correctly.
    with pytest.raises(package_environment.EnvironmentRefusal, match="base_invalid"):
        package_environment.materialize(
            material, install_root, base=base, dependency_python=source / "bin/python"
        )
    assert not list((install_root / "receipts").iterdir())
    assert not list((install_root / "contents").iterdir())
    assert not list(install_root.glob(".environment-staging-*"))
    assert not (source / "bin/executed").exists()


def test_required_closure_tracks_extras_markers_cycles_and_captured_wheels(tmp_path: Path) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    site = _venv(source)
    target_site = _venv(target)
    _distribution(site, "cozy-runtime", {"cozy_runtime/__init__.py": b""})
    _distribution(
        site,
        "model",
        {"model.pth": b"mutable"},
        editable=True,
        requirements=("unrelated-sdk-package",),
    )
    _distribution(
        site,
        "codec",
        {"codec.py": b"codec = 1"},
        requirements=(
            'pixels>=1; extra == "images"',
            'sound>=1; extra == "audio"',
            "cycle",
            'windows-only; sys_platform == "win32"',
        ),
    )
    _distribution(site, "cycle", {"cycle.py": b""}, requirements=("codec[audio]",))
    _distribution(site, "pixels", {"pixels.py": b""})
    _distribution(site, "sound", {"sound.py": b""})
    unrelated = _distribution(site, "unrelated-sdk-package", {"huge.bin": b"unused"})
    # An unrelated package's bytes/RECORD must never be read, copied, or verified.
    (unrelated / "RECORD").unlink()
    provided = {"model": ("2.0", (Requirement("codec[images]>=1"),))}
    included = dependencies.required_distributions(
        source / "bin/python",
        provided,
        {key: str(value) for key, value in default_environment().items()},
    )
    assert included == {"cozy-runtime", "codec", "cycle", "pixels", "sound"}
    manifest = dependencies.snapshot(
        source / "bin/python", target, set(provided), included=included
    )
    dependencies.verify_files(target, manifest)
    assert (target_site / "sound.py").exists() and (target_site / "pixels.py").exists()
    assert not (target_site / "huge.bin").exists() and not (target_site / "model.pth").exists()
    assert not (source / "bin/executed").exists()


@pytest.mark.parametrize(
    "requirement,code",
    [
        ("missing", "dependency_missing"),
        ("codec>=2", "dependency_base_conflict"),
        ("codec[images]", "dependency_missing"),
    ],
)
def test_required_closure_refuses_incomplete_or_incompatible_graph(
    tmp_path: Path, requirement: str, code: str
) -> None:
    site = _venv(tmp_path / "source")
    _distribution(site, "cozy-runtime", {"cozy_runtime/__init__.py": b""})
    _distribution(site, "codec", {"codec.py": b""}, requirements=('pixels; extra == "images"',))
    with pytest.raises(package_environment.EnvironmentRefusal, match=code):
        dependencies.required_distributions(
            tmp_path / "source/bin/python",
            {"project": ("1.0", (Requirement(requirement),))},
            {key: str(value) for key, value in default_environment().items()},
        )


def test_dependency_copy_accepts_concurrent_uv_hardlink_churn(tmp_path: Path) -> None:
    source, target, sibling = tmp_path / "source", tmp_path / "copied", tmp_path / "cache-link"
    body = b"recorded dependency bytes\n" * (1 << 20)
    source.write_bytes(body)
    expected = (
        "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
    )
    before = source.stat()
    started, stop = threading.Event(), threading.Event()
    links = 0

    def churn() -> None:
        nonlocal links
        while not stop.is_set():
            os.link(source, sibling)
            sibling.unlink()
            links += 1
            started.set()

    thread = threading.Thread(target=churn)
    thread.start()
    try:
        assert started.wait(timeout=5)
        dependencies._copy_dependency_file(source, target, expected, str(len(body)))
    finally:
        stop.set()
        thread.join(timeout=5)
    after = source.stat()
    assert not thread.is_alive()
    assert links > 1
    assert after.st_ctime_ns != before.st_ctime_ns
    assert after.st_mtime_ns == before.st_mtime_ns
    assert source.read_bytes() == target.read_bytes() == body


def test_dependency_copy_refuses_changed_bytes_even_with_restored_mtime(tmp_path: Path) -> None:
    source, target = tmp_path / "source", tmp_path / "copied"
    original = b"accepted bytes"
    source.write_bytes(original)
    before = source.stat()
    expected = (
        "sha256="
        + base64.urlsafe_b64encode(hashlib.sha256(original).digest()).rstrip(b"=").decode()
    )
    source.write_bytes(b"modified bytes")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_size == before.st_size
    assert source.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(package_environment.EnvironmentRefusal, match="member_changed"):
        dependencies._copy_dependency_file(source, target, expected, str(len(original)))


@pytest.mark.parametrize("algorithm", ["sha256", "sha384", "sha512", ""])
def test_dependency_record_verification_also_returns_canonical_content_digest(
    tmp_path: Path, algorithm: str
) -> None:
    source, target = tmp_path / "source", tmp_path / "copied"
    body = b"installed bytes with an independently recorded hash\n" * 1024
    source.write_bytes(body)
    expected = ""
    if algorithm:
        encoded = base64.urlsafe_b64encode(hashlib.new(algorithm, body).digest()).rstrip(b"=")
        expected = algorithm + "=" + encoded.decode()
    actual = dependencies._copy_dependency_file(source, target, expected, str(len(body)))
    assert actual == ("sha256:" + hashlib.sha256(body).hexdigest(), len(body))
    assert source.stat().st_ino != target.stat().st_ino


def test_cached_payloads_are_immutable_and_independent_of_source_and_cache(tmp_path: Path) -> None:
    source, first, second, cache = (
        tmp_path / part for part in ("source", "first", "second", "cache")
    )
    site = _venv(source)
    first_site, second_site = _venv(first), _venv(second)
    _distribution(
        site,
        "dependency",
        {
            "dependency.py": b"value = 47\n",
            "dependency/data.bin": bytes(range(256)) * 4096,
            "dependency-1.0.dist-info/entry_points.txt": b"[console_scripts]\nproof = sys:exit\n",
        },
    )
    initial = dependencies.snapshot(source / "bin/python", first, set(), cache=cache)
    dependencies.verify_files(first, initial)
    repeated = dependencies.snapshot(source / "bin/python", second, set(), cache=cache)
    assert initial == repeated
    dependencies.verify_files(second, repeated)
    payload = "dependency/data.bin"
    assert (first_site / payload).stat().st_ino == (second_site / payload).stat().st_ino
    assert (first_site / payload).stat().st_ino != (site / payload).stat().st_ino
    assert not (first_site / payload).stat().st_mode & 0o222
    assert (first_site / "dependency-1.0.dist-info/RECORD").stat().st_ino != (
        second_site / "dependency-1.0.dist-info/RECORD"
    ).stat().st_ino
    (site / payload).write_bytes(b"mutated source bytes")
    shutil.rmtree(source)
    shutil.rmtree(cache)
    package_environment._make_read_only(first)
    package_environment._remove_tree(first)
    assert not (second_site / payload).stat().st_mode & 0o222
    dependencies.verify_files(second, repeated)


def test_cache_reuse_still_rejects_changed_source_with_unchanged_record(tmp_path: Path) -> None:
    source, first, second, cache = (
        tmp_path / part for part in ("source", "first", "second", "cache")
    )
    site = _venv(source)
    _venv(first)
    _venv(second)
    _distribution(site, "dependency", {"dependency.py": b"value = 47\n"})
    initial = dependencies.snapshot(source / "bin/python", first, set(), cache=cache)
    before = (site / "dependency.py").stat()
    (site / "dependency.py").write_bytes(b"value = 48\n")
    os.utime(site / "dependency.py", ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(package_environment.EnvironmentRefusal, match="member_changed"):
        dependencies.snapshot(source / "bin/python", second, set(), cache=cache)
    dependencies.verify_files(first, initial)


def test_concurrent_captures_publish_only_complete_immutable_objects(tmp_path: Path) -> None:
    source, cache = tmp_path / "source", tmp_path / "cache"
    site = _venv(source)
    _distribution(site, "dependency", {"payload.bin": bytes(range(256)) * (1 << 14)})
    barrier = threading.Barrier(2)

    def capture(name: str) -> tuple[Path, bytes]:
        target = tmp_path / name
        _venv(target)
        barrier.wait()
        manifest = dependencies.snapshot(source / "bin/python", target, set(), cache=cache)
        dependencies.verify_files(target, manifest)
        return target, manifest

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(capture, ["first", "second"]))
    assert first[1] == second[1]
    payload = Path("lib/python3.12/site-packages/payload.bin")
    assert (first[0] / payload).samefile(second[0] / payload)
    assert not list(cache.glob(".copy-*"))
    assert all(not path.stat().st_mode & 0o222 for path in cache.iterdir())


def test_cached_capture_refuses_mutable_cache_and_staged_collisions(tmp_path: Path) -> None:
    source, first, second, cache = (
        tmp_path / part for part in ("source", "first", "second", "cache")
    )
    site = _venv(source)
    _venv(first)
    target_site = _venv(second)
    _distribution(site, "dependency", {"payload.bin": b"retained immutable bytes"})
    original = dependencies.snapshot(source / "bin/python", first, set(), cache=cache)
    (target_site / "payload.bin").write_bytes(b"installed package owns this path")
    with pytest.raises(package_environment.EnvironmentRefusal, match="member_collision"):
        dependencies.snapshot(source / "bin/python", second, set(), cache=cache)
    dependencies.verify_files(first, original)
    # A writable cache inode cannot become a shared captured dependency.
    target = tmp_path / "third"
    _venv(target)
    cached = next(cache.iterdir())
    cached.chmod(0o644)
    with pytest.raises(package_environment.EnvironmentRefusal, match="cache_invalid"):
        dependencies.snapshot(source / "bin/python", target, set(), cache=cache)


def test_cross_filesystem_cache_uses_independent_copies(tmp_path: Path) -> None:
    memory = Path("/dev/shm")
    if not memory.is_dir() or memory.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("requires two actual filesystems for the hardlink fallback")
    source, target = tmp_path / "source", tmp_path / "target"
    site, destination = _venv(source), _venv(target)
    _distribution(site, "dependency", {"payload.bin": b"independent copied bytes"})
    with tempfile.TemporaryDirectory(prefix="cozy-dependency-cache-", dir=memory) as root:
        cache = Path(root)
        manifest = dependencies.snapshot(source / "bin/python", target, set(), cache=cache)
        dependencies.verify_files(target, manifest)
        assert not list(cache.iterdir())
    assert not (site / "payload.bin").samefile(destination / "payload.bin")
