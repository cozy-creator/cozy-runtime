"""Cold/warm live proof and essential red arms for cr-027's offline mechanism."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import platform
import shutil
import stat
import sys
import tempfile
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path

from cozy_runtime.internal import canonical
from cozy_runtime.internal.environment import (
    BlobGrant,
    EnvironmentInput,
    EnvironmentRefusal,
    WheelInventoryEntry,
    materialize,
    verify_generation,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def record_hash(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return "sha256=" + encoded


def build_wheel(
    root: Path,
    distribution: str,
    version: str,
    package: str,
    package_source: bytes | None = None,
) -> Path:
    stem = distribution.replace("-", "_")
    filename = f"{stem}-{version}-py3-none-any.whl"
    dist_info = f"{stem}-{version}.dist-info"
    files = {
        f"{package}/__init__.py": (
            package_source if package_source is not None else f"VALUE = {version!r}\n".encode()
        ),
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.3\nName: {distribution}\nVersion: {version}\n\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: cr-027-live\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n\n"
        ),
    }
    rows = [[name, record_hash(data), str(len(data))] for name, data in sorted(files.items())]
    rows.append([f"{dist_info}/RECORD", "", ""])
    rendered = io.StringIO(newline="")
    csv.writer(rendered, lineterminator="\n").writerows(rows)
    files[f"{dist_info}/RECORD"] = rendered.getvalue().encode()
    path = root / filename
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as wheel:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            wheel.writestr(info, data)
    return path


def inventory(path: Path, distribution: str, version: str, owner: str) -> WheelInventoryEntry:
    data = path.read_bytes()
    return WheelInventoryEntry(
        distribution=distribution,
        version=version,
        filename=path.name,
        digest=digest(data),
        length=len(data),
        tags=("py3-none-any",),
        owner=owner,
    )


def grant(path: Path, entry: WheelInventoryEntry) -> BlobGrant:
    return BlobGrant(entry.digest, entry.length, path)


def duplicate_member_wheel(source: Path, root: Path) -> Path:
    root.mkdir()
    path = root / source.name
    shutil.copyfile(source, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as wheel:
            wheel.writestr("exact_dependency/__init__.py", b"VALUE = 'duplicate'\n")
    return path


def expanding_wheel(source: Path, root: Path) -> Path:
    root.mkdir()
    path = root / source.name
    shutil.copyfile(source, path)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as wheel:
        wheel.writestr("expansion-payload", b"\0" * (2 << 20))
    return path


def refusal(code: str, call: object) -> None:
    try:
        assert callable(call)
        call()
    except EnvironmentRefusal as exc:
        assert exc.code == code, f"wanted {code}, got {exc.code}: {exc.detail}"
    else:
        raise AssertionError(f"{code} arm did not fire")


def writable_cleanup(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    shutil.rmtree(root)


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="cr027-live-"))
    try:
        blobs = root / "cas"
        blobs.mkdir()
        dependency_path = build_wheel(blobs, "exact-dependency", "1.2.3", "exact_dependency")
        project_path = build_wheel(blobs, "exact-project", "4.5.6", "exact_project")
        extra_path = build_wheel(blobs, "extra-wheel", "7.8.9", "extra_wheel")
        torch_path = build_wheel(
            blobs,
            "torch",
            "2.9.1",
            "torch",
            b"class version:\n    cuda = '12.9'\n"
            b"class cuda:\n    @staticmethod\n    def is_available(): return False\n",
        )
        dependency = inventory(dependency_path, "exact-dependency", "1.2.3", "wheelhouse")
        project = inventory(project_path, "exact-project", "4.5.6", "project")
        extra = inventory(extra_path, "extra-wheel", "7.8.9", "bundle")
        torch = inventory(torch_path, "torch", "2.9.1", "wheelhouse")

        wheelhouse = canonical.write(
            {
                "format": "test.cr027.WheelhouseManifest/1",
                "wheels": [dependency.digest],
            }
        )
        bundle = canonical.write(
            {"format": "test.cr027.EndpointBundle/1", "wheels": [project.digest]}
        )
        libc_name, libc_version = platform.libc_ver()
        target = pb.PlatformTarget(
            os_arch="linux/amd64",
            libc=f"{libc_name}{libc_version}",
            python_abi=f"cp{sys.version_info.major}{sys.version_info.minor}",
            accelerator_backend="none",
            accelerator_abi="none",
        )
        spec_bytes = documents.canonical_bytes(
            pb.EndpointEnvironmentSpec(
                platform_target=target,
                endpoint_bundle_digest=documents.digest_of(bundle),
                project_wheel_digest=documents.raw(project.digest),
                wheelhouse_manifest_digest=documents.digest_of(wheelhouse),
            )
        )
        source = EnvironmentInput(
            environment_spec_bytes=spec_bytes,
            environment_spec_digest=digest(spec_bytes),
            endpoint_bundle_bytes=bundle,
            wheelhouse_manifest_bytes=wheelhouse,
            wheels=(dependency, project),
            grants=(grant(dependency_path, dependency), grant(project_path, project)),
        )
        uv = Path(shutil.which("uv") or "")
        assert uv.is_absolute(), "uv is not installed"
        materialized_root = root / "installed"
        cold = materialize(source, materialized_root, uv=uv)
        rebuilt = materialize(source, root / "rebuilt", uv=uv)
        raced = materialize(source, materialized_root, uv=uv)
        warm = materialize(
            source,
            materialized_root,
            uv=uv,
            expected_receipt_digest=cold.receipt_digest,
        )
        assert cold.receipt_bytes == rebuilt.receipt_bytes == raced.receipt_bytes, (
            cold.receipt_bytes.decode(),
            rebuilt.receipt_bytes.decode(),
        )
        assert raced.receipt_bytes == warm.receipt_bytes
        assert not rebuilt.reused and raced.reused and warm.reused
        result = __import__("subprocess").run(
            [
                str(warm.python),
                "-I",
                "-c",
                "import exact_dependency,exact_project; "
                "print(exact_dependency.VALUE, exact_project.VALUE)",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout.strip() == "1.2.3 4.5.6"

        cuda_target = pb.PlatformTarget(
            os_arch=target.os_arch,
            libc=target.libc,
            python_abi=target.python_abi,
            accelerator_backend="cuda",
            accelerator_abi="cu129",
        )
        cuda_spec_bytes = documents.canonical_bytes(
            pb.EndpointEnvironmentSpec(
                platform_target=cuda_target,
                endpoint_bundle_digest=documents.digest_of(bundle),
                project_wheel_digest=documents.raw(project.digest),
                wheelhouse_manifest_digest=documents.digest_of(wheelhouse),
            )
        )
        cuda_source = replace(
            source,
            environment_spec_bytes=cuda_spec_bytes,
            environment_spec_digest=digest(cuda_spec_bytes),
            wheels=(dependency, project, torch),
            grants=(*source.grants, grant(torch_path, torch)),
        )
        cuda_environment = materialize(cuda_source, root / "cuda-installed", uv=uv)
        unavailable = __import__("subprocess").run(
            [
                str(cuda_environment.python),
                "-I",
                "-c",
                "import torch; print(torch.version.cuda, torch.cuda.is_available())",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        assert unavailable.stdout.strip() == "12.9 False"

        refusal(
            "wheel_blob_missing",
            lambda: materialize(
                replace(source, grants=(source.grants[1],)), root / "missing", uv=uv
            ),
        )
        refusal(
            "wheel_blob_extra",
            lambda: materialize(
                replace(source, grants=(*source.grants, grant(extra_path, extra))),
                root / "extra",
                uv=uv,
            ),
        )
        corrupt_path = blobs / "corrupt-object"
        corrupt = bytearray(dependency_path.read_bytes())
        corrupt[len(corrupt) // 2] ^= 1
        corrupt_path.write_bytes(corrupt)
        refusal(
            "wheel_digest_mismatch",
            lambda: materialize(
                replace(
                    source,
                    grants=(
                        BlobGrant(dependency.digest, dependency.length, corrupt_path),
                        source.grants[1],
                    ),
                ),
                root / "corrupt",
                uv=uv,
            ),
        )
        refusal(
            "wheel_filename_mismatch",
            lambda: materialize(
                replace(
                    source,
                    wheels=(
                        replace(dependency, filename="renamed-1-py3-none-any.whl"),
                        project,
                    ),
                ),
                root / "renamed",
                uv=uv,
            ),
        )
        duplicate_path = duplicate_member_wheel(dependency_path, root / "duplicate-cas")
        duplicate = inventory(duplicate_path, "exact-dependency", "1.2.3", "wheelhouse")
        refusal(
            "wheel_member_duplicate",
            lambda: materialize(
                replace(
                    source,
                    wheels=(duplicate, project),
                    grants=(grant(duplicate_path, duplicate), source.grants[1]),
                ),
                root / "duplicate",
                uv=uv,
            ),
        )
        expanding_path = expanding_wheel(dependency_path, root / "expanding-cas")
        expanding = inventory(expanding_path, "exact-dependency", "1.2.3", "wheelhouse")
        refusal(
            "wheel_expansion_cap",
            lambda: materialize(
                replace(
                    source,
                    wheels=(expanding, project),
                    grants=(grant(expanding_path, expanding), source.grants[1]),
                ),
                root / "expanding",
                uv=uv,
            ),
        )
        site = next((warm.generation / "venv" / "lib").glob("python*/site-packages"))
        site_mode = site.stat().st_mode
        site.chmod(site_mode | stat.S_IWUSR)
        unrecorded = site / "planted_unrecorded.py"
        unrecorded.write_text("VALUE = 'unrecorded'\n")
        unrecorded.chmod(unrecorded.stat().st_mode & ~0o222)
        site.chmod(site_mode)
        refusal(
            "installed_unrecorded_file",
            lambda: verify_generation(
                warm.generation,
                receipt_digest=warm.receipt_digest,
                environment_spec_digest=source.environment_spec_digest,
            ),
        )
        site.chmod(site_mode | stat.S_IWUSR)
        unrecorded.unlink()
        site.chmod(site_mode)
        installed_file = warm.generation / "venv" / "lib"
        installed_file = next(
            installed_file.glob("python*/site-packages/exact_project/__init__.py")
        )
        installed_file.chmod(installed_file.stat().st_mode | stat.S_IWUSR)
        installed_file.write_text("VALUE = 'tampered'\n")
        installed_file.chmod(installed_file.stat().st_mode & ~0o222)
        refusal(
            "installed_record_mismatch",
            lambda: verify_generation(
                warm.generation,
                receipt_digest=warm.receipt_digest,
                environment_spec_digest=source.environment_spec_digest,
            ),
        )
        print(
            "cr-027 live: two cold installs + publish-race/warm reuse + isolated imports; "
            "CUDA cu129 accepted without a physical device; "
            "missing/extra/corrupt/renamed/duplicate/expanded/unrecorded/tampered "
            "bytes refused; "
            f"receipt={cold.receipt_digest}"
        )
    finally:
        writable_cleanup(root)


if __name__ == "__main__":
    main()
