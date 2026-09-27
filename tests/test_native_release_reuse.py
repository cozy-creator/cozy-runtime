"""Exercise the actual reuse command with a reviewed donor, never compiling CUDA."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import re
import subprocess
import tomllib
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "cozy_runtime/_kernels/_C.abi3.so"
TAG = "cp312-abi3-manylinux_2_28_x86_64"


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


@pytest.fixture(scope="session")
def donor(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The public donor publish.yaml pins, fetched once and checked against that pin."""
    workflow = (ROOT / ".github/workflows/publish.yaml").read_text()
    filename = re.search(r'filename = "(cozy_runtime-[^"]+\.whl)"', workflow)
    pinned = re.search(r"--native-sha256 ([0-9a-f]{64})", workflow)
    assert filename and pinned, "publish.yaml no longer names its native donor"
    version = filename.group(1).split("-")[1]
    with urllib.request.urlopen(f"https://pypi.org/pypi/cozy-runtime/{version}/json") as response:
        (url,) = [
            row["url"]
            for row in json.load(response)["urls"]
            if row["filename"] == filename.group(1)
        ]
    path = tmp_path_factory.mktemp("donor") / filename.group(1)
    with urllib.request.urlopen(url) as response:
        path.write_bytes(response.read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pinned.group(1)
    return path


@pytest.fixture
def release_copy(tmp_path: Path, donor: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "source"
    cloned = run("git", "clone", "--shared", str(ROOT), str(repo), cwd=tmp_path)
    assert cloned.returncode == 0, cloned.stderr
    for key, value in (
        ("user.name", "Native reuse proof"),
        ("user.email", "proof@example.invalid"),
    ):
        assert run("git", "config", key, value, cwd=repo).returncode == 0
    commit = run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    return repo, donor, commit


def build(
    repo: Path, donor: Path, commit: str, out: Path, *, sha256: str | None = None
) -> subprocess.CompletedProcess[str]:
    return run(
        "uv",
        "run",
        "--script",
        str(repo / "scripts/build-dev-wheel.py"),
        "--native-wheel",
        str(donor),
        "--native-sha256",
        sha256 or hashlib.sha256(donor.read_bytes()).hexdigest(),
        "--release-commit",
        commit,
        "--reuse-or-build",
        "--out",
        str(out),
        cwd=repo,
    )


def test_public_native_reuse_preserves_record_tags_version_and_provenance(
    release_copy: tuple[Path, Path, str],
    tmp_path: Path,
) -> None:
    repo, donor, commit = release_copy
    result = build(repo, donor, commit, tmp_path / "dist")
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    wheel = Path(receipt["wheel"])
    version = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["version"]
    assert wheel.name == f"cozy_runtime-{version}-{TAG}.whl"
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == receipt["sha256"]
    with zipfile.ZipFile(wheel) as archive, zipfile.ZipFile(donor) as original:
        prefix = f"cozy_runtime-{version}.dist-info/"
        metadata = BytesParser().parsebytes(archive.read(prefix + "METADATA"))
        assert metadata["Name"] == "cozy-runtime" and metadata["Version"] == version
        tags = BytesParser().parsebytes(archive.read(prefix + "WHEEL"))
        assert tags.get_all("Tag") == [TAG] and tags["Root-Is-Purelib"] == "false"
        stamp = archive.read("cozy_runtime/_build_provenance.py").decode()
        assert f'COMMIT = "{commit}"' in stamp and "unknown" not in stamp
        assert archive.read(KERNEL) == original.read(KERNEL)
        assert (
            archive.read("cozy_runtime/author/_calls.py")
            == (repo / "src/cozy_runtime/author/_calls.py").read_bytes()
        )
        provenance = json.loads(archive.read("cozy_runtime/_kernels/native-provenance.json"))
        assert provenance["native_wheel_sha256"] == hashlib.sha256(donor.read_bytes()).hexdigest()
        assert (
            provenance["native_extension_sha256"]
            == hashlib.sha256(original.read(KERNEL)).hexdigest()
        )
        assert (
            f'COMMIT = "{provenance["native_source_commit"]}"'
            in original.read("cozy_runtime/_build_provenance.py").decode()
        )
        rows = list(csv.reader(io.StringIO(archive.read(prefix + "RECORD").decode())))
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, encoded, length in rows:
            if name.endswith("/RECORD"):
                assert encoded == length == ""
                continue
            body = archive.read(name)
            assert int(length) == len(body)
            assert (
                encoded
                == "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
            )
    assert run("git", "status", "--porcelain", cwd=repo).stdout == ""


@pytest.mark.parametrize(
    "change", ["native", "backend", "dirty", "commit", "donor_hash", "record", "local_version"]
)
def test_only_changed_native_inputs_authorize_a_full_build(
    release_copy: tuple[Path, Path, str],
    tmp_path: Path,
    change: str,
) -> None:
    repo, donor, commit = release_copy
    sha256 = None
    if change in {"native", "backend", "local_version"}:
        path = repo / ("CMakeLists.txt" if change == "native" else "pyproject.toml")
        source = path.read_text()
        if change == "native":
            source += "\n# changed native input\n"
        elif change == "backend":
            source = source.replace('cmake.build-type = "Release"', 'cmake.build-type = "Debug"')
        else:
            version = tomllib.loads(source)["project"]["version"]
            source = source.replace(f'version = "{version}"', f'version = "{version}+dev.proof"', 1)
        assert source != path.read_text()
        path.write_text(source)
        assert run("git", "add", str(path), cwd=repo).returncode == 0
        assert run("git", "commit", "-m", "Owned negative fixture", cwd=repo).returncode == 0
        commit = run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    elif change == "dirty":
        (repo / "uncommitted.py").write_text("# must not enter a release\n")
    elif change == "commit":
        commit = "0" * 40
    elif change == "donor_hash":
        sha256 = "0" * 64
    elif change == "record":
        corrupt = tmp_path / donor.name
        with zipfile.ZipFile(donor) as archive, zipfile.ZipFile(corrupt, "w") as target:
            for name in archive.namelist():
                body = archive.read(name)
                target.writestr(name, body + b"changed" if name == KERNEL else body)
        donor = corrupt
    result = build(repo, donor, commit, tmp_path / "dist", sha256=sha256)
    assert result.returncode == (3 if change in {"native", "backend"} else 1), result.stderr
    assert not list((tmp_path / "dist").glob("*.whl"))
