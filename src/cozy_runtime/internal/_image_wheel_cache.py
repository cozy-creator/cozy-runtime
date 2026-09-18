"""Image-owned inspected wheel facts; uv owns every installation and cached hash.

This cache is immutable public image content, separate from the private, evictable
machine cache. Symlinks avoid Docker overlayfs copying a whole lower-layer Torch
payload up when the first private environment is created.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from typing import TYPE_CHECKING

from cozy_runtime.internal.config import package_install_environment

if TYPE_CHECKING:
    from cozy_runtime.internal import base_observation, package_environment
from dataclasses import asdict
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path("/opt/cozy/dependency-seed")
INDEXES = (
    "https://pypi.org/simple",
    "https://download.pytorch.org/whl/cpu",
    "https://download.pytorch.org/whl/cu130",
)


def record(path: Path, base: base_observation.Observation) -> None:
    """Build-only admission of an original, hash-verified wheel before discarding it."""
    from cozy_runtime.internal import package_environment as pe

    item, _ = pe._measure_inspected_wheel(path, "custom", base, allow_base_owned=True)
    # Prefix resources need the original archive for their post-install verifier.
    # Keep those uncommon wheels on the ordinary verified archive path.
    if any(pe._prefix_resource(m.filename) is not None for m in item.members):
        return
    directory = ROOT / "facts"
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o755)
    value = {
        "uv_version": base.uv_version,
        "wheel": asdict(item.wheel),
        "members": [[m.filename, m.file_size] for m in item.members],
        "requires_python": item.requires_python,
        "requirements": [str(r) for r in item.requirements],
        "native_members": list(item.native_members),
    }
    target = directory / (item.wheel.digest.removeprefix("sha256:") + ".json")
    target.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    target.chmod(0o444)


def read(digest: str, base: base_observation.Observation) -> package_environment._Inspected | None:
    from cozy_runtime.internal import package_environment as pe

    pe._require_digest(digest, "seed.digest")
    path = ROOT / "facts" / (digest.removeprefix("sha256:") + ".json")
    if not path.exists() and not path.is_symlink():
        return None
    try:
        for target in (ROOT, ROOT / "facts", path):
            info = target.lstat()
            if target.resolve() != target or info.st_uid != os.geteuid() or info.st_mode & 0o022:
                raise ValueError("image seed is not owned immutable image content")
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o222
            or not 0 < info.st_size <= pe.MAX_METADATA_BYTES
        ):
            raise ValueError("invalid image seed facts")
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("image seed facts must be an object")
        if value["uv_version"] != base.uv_version:
            return None
        row = value["wheel"]
        row["tags"] = tuple(row["tags"])
        row["import_roots"] = tuple(row["import_roots"])
        wheel = pe.Wheel(**row)
        if wheel.digest != digest or wheel.owner != "custom":
            raise ValueError("image seed wheel identity changed")
        try:
            pe._require_compatible_tags(wheel, base)
        except pe.EnvironmentRefusal as exc:
            if exc.code == "package_environment_wheel_platform_incompatible":
                return None
            raise
        members = []
        if not isinstance(value["members"], list) or len(value["members"]) > pe.MAX_WHEEL_MEMBERS:
            raise ValueError("invalid image seed member count")
        for name, size in value["members"]:
            if (
                not isinstance(name, str)
                or not name
                or type(size) is not int
                or not 0 <= size <= pe.MAX_WHEEL_BYTES
            ):
                raise ValueError("invalid image seed member")
            member = zipfile.ZipInfo(name)
            member.file_size = size
            members.append(member)
        return pe._Inspected(
            wheel,
            path,
            tuple(members),
            frozenset(wheel.import_roots),
            value["requires_python"],
            tuple(Requirement(r) for r in value["requirements"]),
            tuple(value["native_members"]),
            image_seed=True,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise pe.EnvironmentRefusal("package_environment_image_seed_invalid", str(path)) from exc


def command(uv: Path | str, python: Path | str, requirements: Path) -> list[str]:
    return [
        str(uv),
        "pip",
        "install",
        "--no-config",
        "--offline",
        "--no-deps",
        "--no-build",
        "--require-hashes",
        "--link-mode",
        "symlink",
        "--cache-dir",
        str(ROOT / "uv-cache"),
        "--index-url",
        INDEXES[0],
        "--index-strategy",
        "unsafe-best-match",
        *(part for index in INDEXES[1:] for part in ("--extra-index-url", index)),
        "--python",
        str(python),
        "--requirements",
        str(requirements),
    ]


def requirement(item: package_environment._Inspected) -> str:
    return f"{item.wheel.distribution}=={item.wheel.version} --hash={item.wheel.digest}\n"


def available(item: package_environment._Inspected, python: Path, cache_root: Path) -> bool:
    """Ask uv itself to prove exact cached bytes, including after cache eviction.

    A dry run is insufficient: uv reports success even when it would need to
    download a different hash. An actual offline installation checks the hash.
    The temporary target contains only symlinks and small private metadata.
    """
    from cozy_runtime.internal import package_environment as pe
    from cozy_runtime.internal import storage_admission

    cache = ROOT / "uv-cache"
    if not cache.exists() and not cache.is_symlink():
        return False
    info = cache.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or cache.resolve() != cache
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o022
    ):
        raise pe.EnvironmentRefusal("package_environment_image_seed_invalid", str(cache))
    count = len(item.members)
    with (
        storage_admission.admit(
            storage_admission.Write(cache_root, (count + 128) * 4096, count + 128)
        ),
        tempfile.TemporaryDirectory(prefix=".image-seed-probe-", dir=cache_root) as temporary,
    ):
        root = Path(temporary)
        req = root / "requirements.txt"
        req.write_text(requirement(item))
        result = subprocess.run(
            [*command(pe._uv_binary(), python, req), "--target", str(root / "site")],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=package_install_environment(),
            check=False,
        )
        return result.returncode == 0


def private_metadata(staging: Path) -> None:
    """Keep venv metadata private while sharing only the immutable package payload."""
    for info in staging.glob("lib/python*/site-packages/*.dist-info"):
        for path in info.rglob("*"):
            if path.is_symlink() and path.resolve().is_relative_to(ROOT / "uv-cache"):
                temporary = path.with_name(path.name + ".image-copy")
                shutil.copyfile(path, temporary)
                os.replace(temporary, path)
