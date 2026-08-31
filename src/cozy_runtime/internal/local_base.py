"""Truthful ownership manifest for the installed local Runtime environment."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import re
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from cozy_runtime.internal import canonical, overlay

_NAME_RUNS = re.compile(r"[-_.]+")
_ROOT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class LocalBaseRefusal(Exception):
    pass


def manifest_bytes() -> bytes:
    """Observe the exact interpreter environment this Runtime will use as its base."""

    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise LocalBaseRefusal("local base requires CPython 3.12")
    machine = platform.machine().lower()
    if sys.platform != "linux" or machine not in {"amd64", "x86_64"}:
        raise LocalBaseRefusal("local base currently requires Linux x86-64")
    libc, libc_version = platform.libc_ver()
    if libc != "glibc" or not re.fullmatch(r"[0-9]+\.[0-9]+", libc_version):
        raise LocalBaseRefusal("local base requires a readable glibc version")

    distributions, roots = _ownership()
    if any(row["distribution"] == "torch" for row in distributions):
        raise LocalBaseRefusal(
            "the current local Runtime environment contains Torch; use a Runtime-only base"
        )
    raw = canonical.write(
        {
            "base_distributions": distributions,
            "base_import_roots": roots,
            "compatibility_profile": {
                "accelerator_build": "cpu",
                "os_cpu": "linux-x86",
                "python_abi": "cp312",
                "torch_release": "none",
            },
            "format": "WheelhouseManifest/1",
            "minimum_cuda_version": "none",
            "minimum_driver_version": "none",
            "ownership": "observed-local",
            "platform_target": {
                "accelerator_abi": "none",
                "accelerator_backend": "none",
                "libc": "glibc" + libc_version,
                "os_arch": "linux/amd64",
                "python_abi": "cp312",
            },
            # Installed environments do not retain original wheel bytes. Claiming their
            # digests here would be fabricated; distribution/import ownership is what the
            # overlay collision and requirement checks consume.
            "wheels": [],
        }
    )
    overlay.parse_base_manifest(raw, _digest(raw))
    return raw


def write_manifest(path: Path) -> tuple[str, int, bool]:
    raw = manifest_bytes()
    digest = _digest(raw)
    try:
        if path.read_bytes() == raw:
            return digest, len(raw), False
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise LocalBaseRefusal(f"cannot read local base manifest: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise LocalBaseRefusal(f"cannot write local base manifest: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest, len(raw), True


def _ownership() -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    versions: dict[str, str] = {}
    providers: dict[str, set[str]] = {}
    grouped: dict[str, list[importlib.metadata.Distribution]] = {}
    for distribution in importlib.metadata.distributions():
        name = _normalize(distribution.metadata.get("Name", ""))
        if not name or not distribution.version:
            raise LocalBaseRefusal("installed distribution has no normalized name or version")
        grouped.setdefault(name, []).append(distribution)
    for name, candidates in grouped.items():
        ranked = sorted(
            ((_path_rank(candidate), candidate) for candidate in candidates), key=lambda row: row[0]
        )
        rank, selected = ranked[0]
        if any(
            other_rank == rank and other.version != selected.version
            for other_rank, other in ranked[1:]
        ):
            raise LocalBaseRefusal(f"installed distribution {name} has conflicting versions")
        versions[name] = selected.version
        for root in _distribution_roots(selected):
            providers.setdefault(root, set()).add(name)
    for root in sys.stdlib_module_names:
        try:
            present = importlib.util.find_spec(root) is not None
        except (ImportError, AttributeError, ValueError):
            present = False
        if present and _ROOT.fullmatch(root):
            providers.setdefault(root, set()).add("python-stdlib")
    return (
        [{"distribution": name, "version": versions[name]} for name in sorted(versions)],
        [{"providers": sorted(providers[root]), "root": root} for root in sorted(providers)],
    )


def _distribution_roots(distribution: importlib.metadata.Distribution) -> set[str]:
    roots: set[str] = set()
    for entry in distribution.files or ():
        parts = PurePosixPath(str(entry)).parts
        if not parts:
            continue
        first = parts[0]
        if first.endswith((".dist-info", ".egg-info")):
            continue
        if first.endswith(".data"):
            if len(parts) < 3 or parts[1] not in {"purelib", "platlib"}:
                continue
            first = parts[2]
        if first == "__pycache__" or first.endswith(".libs"):
            continue
        if first.endswith((".py", ".pyi")):
            first = first.rsplit(".", 1)[0]
        elif "." in first:
            first = first.split(".", 1)[0]
        if _ROOT.fullmatch(first):
            roots.add(first)
    return roots


def _path_rank(distribution: importlib.metadata.Distribution) -> int:
    root = Path(str(distribution.locate_file(""))).resolve()
    for index, entry in enumerate(sys.path):
        if entry and Path(entry).resolve() == root:
            return index
    raise LocalBaseRefusal(
        f"installed distribution {distribution.metadata.get('Name', '')} is outside sys.path"
    )


def _normalize(value: str) -> str:
    return _NAME_RUNS.sub("-", value.strip().lower())


def _digest(raw: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(raw).hexdigest()
