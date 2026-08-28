"""Digest-keyed, immutable, hardlinked store of unpacked wheel trees.

Every venv used to unpack the whole torch family independently: two generations shared zero
inodes and each paid a full ``uv pip install``.  The hermetic seal is why -- ``--no-cache``
is one clause of ``--no-config --no-index --offline --no-deps --no-build --no-cache`` -- and
the seal stays.  uv's own cache is mutable, ambient, undigested state and would become a
cross-tenant poisoning surface under endpoint stacking; this store is none of those things.

    wheel digest + layout key -> one immutable unpacked tree, hardlinked into every venv

The store is a first-party input, never a resolver: it downloads nothing, decides nothing,
and adds no distribution the exact inventory did not already name.  It cannot poison a
generation either -- :func:`environment._verify_installed` re-hashes every linked file
against its distribution's ``RECORD`` and the receipt re-measures every ``record_digest``,
so a corrupted entry is a refusal, not an install.

Hardlinks, never symlinks: ``_verify_record`` refuses a symlinked ``RECORD`` row, while a
hardlink is an ordinary file whose ``resolve()`` stays inside the venv root.  That is also
why the store must share a filesystem with the environments root; it fails closed if not.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ENTRY_FORMAT = "cozy.runtime.WheelStoreEntry/1"
ENTRY_NAME = "entry.json"
TREE_NAME = "tree"
MAX_ENTRY_PATHS = 200_000
_NORMALIZE = re.compile(r"[-_.]+")


class WheelStoreRefusal(Exception):
    """A fail-closed wheel-store verdict with a stable machine-readable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class InstalledDistribution:
    """One distribution as uv left it in a venv: what it owns, relative to the venv root."""

    distribution: str
    wheel_digest: str
    record_digest: str
    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WheelStore:
    """One layout's shard of the store; ``root`` is the shard, not the store root."""

    root: Path
    layout_digest: str

    @classmethod
    def open(
        cls, store_root: Path, environments_root: Path, *, layout: Mapping[str, str]
    ) -> WheelStore:
        """Bind one layout shard, refusing any store that cannot hardlink into the venvs."""

        if not store_root.is_absolute() or not environments_root.is_absolute():
            raise WheelStoreRefusal("wheel_store_path_invalid", str(store_root))
        try:
            store_root.mkdir(parents=True, exist_ok=True, mode=0o755)
            environments_root.mkdir(parents=True, exist_ok=True, mode=0o755)
            store_device = store_root.stat().st_dev
            environments_device = environments_root.stat().st_dev
        except OSError as exc:
            raise WheelStoreRefusal("wheel_store_unusable", str(exc)) from exc
        if store_device != environments_device:
            raise WheelStoreRefusal(
                "wheel_store_cross_device",
                f"{store_root} (dev {store_device}) cannot hardlink into "
                f"{environments_root} (dev {environments_device})",
            )
        digest = _sha(_canonical(dict(layout))).removeprefix("sha256:")
        root = store_root / digest
        try:
            root.mkdir(exist_ok=True, mode=0o755)
        except OSError as exc:
            raise WheelStoreRefusal("wheel_store_unusable", str(exc)) from exc
        return cls(root=root, layout_digest=digest)

    def entry(self, wheel_digest: str) -> Path:
        return self.root / wheel_digest.removeprefix("sha256:")

    def holds(self, wheel_digest: str) -> bool:
        return (self.entry(wheel_digest) / ENTRY_NAME).is_file()

    def link_in(self, wheel_digest: str, venv: Path) -> tuple[str, ...]:
        """Hardlink one immutable entry into ``venv``; the caller still verifies RECORD."""

        entry = self.entry(wheel_digest)
        paths = self._read_entry(entry, wheel_digest)
        tree = entry / TREE_NAME
        for relative in paths:
            source = tree / relative
            destination = venv / relative
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, destination)
            except FileExistsError as exc:
                raise WheelStoreRefusal("wheel_store_path_conflict", relative) from exc
            except OSError as exc:
                if exc.errno == errno.EXDEV:
                    raise WheelStoreRefusal("wheel_store_cross_device", relative) from exc
                raise WheelStoreRefusal("wheel_store_link_failed", f"{relative}: {exc}") from exc
        return paths

    def publish(self, installed: InstalledDistribution, venv: Path) -> None:
        """Harvest one freshly unpacked distribution, or lose the race and keep the winner.

        The entry is built in a private staging directory and renamed into place, so a
        concurrent materialization observes the entry complete or absent, never partial.
        """

        final = self.entry(installed.wheel_digest)
        if (final / ENTRY_NAME).is_file():
            return
        staging = self.root / f".staging-{uuid.uuid4().hex}"
        try:
            staging.mkdir(mode=0o755)
            tree = staging / TREE_NAME
            for relative in installed.paths:
                source = venv / relative
                destination = tree / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError as exc:
                    if exc.errno == errno.EXDEV:
                        raise WheelStoreRefusal("wheel_store_cross_device", relative) from exc
                    raise WheelStoreRefusal(
                        "wheel_store_publish_failed", f"{relative}: {exc}"
                    ) from exc
                # The inode is shared with the venv; clearing write here is exactly what
                # environment._make_immutable would do to it a moment later.
                info = destination.stat()
                if info.st_mode & 0o222:
                    destination.chmod(info.st_mode & ~0o222)
            document = {
                "format": ENTRY_FORMAT,
                "distribution": installed.distribution,
                "layout_digest": self.layout_digest,
                "paths": list(installed.paths),
                "record_digest": installed.record_digest,
                "wheel_digest": installed.wheel_digest,
            }
            _write_durable(staging / ENTRY_NAME, _canonical(document))
            _make_readonly(staging)
            _fsync_dir(staging)
            try:
                os.rename(staging, final)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise WheelStoreRefusal("wheel_store_publish_failed", str(exc)) from exc
                remove_tree(staging)
                return
            _fsync_dir(self.root)
        except WheelStoreRefusal:
            if staging.exists():
                remove_tree(staging)
            raise
        except OSError as exc:
            if staging.exists():
                remove_tree(staging)
            raise WheelStoreRefusal("wheel_store_publish_failed", str(exc)) from exc

    def _read_entry(self, entry: Path, wheel_digest: str) -> tuple[str, ...]:
        try:
            raw = (entry / ENTRY_NAME).read_bytes()
            value = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise WheelStoreRefusal("wheel_store_entry_unreadable", str(exc)) from exc
        if not isinstance(value, dict) or value.get("format") != ENTRY_FORMAT:
            raise WheelStoreRefusal("wheel_store_entry_invalid", str(entry))
        if (
            value.get("wheel_digest") != wheel_digest
            or value.get("layout_digest") != self.layout_digest
        ):
            raise WheelStoreRefusal("wheel_store_entry_wrong_key", str(entry))
        paths = value.get("paths")
        if not isinstance(paths, list) or not paths or len(paths) > MAX_ENTRY_PATHS:
            raise WheelStoreRefusal("wheel_store_entry_invalid", "paths is not a bounded list")
        for relative in paths:
            if not isinstance(relative, str):
                raise WheelStoreRefusal("wheel_store_entry_invalid", "path is not a string")
            _check_relative(relative)
        if list(paths) != sorted(set(paths)):
            raise WheelStoreRefusal("wheel_store_entry_invalid", "paths are not sorted unique")
        return tuple(paths)


def installed_distributions(
    venv: Path, site: Path, wheel_digests: Mapping[str, str]
) -> tuple[InstalledDistribution, ...]:
    """Read what uv just unpacked, from the RECORDs it wrote, without hashing the payload.

    ``environment._verify_installed`` is the authority that these paths are real, owned, and
    byte-exact; this only needs the ownership map so the harvest links the right inodes.
    """

    root = venv.resolve()
    results: list[InstalledDistribution] = []
    for distribution, wheel_digest in sorted(wheel_digests.items()):
        matches = [
            path
            for path in site.glob("*.dist-info")
            if _name(path.name.split("-")[0]) == distribution
        ]
        if len(matches) != 1:
            raise WheelStoreRefusal("wheel_store_harvest_ambiguous", distribution)
        record = matches[0] / "RECORD"
        try:
            with record.open(newline="") as stream:
                rows = [row for row in csv.reader(stream) if row and row[0]]
        except OSError as exc:
            raise WheelStoreRefusal("wheel_store_harvest_unreadable", str(exc)) from exc
        paths: set[str] = set()
        for row in rows:
            resolved = (site / row[0]).resolve()
            if resolved != root and root not in resolved.parents:
                raise WheelStoreRefusal("wheel_store_harvest_escape", row[0])
            paths.add(resolved.relative_to(root).as_posix())
        if not paths:
            raise WheelStoreRefusal("wheel_store_harvest_empty", distribution)
        results.append(
            InstalledDistribution(
                distribution=distribution,
                wheel_digest=wheel_digest,
                record_digest=_hash_file(record),
                paths=tuple(sorted(paths)),
            )
        )
    return tuple(results)


def remove_tree(root: Path) -> None:
    """Delete a tree that may hold store hardlinks.

    Only directories are made writable.  Unlinking a file needs write permission on its
    directory, never on the file, and chmodding a hardlinked file would make the shared
    store inode writable -- silently breaking the store's immutability from a venv teardown.
    """

    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
    root.chmod(root.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
    shutil.rmtree(root)


def _check_relative(relative: str) -> None:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative or not pure.parts:
        raise WheelStoreRefusal("wheel_store_entry_path", relative)


def _make_readonly(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def _write_durable(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 << 20):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _name(value: str) -> str:
    return _NORMALIZE.sub("-", value).lower()


__all__ = [
    "ENTRY_FORMAT",
    "InstalledDistribution",
    "WheelStore",
    "WheelStoreRefusal",
    "installed_distributions",
    "remove_tree",
]
