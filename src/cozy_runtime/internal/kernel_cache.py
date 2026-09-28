"""The machine's compiled artifacts, keyed by every input that shapes their bytes.

An artifact is anything compiled on this machine for this machine: a CuTe object (Sol's per-length
forward), an extension kernel's importable tree (SageAttention built by its own setup.py for
the card present), a Triton cache fill. Its key is the complete input document: sources by
digest, target architecture, torch, CUDA, Python and compiler, plus any specialization.

Entries live in WRITER namespaces, `<machine kernels>/u<uid>`. An executor writes only its own
uid's namespace and also reads the worker's; it never reads another placement's. A user-run
machine runs worker and executors as one uid, so that is one machine-wide namespace; under uid
isolation no placement can plant an artifact that another placement loads.

    <namespace>/<kernel>/<digest>/entry.json   key document and producer, written last
    <namespace>/<kernel>/<digest>/object       an object entry's bytes (sha256 in entry.json)
    <namespace>/<kernel>/<digest>/site/        a tree entry's importable directory
    <namespace>/<kernel>/<digest>.lock         one builder per key, across processes
    <namespace>/<kernel>/<digest>.state.json   the live builder's progress, or its failure

An entry is built in a private staging directory and renamed into place whole, so a reader
never sees half of one. A lock is `flock`, so a builder that dies releases it: waiting on it
is waiting on a live compile, never on a clock.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import msgspec

SCHEMA = 1
_KERNEL = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_OBJECT = "object"
_ENTRY = "entry.json"
SITE = "site"


class KernelCacheRefusal(Exception):
    pass


class _Entry(msgspec.Struct, frozen=True):
    key: object
    sha256: str = ""
    size: int = -1


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise KernelCacheRefusal(f"kernel key input is not a JSON scalar: {value!r}")


@dataclass(frozen=True, slots=True)
class Key:
    kernel: str
    inputs: str
    """Canonical JSON of every compile input, so equality is exact equality of the document."""

    @classmethod
    def of(cls, kernel: str, inputs: Mapping[str, object]) -> Key:
        if _KERNEL.fullmatch(kernel) is None:
            raise KernelCacheRefusal(f"invalid kernel name {kernel!r}")
        return cls(kernel, json.dumps(_canonical(inputs), sort_keys=True, separators=(",", ":")))

    def document(self) -> dict[str, object]:
        return {"schema": SCHEMA, "kernel": self.kernel, "inputs": json.loads(self.inputs)}

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.document(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class Store:
    """One writer namespace plus, read-only, the worker's."""

    def __init__(self, own: Path, trusted: Path | None = None) -> None:
        if not own.is_absolute() or (trusted is not None and not trusted.is_absolute()):
            raise KernelCacheRefusal("kernel cache namespaces must be absolute")
        self.own = own
        self.trusted = trusted if trusted is not None and trusted != own else None

    def get(self, key: Key) -> bytes | None:
        found = _read(self.own, key, repair=True)
        if found is None and self.trusted is not None:
            found = _read(self.trusted, key, repair=False)
        return found

    def entry(self, key: Key) -> Path | None:
        """A published entry's directory, own namespace first, when its key document is this
        key's; None when absent. An object entry's bytes are verified by `get`, not here."""
        for namespace in (self.own, self.trusted):
            if namespace is None:
                continue
            entry_dir = namespace / key.kernel / key.digest
            try:
                entry = msgspec.json.decode((entry_dir / _ENTRY).read_bytes(), type=_Entry)
            except (OSError, ValueError):
                continue
            if entry.key == key.document():
                return entry_dir
        return None

    def put(self, key: Key, data: bytes, producer: Mapping[str, object]) -> None:
        """Publish an object entry; every object is a pure function of its key."""

        def fill(staging: Path) -> Mapping[str, object]:
            _write(staging / _OBJECT, data)
            return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), **producer}

        self.publish(key, fill)

    def publish(self, key: Key, fill: Callable[[Path], Mapping[str, object]]) -> Path:
        """Build an entry in a private staging directory, then rename it into place whole.

        `fill` writes the entry's content and returns its producer facts; `sha256` and
        `size` among them verify an object entry. The first complete entry wins and a later
        one is dropped: only the lock holder publishes a digest, so any leftover staging
        directory is a dead writer's."""
        kernel = self._kernel_dir(key)
        final = kernel / key.digest
        if final.exists():
            return final
        for stale in kernel.glob(f".tmp-{key.digest}-*"):
            shutil.rmtree(stale, ignore_errors=True)
        staging = kernel / f".tmp-{key.digest}-{secrets.token_hex(8)}"
        staging.mkdir(mode=0o755)
        try:
            facts = dict(fill(staging))
            entry: dict[str, object] = {"key": key.document(), "producer": facts}
            for name in ("sha256", "size"):
                if name in facts:
                    entry[name] = facts.pop(name)
            _write(staging / _ENTRY, json.dumps(entry, sort_keys=True, indent=1).encode())
            _fsync_dir(staging)
            try:
                os.rename(staging, final)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
            _fsync_dir(kernel)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return final

    def state(self, key: Key) -> dict[str, object] | None:
        """The builder's last word on this key, own namespace first: progress or failure."""
        for namespace in (self.own, self.trusted):
            if namespace is None:
                continue
            path = namespace / key.kernel / f"{key.digest}.state.json"
            try:
                found = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(found, dict):
                return found
        return None

    def set_state(self, key: Key, state: Mapping[str, object] | None) -> None:
        path = self._kernel_dir(key) / f"{key.digest}.state.json"
        if state is None:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            return
        scratch = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
        scratch.write_text(json.dumps(dict(state), sort_keys=True))
        os.replace(scratch, path)

    @contextlib.contextmanager
    def building(self, key: Key) -> Iterator[None]:
        """Hold the key's build lock, waiting for whichever live process holds it now."""
        fd = self._open_lock(key)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def claim(self, key: Key) -> int | None:
        """Take the build lock without waiting, for a builder process to inherit; or None
        when someone already builds this key. The caller hands the descriptor to exactly one
        child and closes its own copy: the lock then lives exactly as long as that child."""
        fd = self._open_lock(key)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        return fd

    def built_by_worker(self, key: Key) -> bool:
        """The worker's live builder holds this key: an executor under uid isolation reads
        its lock without writing there, so it waits on the machine's build, never duplicates
        it."""
        if self.trusted is None:
            return False
        try:
            fd = os.open(
                self.trusted / key.kernel / f"{key.digest}.lock", os.O_RDONLY | os.O_CLOEXEC
            )
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            os.close(fd)
        return False

    def _kernel_dir(self, key: Key) -> Path:
        kernel = self.own / key.kernel
        kernel.mkdir(mode=0o755, parents=True, exist_ok=True)
        return kernel

    def _open_lock(self, key: Key) -> int:
        path = self._kernel_dir(key) / f"{key.digest}.lock"
        return os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)


def _read(namespace: Path, key: Key, *, repair: bool) -> bytes | None:
    entry_dir = namespace / key.kernel / key.digest
    entry: _Entry | None
    try:
        entry = msgspec.json.decode((entry_dir / _ENTRY).read_bytes(), type=_Entry)
        data = (entry_dir / _OBJECT).read_bytes()
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        entry, data = None, b""
    if (
        entry is not None
        and entry.size >= 0
        and entry.key == key.document()
        and entry.size == len(data)
        and entry.sha256 == hashlib.sha256(data).hexdigest()
    ):
        return data
    if repair:
        # A torn or foreign entry is a miss, and it must not shadow the rebuild.
        doomed = entry_dir.with_name(f".bad-{key.digest}-{secrets.token_hex(8)}")
        with contextlib.suppress(OSError):
            os.rename(entry_dir, doomed)
            shutil.rmtree(doomed, ignore_errors=True)
    return None


def _write(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def namespace(root: Path, uid: int) -> Path:
    return root / f"u{uid}"


def from_sealed(sealed: Mapping[str, str]) -> Store | None:
    """The executor's store, from the two locations its worker imposed; none when absent."""
    own = sealed.get(OWN_ENV, "")
    if not own:
        return None
    trusted = sealed.get(TRUSTED_ENV, "")
    return Store(Path(own), Path(trusted) if trusted else None)


_MACHINE_STORE: list[Store | None] = [None]


def configure(store: Store | None) -> None:
    """Set once by the executor entrypoint from its sealed configuration."""
    _MACHINE_STORE[0] = store


def configured() -> Store | None:
    """This process's store, or None: kernels then compile in-process as upstream does."""
    return _MACHINE_STORE[0]


#: Where this executor writes compiled kernel objects: its own uid's namespace.
OWN_ENV = "COZY_KERNEL_CACHE"
#: The worker's namespace, read-only to an isolated executor.
TRUSTED_ENV = "COZY_KERNEL_CACHE_TRUSTED"
