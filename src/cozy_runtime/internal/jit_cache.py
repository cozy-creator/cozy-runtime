"""Pod-scoped node-local directories for upstream compiler caches.

Upstream keys are an optimization, not a same-UID security boundary. Tensorhub guarantees one
public trust domain per pod; private renter cross-org stacking retains its documented same-UID
risk. Runtime scopes cache reuse to one unpredictable worker boot plus an installed executor
lifetime and deletes the whole boot scope only after worker shutdown has reclaimed the executor
tree.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from cozy_runtime.internal import kernel_cache
from cozy_runtime.internal.child_env import JIT_CACHE_ENV

_INSTALLATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")


class JITCacheRefusal(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Scope:
    root: Path
    environment: dict[str, str]


def scope(
    cozy_home: Path,
    installation_id: str,
    *,
    pod_scope: str,
    uid: int = -1,
    gid: int = -1,
) -> Scope:
    """Create/reuse one installation cache inside one trusted worker-boot scope."""

    if (
        not cozy_home.is_absolute()
        or _INSTALLATION.fullmatch(installation_id) is None
        or not pod_scope
        or len(pod_scope) > 256
    ):
        raise JITCacheRefusal("invalid node-local JIT cache identity")
    pod_key = hashlib.sha256(pod_scope.encode()).hexdigest()
    parent = cozy_home / "jit-cache"
    pod = parent / pod_key
    content = pod / installation_id
    for boundary in (parent, pod):
        boundary.mkdir(parents=True, exist_ok=True, mode=0o711)
        _require_boundary(boundary)
        boundary.chmod(0o711)
    wanted_gid = gid if gid >= 0 else os.getegid()
    _owned_directory(content, uid, wanted_gid)
    _owned_directory(content / "torch-kernels", uid, wanted_gid)
    environment = {name: str(content / directory) for name, directory in JIT_CACHE_ENV}
    return Scope(content, {**environment, "TMPDIR": str(content)})


def remove_pod(cozy_home: Path, pod_scope: str) -> None:
    """Delete one worker boot's caches after its process tree is proven absent."""

    if not pod_scope or len(pod_scope) > 256:
        raise JITCacheRefusal("invalid JIT cache pod scope")
    parent = cozy_home / "jit-cache"
    pod = parent / hashlib.sha256(pod_scope.encode()).hexdigest()
    try:
        relative = pod.relative_to(parent)
    except ValueError as exc:
        raise JITCacheRefusal(f"JIT cache pod escaped its root: {pod}") from exc
    if len(relative.parts) != 1 or re.fullmatch(r"[0-9a-f]{64}", relative.name) is None:
        raise JITCacheRefusal(f"invalid JIT cache pod path: {pod}")
    if not pod.exists():
        return
    info = pod.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid != os.geteuid():
        raise JITCacheRefusal(f"JIT cache pod is mutable or unowned: {pod}")
    pod.chmod(0o700)
    shutil.rmtree(pod)


def machine(root: Path, *, uid: int = -1, gid: int = -1, installation: str = "") -> dict[str, str]:
    """The executor's namespaces of the persistent kernel store, and FA4's, Triton's and, for
    an installed environment, PyTorch's NVRTC caches inside its own.

    Unlike the boot scope above, nothing here is deleted: entries are keyed by every compile
    input, so a worker restart, a new package version or a Runtime update reads them as it
    found them. The boundary is the WRITER: an executor writes only `u<its uid>` and reads the
    worker's `u<worker uid>`, so under uid isolation no placement writes where another reads.
    """

    if not root.is_absolute():
        raise JITCacheRefusal("kernel cache root must be absolute")
    worker = kernel_cache.namespace(root, os.geteuid())
    root.mkdir(parents=True, exist_ok=True, mode=0o711)
    worker.mkdir(exist_ok=True, mode=0o755)
    for boundary in (root, worker):
        _require_boundary(boundary)
    root.chmod(0o711)
    worker.chmod(0o755)
    own = worker
    if uid >= 0 and uid != os.geteuid():
        own = kernel_cache.namespace(root, uid)
        _owned_directory(own, uid, gid)
    environment = {
        kernel_cache.OWN_ENV: str(own),
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": str(own / "flash-attn4"),
        # Triton keys every entry by kernel source, constants, target and its own version, so
        # a shape compiled once (Sol's per-length preprocess, a package's kernels) serves the
        # machine's later executors and boots too (owner, 2026-09-27).
        "TRITON_CACHE_DIR": str(own / "triton"),
    }
    if installation:
        if _INSTALLATION.fullmatch(installation) is None:
            raise JITCacheRefusal("invalid installation identity for the kernel cache")
        # PyTorch keys an entry by kernel name, arch, NVRTC version and source hash but not by
        # its own build, so one installation's Torch never reads another's. PyTorch never
        # creates the directory it is named.
        torch_kernels = own / f"torch-kernels.{installation}"
        _owned_directory(torch_kernels, uid, gid)
        environment["PYTORCH_KERNEL_CACHE_PATH"] = str(torch_kernels)
    if own != worker:
        environment[kernel_cache.TRUSTED_ENV] = str(worker)
    return environment


def _owned_directory(path: Path, uid: int, gid: int) -> None:
    """A private real directory owned by `uid`/`gid`; -1 keeps this process's uid and the
    directory's group."""
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise JITCacheRefusal(f"JIT cache directory is not a real directory: {path}")
    wanted = (uid if uid >= 0 else os.geteuid(), gid if gid >= 0 else info.st_gid)
    if (info.st_uid, info.st_gid) != wanted:
        if os.geteuid() != 0:
            raise JITCacheRefusal(f"JIT cache directory has the wrong owner: {path}")
        os.chown(path, *wanted)
    path.chmod(0o700)


def _require_boundary(path: Path) -> None:
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o022
    ):
        raise JITCacheRefusal(f"JIT cache boundary is mutable or unowned: {path}")
