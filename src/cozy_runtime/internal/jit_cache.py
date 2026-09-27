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
    content.mkdir(mode=0o700, exist_ok=True)
    info = content.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise JITCacheRefusal(f"JIT cache content scope is not a real directory: {content}")
    wanted_uid = uid if uid >= 0 else os.geteuid()
    wanted_gid = gid if gid >= 0 else os.getegid()
    if info.st_uid != wanted_uid or info.st_gid != wanted_gid:
        if os.geteuid() != 0:
            raise JITCacheRefusal(f"JIT cache content scope has the wrong owner: {content}")
        os.chown(content, wanted_uid, wanted_gid)
    content.chmod(0o700)
    return Scope(content, {name: str(content / directory) for name, directory in JIT_CACHE_ENV})


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


def _require_boundary(path: Path) -> None:
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o022
    ):
        raise JITCacheRefusal(f"JIT cache boundary is mutable or unowned: {path}")
