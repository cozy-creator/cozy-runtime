"""Node-local upstream compiler-cache scoping; cache bytes are never Cozy artifacts."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from cozy_runtime.internal.child_env import JIT_CACHE_ENV


class JITCacheRefusal(Exception):
    pass


def scope(
    cozy_home: Path,
    content_digest: str,
    *,
    uid: int = -1,
    gid: int = -1,
) -> dict[str, str]:
    """Create/reuse one exact base-plus-overlay scope and return upstream env values."""

    if not cozy_home.is_absolute() or not re.fullmatch(r"sha256:[0-9a-f]{64}", content_digest):
        raise JITCacheRefusal("invalid node-local JIT cache identity")
    parent = cozy_home / "jit-cache"
    cache = parent / content_digest.removeprefix("sha256:")
    parent.mkdir(parents=True, exist_ok=True, mode=0o711)
    parent_info = parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise JITCacheRefusal(f"JIT cache parent is not a real directory: {parent}")
    cache.mkdir(mode=0o700, exist_ok=True)
    info = cache.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise JITCacheRefusal(f"JIT cache scope is not a real directory: {cache}")
    wanted_uid = uid if uid >= 0 else os.geteuid()
    wanted_gid = gid if gid >= 0 else os.getegid()
    if info.st_uid != wanted_uid or info.st_gid != wanted_gid:
        if os.geteuid() != 0:
            raise JITCacheRefusal(f"JIT cache scope has the wrong owner: {cache}")
        os.chown(cache, wanted_uid, wanted_gid)
    cache.chmod(0o700)
    return {name: str(cache / directory) for name, directory in JIT_CACHE_ENV}
