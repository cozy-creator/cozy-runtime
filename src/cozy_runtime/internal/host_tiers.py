"""Optional machine-created native host tiers over the existing executor socket.

Only preparation carries a typed partition descriptor. This adds no inference-step exchange
and does not treat a progress/idle observation as a CUDA release fence.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import msgspec
from tensorfs import plane as native_plane

from cozy_runtime.author._executor_requests import HostTierPlan, HostTierPrepare
from cozy_runtime.internal import canonical

CAPABILITY = "host_tiers.owner/1"
INITIAL_BUDGET = "host_tiers.initial_budget/1"
DISK_CAPABILITY = "host_tiers.disk/1"
_LIBC = ctypes.CDLL(None, use_errno=True)
_MEMFD = getattr(_LIBC, "memfd_create", None)
if _MEMFD is not None:
    _MEMFD.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    _MEMFD.restype = ctypes.c_int
# Linux UAPI: older python-build-standalone omits these fcntl constants.
_ADD_SEALS = 1033
_GET_SEALS = 1034
_SEALS = 0x0F  # SEAL, SHRINK, GROW, WRITE


def available() -> bool:
    return sys.platform == "linux" and _MEMFD is not None


def disk_available() -> bool:
    return available() and hasattr(native_plane.Plane, "register_machine_host")


@contextmanager
def sealed_plan(plan: HostTierPlan) -> Iterator[HostTierPrepare]:
    """Native model-library metadata; immutable and independent of control-frame size."""
    body = canonical.write(msgspec.to_builtins(plan))
    if _MEMFD is None:
        raise OSError("host partition memfd capability unavailable")
    fd = _MEMFD(b"cozy-host-tier-plan", 1 | 2)  # MFD_CLOEXEC | MFD_ALLOW_SEALING
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create")
    try:
        view = memoryview(body)
        while view:
            view = view[os.write(fd, view):]
        fcntl.fcntl(fd, _ADD_SEALS, _SEALS)
        yield HostTierPrepare(manifest=plan.manifest, name=plan.name, layout=plan.layout,
                              length=len(body), sha256=hashlib.sha256(body).hexdigest(), memfd=fd)
    finally:
        os.close(fd)
