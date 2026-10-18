"""DIAGNOSTIC ONLY, never merged: at a Triton launch the driver refuses a pointer of, what each
tensor operand is as the driver, the spill tables and the weight plane see it. The text rides
the exception as notes; the invocation boundary writes them to the run log."""

from __future__ import annotations

import ctypes
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Iterator
from typing import Any

try:
    from triton.backends.nvidia import driver as _driver
except ImportError:  # no Triton: nothing to watch
    _driver = None

_LIB: list[Any] = []
#: (monotonic s, what, weight set, region, its address then, the caller) of every unbinding
EVENTS: deque[tuple[float, str, str, int, int, str]] = deque(maxlen=4096)


def note(what: str, ws: str, region: int, address: int | None) -> None:
    try:
        caller = sys._getframe(2)
    except ValueError:
        caller = sys._getframe(1)
    EVENTS.append(
        (time.monotonic(), what, ws, region, address or 0, f"{caller.f_code.co_name}:{caller.f_lineno}")
    )


def _history(ws: str, region: int) -> str:
    now = time.monotonic()
    rows = [e for e in EVENTS if e[2] == ws and e[3] == region][-4:]
    return "; ".join(f"{e[1]} {now - e[0]:.2f}s ago at {e[4]:#x} by {e[5]}" for e in rows) or "no unbinding"


def _lib() -> Any:
    if not _LIB:
        _LIB.append(ctypes.CDLL("libcuda.so.1"))
    return _LIB[0]


def _attr(ptr: int, which: int) -> tuple[int, int]:
    out = ctypes.c_uint64(0)
    status = _lib().cuPointerGetAttribute(ctypes.byref(out), ctypes.c_int(which), ctypes.c_uint64(ptr))
    return int(status), int(out.value)


def _context() -> str:
    lib = _lib()
    current, device, primary = ctypes.c_void_p(0), ctypes.c_int(-1), ctypes.c_void_p(0)
    s1 = lib.cuCtxGetCurrent(ctypes.byref(current))
    s2 = lib.cuCtxGetDevice(ctypes.byref(device))
    s3 = lib.cuDevicePrimaryCtxRetain(ctypes.byref(primary), device)
    if s3 == 0:
        getattr(lib, "cuDevicePrimaryCtxRelease_v2", lib.cuDevicePrimaryCtxRelease)(device)
    same = current.value == primary.value
    return (
        f"thread={threading.current_thread().name} ctx={current.value} ({s1}) dev={device.value} "
        f"({s2}) primary={primary.value} ({s3}) current_is_primary={same}"
    )


def _tensors(value: object, depth: int = 0) -> Iterator[Any]:
    if hasattr(value, "untyped_storage") and hasattr(value, "data_ptr"):
        yield value
    elif depth < 2 and isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item, depth + 1)
    elif depth < 2 and isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item, depth + 1)


def _weight(ptr: int) -> str:
    weights = sys.modules.get("cozy_runtime.internal.weights")
    if weights is None:
        return ""
    for held in list(getattr(weights, "_REFUSALS", ())):
        for component in held.components.values():
            cursor = component.cursor
            ring = (cursor.ring_ptr, cursor.ring_ptr + cursor.ring_nbytes) if cursor else (0, 0)
            orphan = any(o.ring_ptr <= ptr < o.ring_ptr + o.ring_nbytes for o in component.orphans)
            for region in component.regions:
                homes = {*([region.bound] if region.bound is not None else []), *region.views}
                for a in homes:
                    if a <= ptr < a + region.span:
                        return (
                            f"weight {component.name}.{region.path}#{region.index} "
                            f"{'bound' if a == region.bound else 'cached-view'} "
                            f"bound={region.bound} in_ring={ring[0] <= a < ring[1]} "
                            f"orphan_ring={orphan} running={component.key in held.running} "
                            f"held={region.index in component.holds} "
                            f"history: {_history(str(component.name), region.index)}"
                        )
                for d in region.derived:
                    if hasattr(d, "data_ptr") and d.data_ptr() == ptr:
                        return f"derived of {component.name}.{region.path}"
            if ring[0] <= ptr < ring[1]:
                return f"ring of {component.name}"
            if orphan:
                return f"orphan ring of {component.name}"
    return ""


def leases(component: Any) -> list[str]:
    """What this process holds of a weight set, as Runtime keeps it."""
    rows = [
        f"DIAG set {component.name}: holds={sorted(component.holds)} lease={component.lease is not None} "
        f"cursor={component.cursor is not None} orphans={len(component.orphans)} "
        f"shared={sorted(component.shared)} machine_leases={len(component.leases)} "
        f"filling={len(component.filling)} spared={component.spared}",
        "DIAG bound regions: "
        + ",".join(str(r.index) for r in component.regions if r.bound is not None)[:900],
    ]
    rows.extend(
        f"DIAG unbinding {e[1]} r{e[3]} at {e[4]:#x} by {e[5]} {time.monotonic() - e[0]:.2f}s ago"
        for e in [e for e in EVENTS if e[2] == str(component.name)][-12:]
    )
    return rows


def describe(tensor: Any) -> tuple[bool, str]:
    """(whether the driver refuses its pointer, a line on it)."""
    spill = sys.modules.get("cozy_runtime.internal.spill")
    try:
        ptr = int(tensor.data_ptr())
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(tensor).__name__} data_ptr raises {type(exc).__name__}"
    status, _ = _attr(ptr, 3) if ptr else (0, 0)
    head = (
        f"{type(tensor).__name__} {tuple(tensor.shape)} {tensor.dtype} {tensor.device} "
        f"ptr={ptr:#x} devptr_status={status}"
    )
    if not status:
        return False, head
    kind = _attr(ptr, 2)
    start, size = _attr(ptr, 11), _attr(ptr, 12)
    flags = [f"memtype={kind}", f"range={start[1]:#x}+{size[1]} ({start[0]},{size[0]})"]
    try:
        storage = tensor.untyped_storage()
        key, base = int(storage._cdata), int(storage.data_ptr())
        flags.append(f"storage_base={base:#x} nbytes={storage.nbytes()} resizable={storage.resizable()}")
        if spill is not None:
            off = spill.OFF.get(key)
            if off is not None:
                flags.append("spill:" + ("out" if off.out else "mapped"))
            flags.append(f"host_block={base in spill.HELD} native={key in spill.NATIVE}")
            flags.append(f"used={spill.USED.get(key, (None,))[0]} clock={spill._CLOCK[0]}")
    except Exception as exc:  # noqa: BLE001
        flags.append(f"storage raises {type(exc).__name__}: {exc}"[:120])
    try:
        flags.append(_weight(ptr) or "not a plane weight")
    except Exception as exc:  # noqa: BLE001
        flags.append(f"weight lookup raises {exc}"[:120])
    return True, head + " " + " ".join(flags)


def _watch() -> None:
    if _driver is None or getattr(_driver.CudaLauncher, "_diag", False):
        return
    original = _driver.CudaLauncher.__call__

    def call(self: Any, *args: Any) -> Any:
        try:
            return original(self, *args)
        except ValueError as exc:
            if "cannot be accessed from Triton" not in str(exc):
                raise
            notes = []
            try:
                caller = sys._getframe(1).f_locals.get("self")
                name = getattr(caller, "__name__", None) or getattr(
                    getattr(caller, "fn", None), "__name__", type(caller).__name__
                )
                notes.append(f"DIAG kernel={name} {_context()}")
                seen = 0
                for tensor in _tensors(args[9:]):
                    bad, line = describe(tensor)
                    seen += 1
                    if bad:
                        notes.append("DIAG bad " + line)
                notes.append(f"DIAG {seen} tensor operands")
                stack = traceback.extract_stack(limit=24)[:-1]
                notes.extend(f"DIAG at {f.filename.rsplit('/', 2)[-1]}:{f.lineno} {f.name}" for f in stack)
            except Exception as diag:  # noqa: BLE001
                notes.append(f"DIAG failed: {type(diag).__name__}: {diag}")
            for note in notes:
                exc.add_note(note)
            raise

    _driver.CudaLauncher.__call__ = call
    _driver.CudaLauncher._diag = True


_watch()
