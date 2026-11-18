"""PROOF ONLY (var-030; never merge): what a capacity refusal found on the card.

At each refusal the execution layer sees (`capture`), and at the one that ends the call
(`final`, before its frames go): the card's room as varena refused it and now, every live CUDA
tensor (named by the failed frames' variables that hold it), the resident weights by model and
block, and the last ladder events (`event`). The request's record carries them as `refusals`
(`residency.Request.stats`) and stderr says them (`say`).

Only frames that have already failed are read for their variables (a traceback's, below the
frame that caught it): reading a running frame's would keep its tensors alive (3.12's
`f_locals` snapshot).
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from collections import deque
from typing import Any

import tensorfs.varena.torch as vt
import torch
from torch import Tensor, nn

#: the ladder's last decisions, oldest first: selections, refusals, retries, host rungs
EVENTS: deque[dict[str, Any]] = deque(maxlen=400)
#: this request's refusals (`take`)
DUMPS: list[dict[str, Any]] = []
_MOST = 8
_SEEN = "_cozy_var030"
#: the site or block running now: (name, began, the room then)
_AT: list[Any] = ["", 0.0, None]
_BLOCK: list[Any] = ["", 0.0]
#: the room each device's last refused allocation found, until the failure's line takes it
_ROOMS: dict[int, dict[str, int]] = {}
_COUNTERS = (
    "allocated_bytes", "allocated_peak", "allocated_capacity", "mapped_bytes", "mapped_peak",
    "pending_bytes", "free_bytes", "reclaimable_bytes", "unassigned_bytes", "pools_bytes",
    "cache_mapped_bytes", "weights_resident", "driver_reservation", "ooms",
)  # fmt: skip


def refused_room(device: int, take: bool = False) -> dict[str, int] | None:
    """`vt.refusal`, kept: the dump reads it and the failure's line still takes it."""
    try:
        latest = vt.refusal(device)
    except Exception:
        latest = None
    if latest is not None:
        _ROOMS[device] = {k: int(v) for k, v in latest.items()}  # type: ignore[call-overload]
    return _ROOMS.pop(device, None) if take else _ROOMS.get(device)


def _counters(device: int) -> dict[str, Any]:
    try:
        stats, room = vt.stats(device), vt.room(device)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}
    out: dict[str, Any] = {k: int(stats[k]) for k in _COUNTERS if k in stats}  # type: ignore[literal-required]
    out.update({f"room_{k}": int(v) for k, v in room.items()})  # type: ignore[call-overload]
    return out


def mark(device: int, what: str) -> None:
    """A cut site begins: a refusal inside it says the room it began with."""
    try:
        room = {k: int(v) for k, v in vt.room(device).items()}  # type: ignore[call-overload]
    except Exception:
        room = None
    _AT[:] = [what[-160:], time.time(), room]


def block(name: str) -> None:
    """A named block's forward begins."""
    _BLOCK[:] = [name[-160:], time.time()]


def event(kind: str, site: str, **fields: Any) -> None:
    EVENTS.append({"t": round(time.time(), 3), "e": kind, "site": site[-160:], **fields})


def capture(exc: BaseException, where: str, device: int = 0, **fields: Any) -> None:
    """Record what `exc`, a refusal the execution layer may answer, found, once."""
    if getattr(exc, _SEEN, False):
        return
    try:
        setattr(exc, _SEEN, True)
    except Exception:
        pass
    event("refused", where, error=str(exc)[:120])
    if len(DUMPS) >= _MOST:
        return
    try:
        DUMPS.append(_dump(exc, where, device, fields))
    except Exception as failed:  # the record only describes the refusal
        DUMPS.append({"where": where, "dump_error": f"{type(failed).__name__}: {failed}"[:300]})


def final(exc: BaseException, device: int = 0) -> None:
    """The refusal that ended the call, before its frames go: every failed frame's tensors."""
    try:
        dump = _dump(exc, "final", device, {})
    except Exception as failed:
        dump = {"where": "final", "dump_error": f"{type(failed).__name__}: {failed}"[:300]}
    dump["final"] = True
    if len(DUMPS) >= _MOST + 1:
        del DUMPS[0]
    DUMPS.append(dump)


def take() -> list[dict[str, Any]]:
    out = list(DUMPS)
    DUMPS.clear()
    return out


def say(dumps: list[dict[str, Any]]) -> None:
    """The dumps on stderr, a row a line, which the machine's log keeps."""
    lists = ("frames", "tensors", "unnamed", "events", "weights")
    for n, dump in enumerate(dumps):
        head = {k: v for k, v in dump.items() if k not in lists}
        _line(n, "head", head)
        if not dump.get("final") and n >= 2:
            continue
        for key in lists:
            for row in dump.get(key) or []:
                _line(n, key, row)


def _line(n: int, key: str, row: object) -> None:
    try:
        text = json.dumps(row, default=str, separators=(",", ":"))
    except Exception as exc:
        text = f"unsaid ({type(exc).__name__})"
    print(f"cozy-var030 #{n} {key} {text[:3000]}", file=sys.stderr, flush=True)


def _request() -> object:
    try:
        from cozy_runtime.internal.execution import catch

        return catch.CURRENT.get()
    except Exception:
        return None


def _named() -> str | None:
    try:
        from cozy_runtime.internal.execution import catch

        return catch._local.named
    except Exception:
        return None


def _dump(exc: BaseException, where: str, device: int, fields: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    request = _request()
    refused = refused_room(device)
    counters = _counters(device)
    asked = re.search(r"allocating (\d+) bytes", str(exc))
    frames, held = _frames(exc)
    regions, weights = _weights(request)
    named, unnamed, totals = _live(held, regions)
    live_then = refused.get("working_live") if refused else None
    live_now = counters.get("room_working_live")
    return {
        "where": where[-200:],
        "final": False,
        "error": f"{type(exc).__name__}: {exc}"[:240],
        "requested": int(asked[1]) if asked else None,
        "block": _named(),
        "last_block": [_BLOCK[0], round(now - _BLOCK[1], 3)],
        "site": {"name": _AT[0], "seconds": round(now - _AT[1], 3), "room_then": _AT[2]},
        "refused_room": refused,
        "counters_now": counters,
        # what the failed op's own evaluation held and its unwinding freed before this was read
        "freed_since_refusal": (
            live_then - live_now if live_then is not None and live_now is not None else None
        ),
        "running": sorted(str(c) for c in getattr(request, "current", ())),
        "hosted": dict(getattr(request, "hosted", {})),
        "retries": getattr(request, "retries", None),
        "totals": totals,
        "frames": frames,
        "tensors": named,
        "unnamed": unnamed,
        "weights": weights,
        "events": list(EVENTS)[-60:],
        **fields,
    }


def _storage(tensor: Tensor) -> tuple[int, int] | None:
    try:
        if not tensor.is_cuda:
            return None
        storage = tensor.untyped_storage()
        return int(storage.data_ptr()), int(storage.nbytes())
    except Exception:
        return None


def _found(value: object, name: str, depth: int, out: list[tuple[str, Tensor]]) -> None:
    """The tensors `value` holds: itself, a sequence's or mapping's, a plain object's fields."""
    if len(out) > 800:
        return
    if isinstance(value, Tensor):
        out.append((name, value))
        return
    if not depth or isinstance(value, nn.Module | str | bytes | int | float | type):
        return
    if isinstance(value, list | tuple | set | frozenset | deque):
        if len(value) <= 256:
            for i, item in enumerate(value):
                _found(item, f"{name}[{i}]", depth - 1, out)
        return
    if isinstance(value, dict):
        if len(value) <= 256:
            for key, item in list(value.items()):
                _found(item, f"{name}[{str(key)[:40]}]", depth - 1, out)
        return
    fields: dict[str, object] = {}
    try:
        fields.update(vars(value))
    except TypeError:
        pass
    for kind in type(value).__mro__[:4]:
        for slot in vars(kind).get("__slots__", ()) or ():
            if isinstance(slot, str):
                try:
                    fields[slot] = getattr(value, slot)
                except Exception:
                    pass
    if hasattr(value, "func"):  # a functools.partial
        for key in ("args", "keywords"):
            if hasattr(value, key):
                fields[key] = getattr(value, key)
    closure = getattr(value, "__closure__", None)
    if closure:
        names = getattr(getattr(value, "__code__", None), "co_freevars", ())
        for free, cell in zip(names, closure, strict=False):
            try:
                fields[f"<{free}>"] = cell.cell_contents
            except ValueError:
                pass
    if len(fields) <= 64:
        for key, item in fields.items():
            _found(item, f"{name}.{key}", depth - 1, out)


def _frames(exc: BaseException) -> tuple[list[str], dict[int, dict[str, Any]]]:
    """The failure's frames, outermost first, and the CUDA storages the failed ones' variables
    hold. Each traceback's first frame caught the failure and still runs: it is not read."""
    frames: list[str] = []
    held: dict[int, dict[str, Any]] = {}
    chain: list[BaseException] = []
    link: BaseException | None = exc
    while link is not None and link not in chain:
        chain.append(link)
        link = link.__cause__ or link.__context__
    for link in reversed(chain):
        tb, first = link.__traceback__, True
        while tb is not None:
            frame, code = tb.tb_frame, tb.tb_frame.f_code
            path = code.co_filename.rsplit("site-packages/", 1)[-1].rsplit("/source/", 1)[-1]
            frames.append(f"{path}:{tb.tb_lineno} {getattr(code, 'co_qualname', code.co_name)}")
            found: list[tuple[str, Tensor]] = []
            if not first and code.co_flags & 0x1 and frame.f_locals is not frame.f_globals:
                try:
                    for name, value in list(frame.f_locals.items()):
                        _found(value, name, 3, found)
                except Exception:
                    pass
            for name, tensor in found:
                storage = _storage(tensor)
                if storage is None:
                    continue
                row = held.setdefault(storage[0], _row(tensor, storage))
                if len(row["held_by"]) < 6:
                    row["held_by"].append(f"{len(frames) - 1}:{name}")
            del found
            tb, first = tb.tb_next, False
    return frames[-48:], held


def _row(tensor: Tensor, storage: tuple[int, int]) -> dict[str, Any]:
    return {
        "bytes": storage[1],
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "weight": isinstance(tensor, nn.Parameter),
        "held_by": [],
    }


def _live(
    held: dict[int, dict[str, Any]], regions: set[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Every live CUDA storage: those the failed frames name, the largest of the others (with
    the kinds of object that refer to each), and the totals; weights apart (a module's parameters
    and buffers, and whatever shares a weight region's storage)."""
    weights: set[int] = set()
    tensors: list[Tensor] = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, nn.Module):
                for table in (obj._parameters, obj._buffers):
                    weights.update(id(t) for t in table.values() if t is not None)
            elif isinstance(obj, Tensor) and obj.is_cuda:
                tensors.append(obj)
        except Exception:
            continue
    others: dict[int, dict[str, Any]] = {}
    weight_bytes: dict[int, int] = {}
    keep: dict[int, Tensor] = {}
    for tensor in tensors:
        storage = _storage(tensor)
        if storage is None:
            continue
        ptr = storage[0]
        if ptr in regions or id(tensor) in weights or isinstance(tensor, nn.Parameter):
            weight_bytes[ptr] = storage[1]
            continue
        if ptr in held or ptr in others:
            continue
        row = _row(tensor, storage)
        row.pop("held_by")
        others[ptr] = row
        keep[ptr] = tensor
    del tensors
    for ptr in weight_bytes:
        others.pop(ptr, None)
        keep.pop(ptr, None)
        if ptr in held:
            held[ptr]["weight"] = True
    order = sorted(others, key=lambda p: -others[p]["bytes"])
    for ptr in order[:40]:
        others[ptr]["referrers"] = _referrers(keep[ptr])
    keep.clear()
    unnamed = [others[p] for p in order]
    named = sorted(held.values(), key=lambda r: -r["bytes"])
    working = [r for r in named if not r["weight"]]
    totals = {
        "named_working_bytes": sum(r["bytes"] for r in working),
        "named_working": len(working),
        "unnamed_working_bytes": sum(r["bytes"] for r in unnamed),
        "unnamed_working": len(unnamed),
        "weight_tensor_bytes": sum(weight_bytes.values()),
        "weight_tensors": len(weight_bytes),
    }
    return named[:120], unnamed[:60], totals


def _referrers(tensor: Tensor) -> list[str]:
    """The kinds of object that hold `tensor`; a running frame is named, never read."""
    out: list[str] = []
    try:
        for ref in gc.get_referrers(tensor)[:10]:
            kind = type(ref).__name__
            if kind == "frame":
                code = ref.f_code
                out.append(f"frame {code.co_filename.rsplit('/', 1)[-1]}:{code.co_name}")
            elif isinstance(ref, dict):
                keys = [str(k)[:40] for k, v in list(ref.items())[:400] if v is tensor][:3]
                out.append(f"dict{keys}")
            elif isinstance(ref, list | tuple):
                out.append(f"{kind}[{len(ref)}]")
            elif kind != "cell":
                out.append(kind)
            else:
                out.append("cell (a closure's variable)")
    except Exception as exc:
        out.append(f"unread ({type(exc).__name__})")
    return out[:6]


def _weights(request: object) -> tuple[set[int], list[dict[str, Any]]]:
    """The weight regions' storages, and the regions resident now by component and block,
    largest first, under a first row of their totals."""
    memory = getattr(request, "memory", None)
    current = getattr(request, "current", set())
    regions: set[int] = set()
    rows: list[dict[str, Any]] = []
    try:
        for model in list(getattr(memory, "models", {}).values()):
            names = {id(b): name for name, b in getattr(model, "blocks", {}).items()}
            for group in model.groups():
                for storage in (group.native_storage, group.final_storage):
                    if storage is not None:
                        regions.add(int(storage))
                for kind in ("region", "final_region"):
                    region = getattr(group, kind, None)
                    if region is not None and region.resident:
                        rows.append({
                            "model": str(model.component),
                            "block": str(names.get(id(group), "common"))[-80:],
                            "kind": kind,
                            "bytes": int(region.allocated_bytes),
                            "running": model.name in current,
                        })  # fmt: skip
    except Exception as exc:
        rows.append({"error": f"{type(exc).__name__}: {exc}"[:200], "bytes": 0})
    rows.sort(key=lambda r: -r["bytes"])
    by_model: dict[str, int] = {}
    for row in rows:
        if "model" in row:
            by_model[row["model"]] = by_model.get(row["model"], 0) + row["bytes"]
    total = {"resident_regions": len(rows), "resident_bytes": sum(by_model.values()), **by_model}
    return regions, [total, *rows[:60]]
