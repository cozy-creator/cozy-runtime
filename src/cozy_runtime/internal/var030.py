"""PROOF ONLY (var-030), never merge: where a request's time goes on the device and on the host.

A mark is a CUDA timing event recorded on the current stream with a label. The device time from one
mark to the next belongs to the first one's label, so the marks partition the request's wall time
as the device saw it, with no synchronize added; the host's clock between the same two marks is
kept beside it (a label with host time and no device time is the host waiting on work queued
earlier). A span labels everything inside it; `at` labels a part of the innermost span. Each cut
call keeps a row: its site, level, windows, the bytes its windows read from and wrote to a host
whole, and what varena's counters moved by while it ran. `take` reads the events once the request
has ended: sums by (stage, step, label), the rows grouped by (step, site, level), each selection's
figures, the refusals, and a reading of every varena counter at each step.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

import torch

_LIMIT = 900_000
_DIGITS = re.compile(r"\d+")
#: varena counters a cut call's row keeps the change of
_KEEP = ("event_waits", "context_joins", "ooms", "allocations", "driver_memory_ns", "weights_filled")
_PARTS = ("w", "read", "write")


class _State:
    def __init__(self) -> None:
        self.thread: int | None = None
        self.open = False
        #: (event, label id, call index, step index, host perf_counter)
        self.marks: list[tuple[torch.cuda.Event, int, int, int, float]] = []
        self.names: list[str] = []
        self.ids: dict[str, int] = {}
        self.parts: dict[tuple[int, str], int] = {}
        self.stack: list[int] = []
        self.calls: list[dict[str, Any]] = []
        self.call_stack: list[int] = []
        self.counts: dict[str, list[float]] = {}
        self.steps: list[dict[str, Any]] = []
        self.step_keys: dict[tuple[str, int], int] = {}
        self.step = -1
        self.selections: dict[tuple[int, str, int, int], list[int]] = {}
        self.events: list[list[Any]] = []
        self.dropped = 0
        self.error = ""
        self.began = time.time()


_S = _State()
_LAST: dict[str, Any] | None = None


def generic(name: str) -> str:
    """A block path's sites share a label; a split's key (its shape is in it) stays as it is."""
    if name.startswith(("split:", "windows:")):
        return name
    return _DIGITS.sub("N", name)


def begin() -> None:
    """A request begins: its marks start over."""
    global _S
    if _S.open:
        return
    _S = _State()
    _S.open = True


def _ok() -> bool:
    s = _S
    if not s.open or s.error or not torch.cuda.is_initialized():
        return False
    ident = threading.get_ident()
    if s.thread is None:
        s.thread = ident
    if s.thread != ident:
        return False
    return not torch.cuda.is_current_stream_capturing()


def _id(name: str) -> int:
    found = _S.ids.get(name)
    if found is None:
        found = _S.ids[name] = len(_S.names)
        _S.names.append(name)
    return found


def _now() -> tuple[str, int]:
    """The stage and step the request is in (`author._services`)."""
    from cozy_runtime.author import _services

    stage = _services._stage.get()
    name = str(stage[1]) if stage is not None and stage[1] is not None else ""
    step = _services.STEP_NOW.get()
    return name, (int(step[1]) if step is not None and step[0] == name else 0)


def _device() -> tuple[Any, int] | None:
    from cozy_runtime.internal.execution import catch

    request = catch.CURRENT.get()
    if request is None or not request.memory.cuda:
        return None
    return request, request.memory.device.index or 0


def _reading() -> dict[str, Any]:
    """varena's counters now: the working allocator's, the room, and the weights' residency."""
    out: dict[str, Any] = {"unix_ms": round(time.time() * 1000, 1)}
    found = _device()
    if found is None:
        return out
    import tensorfs.varena.torch as vt

    request, device = found
    out["working"] = dict(vt.stats(device))
    out["room"] = dict(vt.room(device))
    native = request.memory.native
    if native is not None:
        weights = dict(native.stats())
        out["weights"] = {k: v for k, v in weights.items() if isinstance(v, int | float | bool)}
    out["moved"] = {k: list(v) for k, v in request.moved.items()}
    out["retries"] = request.retries
    return out


def _counters() -> dict[str, int]:
    """The varena counters a cut call's row keeps the change of."""
    found = _device()
    if found is None:
        return {}
    import tensorfs.varena.torch as vt

    request, device = found
    stats = vt.stats(device)
    out = {k: int(v) for k, v in stats.items() if k.startswith(("reclaim_", "wait_")) or k in _KEEP}
    out.pop("reclaim_longest_ns", None)
    out.pop("reclaim_longest_step", None)
    native = request.memory.native
    if native is not None:
        weights = native.stats()
        for key in ("evictions", "fills", "hits", "reclaimed_bytes"):
            out[f"w_{key}"] = int(weights.get(key, 0))
    return out


def _step_index() -> int:
    s = _S
    key = _now()
    step = s.step_keys.get(key)
    if step is None:
        step = s.step_keys[key] = len(s.steps)
        s.steps.append({"stage": key[0], "step": key[1], **_reading()})
    s.step = step
    return step


def _mark(label: int) -> None:
    s = _S
    if len(s.marks) >= _LIMIT:
        s.dropped += 1
        return
    try:
        step = _step_index()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        s.marks.append(
            (event, label, s.call_stack[-1] if s.call_stack else -1, step, time.perf_counter())
        )
    except Exception as exc:  # the proof never fails a run
        s.error = f"{type(exc).__name__}: {exc}"


class span:
    """Everything inside is `label`'s, less what an inner span or `at` takes."""

    __slots__ = ("label", "on")

    def __init__(self, label: str) -> None:
        self.label, self.on = label, False

    def __enter__(self) -> span:
        if _ok():
            self.on = True
            _S.stack.append(_id(self.label))
            _mark(_S.stack[-1])
        return self

    def __exit__(self, *exc: object) -> None:
        if self.on:
            s = _S
            if s.stack:
                s.stack.pop()
            if _ok():
                _mark(s.stack[-1] if s.stack else _id("outside"))


def at(part: str) -> None:
    """From here the innermost span's time is its `part`'s."""
    if not _ok() or not _S.stack:
        return
    s = _S
    base = s.stack[-1]
    name = s.names[base]
    if " | " in name:  # a part's own span: its base is the span around it
        return
    found = s.parts.get((base, part))
    if found is None:
        found = s.parts[(base, part)] = _id(f"{name} | {part}")
    _mark(found)


def count(name: str, nbytes: int = 0, seconds: float = 0.0) -> None:
    """One more `name`, with its bytes and host seconds."""
    s = _S
    if not s.open:
        return
    row = s.counts.setdefault(name, [0, 0, 0.0])
    row[0] += 1
    row[1] += nbytes
    row[2] += seconds


def event(kind: str, site: str, level: int = -1, detail: str = "") -> None:
    """A refusal, a retry or a host decision, with when and in which step."""
    s = _S
    if not s.open or len(s.events) >= 4000:
        return
    try:
        step = _step_index() if _ok() else s.step
    except Exception:
        step = s.step
    s.events.append([round(time.time() * 1000, 1), step, kind, generic(site), level, detail[:300]])
    count(f"event:{kind}")


def selection(site: str, level: int, chosen: Any = None, usable: int = -1) -> None:
    """A cut's level and what it was chosen on: native selection's figures, or (nested) the room."""
    if not _ok():
        return
    s = _S
    try:
        key = (_step_index(), generic(site), level, 0 if chosen is not None else 1)
        row = s.selections.get(key)
        if chosen is not None:
            avail, need = int(chosen["available_bytes"]), int(chosen["required_bytes"])
            budget, live = int(chosen["budget_bytes"]), int(chosen["live_bytes"])
            full, cap = int(chosen["allocation_backing_bytes"]), int(chosen["capacity_bytes"])
        else:
            avail, need, budget, live, full, cap = usable, 0, 0, 0, 0, 0
        if row is None:
            s.selections[key] = [1, avail, avail, avail, need, need, budget, live, full, cap]
        else:
            row[0] += 1
            row[1] += avail
            row[2] = min(row[2], avail)
            row[3] = max(row[3], avail)
            row[4] += need
            row[5] = max(row[5], need)
            row[6] = min(row[6], budget)
            row[7] = max(row[7], live)
            row[8] = max(row[8], full)
    except Exception as exc:
        s.error = f"selection {type(exc).__name__}: {exc}"


class call:
    """One cut call's row and span: its site and level, what its windows moved, and what varena's
    counters moved by while it ran (a leaf's; a nested one keeps no counters)."""

    __slots__ = ("nested", "on", "row", "span")

    def __init__(self, site: str, level: int, nested: bool = False) -> None:
        self.on, self.nested = False, nested
        self.row: dict[str, Any] = {"site": generic(site), "level": level, "nested": int(nested)}
        self.span: span | None = None

    def __enter__(self) -> call:
        if not _ok():
            return self
        s = _S
        try:
            label = self.row["site"]
            if self.nested and s.stack:
                label = f"{s.names[s.stack[-1]].split(' | ')[0]} > {label}"
            self.row["i0"] = len(s.marks)
            self.row["t0"] = time.perf_counter()
            self.row["unix_ms"] = round(time.time() * 1000, 1)
            self.row["parent"] = s.call_stack[-1] if s.call_stack else -1
            if not self.nested:
                self.row["before"] = _counters()
            s.calls.append(self.row)
            s.call_stack.append(len(s.calls) - 1)
            self.on = True
            self.span = span(label)
            self.span.__enter__()
            self.row["depth"] = len(s.stack)
        except Exception as exc:
            s.error = f"call {type(exc).__name__}: {exc}"
        return self

    def __exit__(self, kind: object, *exc: object) -> None:
        if not self.on:
            return
        s = _S
        try:
            row = self.row
            row["host_ms"] = (time.perf_counter() - row.pop("t0")) * 1000
            row["step"] = s.step
            if kind is not None:
                row["raised"] = str(getattr(kind, "__name__", kind))
                refused = _id(f"refused attempt: {row['site']}")
                for i in range(row["i0"], len(s.marks)):
                    mark = s.marks[i]
                    s.marks[i] = (mark[0], refused, mark[2], mark[3], mark[4])
                event("raised", row["site"], row["level"], row["raised"])
            before = row.pop("before", None)
            if before is not None:
                after = _counters()
                row["delta"] = {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}
            if s.call_stack:
                s.call_stack.pop()
            if self.span is not None:
                self.span.__exit__(None, None, None)
        except Exception as failed:
            s.error = f"call exit {type(failed).__name__}: {failed}"


class kernel:
    """One compiled kernel's launch, labelled with which of its tensor arguments are windows of a
    host whole (their positions among its tensors): those it reads or writes over the link."""

    __slots__ = ("name", "span", "tensors")

    def __init__(self, name: str, tensors: Any) -> None:
        self.name, self.tensors, self.span = name, tensors, None

    def __enter__(self) -> kernel:
        if not _ok() or not _S.stack:
            return self
        s = _S
        try:
            from cozy_runtime.internal.execution import pinned

            host = [i for i, t in enumerate(self.tensors) if pinned.held(t)]
            nbytes = sum(int(self.tensors[i].numel()) * int(self.tensors[i].element_size()) for i in host)
            label = f"{s.names[s.stack[-1]]} > kernel {self.name}"
            if host:
                label += f" [host {','.join(map(str, host))}]"
                moved("kernel_host_bytes", nbytes)
            self.span = span(label)
            self.span.__enter__()
        except Exception as exc:
            s.error = f"kernel {type(exc).__name__}: {exc}"
        return self

    def __exit__(self, *exc: object) -> None:
        if self.span is not None:
            self.span.__exit__(None, None, None)


def piece() -> None:
    """A window begins (`catch.checkpoint`)."""
    if not _ok() or not _S.stack:
        return
    s = _S
    if s.call_stack:
        row = s.calls[s.call_stack[-1]]
        if row.get("depth") == len(s.stack):
            row["windows"] = row.get("windows", 0) + 1
    at("w")


def moved(kind: str, nbytes: int) -> None:
    """A window's bytes read from or written to its whole: `read_host`, `read_host_strided`,
    `wrote_host`, `wrote_host_strided`, `wrote_card`, `inplace_host`, `inplace_card`."""
    s = _S
    if s.open and s.call_stack and threading.get_ident() == s.thread:
        row = s.calls[s.call_stack[-1]]
        row[kind] = row.get(kind, 0) + nbytes
    count(kind, nbytes)


def inplace(window: Any) -> None:
    """A window a split's op writes in place (`split(out=)`): on a host whole, or on the card."""
    try:
        from cozy_runtime.internal.execution import pinned

        nbytes = int(window.numel()) * int(window.element_size())
        moved("inplace_host" if pinned.held(window) else "inplace_card", nbytes)
    except Exception:
        pass


_SUMS = (
    "windows", "read_host", "read_host_strided", "wrote_host", "wrote_host_strided", "wrote_card",
    "inplace_host", "inplace_card", "kernel_host_bytes", "host_ms",
)  # fmt: skip
_ROW = (
    "windows", "read_host", "read_host_strided", "wrote_host", "wrote_host_strided", "wrote_card",
    "inplace_host", "kernel_host_bytes",
)  # fmt: skip
_DEVICE = ("span", "w", "read", "write", "kernel", "kernel_host", "inner")


def take(request_id: str = "") -> dict[str, Any] | None:
    """The request's sums, read once the device has passed every mark."""
    global _LAST
    s = _S
    if not s.open:
        return _LAST
    s.open = False
    out: dict[str, Any] = {"marks": len(s.marks), "dropped": s.dropped, "error": s.error}
    try:
        n = len(s.marks)
        end = torch.cuda.Event(enable_timing=True)
        if n:
            torch.cuda.synchronize()
            end.record()
            end.synchronize()
        end_host = time.perf_counter()
        by: dict[tuple[int, int], list[float]] = {}
        per_call: dict[int, dict[str, float]] = {}
        for i in range(n):
            mark, label, at_call, step, host = s.marks[i]
            after, host_after = (s.marks[i + 1][0], s.marks[i + 1][4]) if i + 1 < n else (end, end_host)
            ms = float(mark.elapsed_time(after))
            row = by.setdefault((step, label), [0.0, 0.0, 0])
            row[0] += ms
            row[1] += (host_after - host) * 1000
            row[2] += 1
            if at_call >= 0:
                name = s.names[label]
                if " > kernel " in name:
                    part = "kernel_host" if "[host " in name else "kernel"
                else:
                    part = name.rsplit(" | ", 1)[-1] if " | " in name else "span"
                sums = per_call.setdefault(at_call, {})
                sums[part] = sums.get(part, 0.0) + ms
                sums["host"] = sums.get("host", 0.0) + (host_after - host) * 1000
        groups: dict[tuple[int, str, int, int, str], dict[str, Any]] = {}
        for index, row in enumerate(s.calls):
            key = (row.get("step", -1), row["site"], row["level"], row["nested"], row.get("raised", ""))
            group = groups.setdefault(key, {"n": 0, "device_ms": {}, "delta": {}})
            group["n"] += 1
            for name in _SUMS:
                if row.get(name):
                    group[name] = group.get(name, 0) + row[name]
            for part, ms in per_call.get(index, {}).items():
                group["device_ms"][part] = group["device_ms"].get(part, 0.0) + ms
            for name, value in (row.get("delta") or {}).items():
                group["delta"][name] = group["delta"].get(name, 0) + value
        # Each leaf call's own row, its nested calls' device time folded in as `inner`.
        inner: dict[int, float] = {}
        for index, row in enumerate(s.calls):
            if row["nested"]:
                parent = row.get("parent", -1)
                while parent >= 0 and s.calls[parent]["nested"]:
                    parent = s.calls[parent].get("parent", -1)
                if parent >= 0:
                    mine = per_call.get(index, {})
                    inner[parent] = inner.get(parent, 0.0) + sum(v for k, v in mine.items() if k != "host")
        rows = []
        for index, row in enumerate(s.calls):
            if row["nested"] or len(rows) >= 6000:
                continue
            mine = dict(per_call.get(index, {}))
            mine["inner"] = inner.get(index, 0.0)
            delta = row.get("delta") or {}
            rows.append(
                [
                    row.get("step", -1),
                    _id(row["site"]),
                    row["level"],
                    1 if row.get("raised") else 0,
                    *[row.get(name, 0) for name in _ROW],
                    round(row.get("host_ms", 0.0), 1),
                    *[round(mine.get(part, 0.0), 1) for part in _DEVICE],
                    round(delta.get("reclaim_ns", 0) / 1e6, 1),
                    round(delta.get("reclaim_weights_ns", 0) / 1e6, 1),
                    round(delta.get("reclaim_pending_working_ns", 0) / 1e6, 1),
                    delta.get("w_evictions", 0),
                    delta.get("w_fills", 0),
                    row.get("unix_ms", 0),
                ]
            )
        out["rows_columns"] = (
            "step site(label id) level raised " + " ".join(_ROW) + " host_ms "
            + " ".join(f"device_{p}_ms" for p in _DEVICE)
            + " reclaim_ms reclaim_weights_ms reclaim_pending_ms evictions fills unix_ms"
        )
        out["rows"] = rows
        out["labels"] = s.names
        out["steps"] = s.steps
        out["by_columns"] = "step label device_ms host_ms marks"
        out["by"] = [[k[0], k[1], round(v[0], 2), round(v[1], 2), v[2]] for k, v in by.items()]
        out["calls_columns"] = "step site level nested raised {n, sums, device_ms by part, delta}"
        out["calls"] = [
            [
                *key,
                {
                    **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in g.items() if k not in ("device_ms", "delta")},
                    "device_ms": {k: round(v, 2) for k, v in g["device_ms"].items()},
                    "delta": g["delta"],
                },
            ]
            for key, g in groups.items()
        ]
        out["selections_columns"] = (
            "step site level nested: n available_sum available_min available_max required_sum "
            "required_max budget_min live_max full_max capacity"
        )
        out["selections"] = [[*key, *row] for key, row in s.selections.items()]
        out["events_columns"] = "unix_ms step kind site level detail"
        out["events"] = s.events
        out["counts"] = {k: [int(v[0]), int(v[1]), round(v[2], 4)] for k, v in s.counts.items()}
        out["end"] = _reading()
        out["began_unix_ms"] = round(s.began * 1000, 1)
    except Exception as exc:
        out["error"] = f"{out['error']} take: {type(exc).__name__}: {exc}"
    s.marks = []
    _LAST = out
    try:  # beside the record: a file the session fetches
        os.makedirs("/var/tmp/var030", exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id)[:40]
        with open(f"/var/tmp/var030/{int(time.time() * 1000)}-{name}.json", "w") as handle:
            json.dump(out, handle, separators=(",", ":"))
    except Exception:
        pass
    return out
