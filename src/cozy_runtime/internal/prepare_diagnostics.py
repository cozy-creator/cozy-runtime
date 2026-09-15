"""Bounded preparation diagnostics, carried by the existing reply and activity log.

Exception messages already have a separate refusal field. This additional trace
contains only exception types and frame locations: no locals, source lines,
exception messages/notes, absolute directories, or environment dump.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

from cozy_runtime.internal import proctree
from cozy_runtime.internal.config import inherited_environment

MAX_TRACE_LINES = 40
MAX_LINE_CHARS = 160
STARTUP_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "NCCL_NVLS_ENABLE",
)


def _printable(value: str) -> str:
    return "".join(c if 32 <= ord(c) <= 126 else "?" for c in value)[:MAX_LINE_CHARS]


def exception_trace(exc: BaseException) -> str:
    """Keep up to four causal exceptions and each one's eight innermost frames."""
    chain: list[list[str]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 4:
        seen.add(id(current))
        frames: deque[str] = deque(maxlen=8)
        tb = current.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            frames.append(
                _printable(f"  {PurePath(code.co_filename).name}:{tb.tb_lineno} in {code.co_name}")
            )
            tb = tb.tb_next
        chain.append([_printable(type(current).__name__), *frames])
        cause = current.__cause__
        current = (
            cause
            if cause is not None
            else current.__context__
            if not current.__suppress_context__
            else None
        )
    lines: list[str] = ["earlier exceptions omitted"] if current is not None else []
    for item in reversed(chain):
        if lines:
            lines.append("followed by:")
        lines.extend(item)
    return "\n".join(lines[:MAX_TRACE_LINES])


def activity_lines(trace: object) -> tuple[str, ...]:
    """Recheck the bound at the worker boundary before retaining executor text."""
    if not isinstance(trace, str):
        return ()
    bounded = trace[: MAX_TRACE_LINES * (MAX_LINE_CHARS + 1)]
    return tuple(_printable(line) for line in bounded.splitlines()[:MAX_TRACE_LINES])


def memory_summary(reply: Mapping[str, Any]) -> str:
    """Absent or sentinel readings are unknown; an explicitly measured zero is valid."""
    parts = []
    for key, label in (
        ("peak_allocator_bytes", "peak allocator"),
        ("allocator_bytes", "allocated"),
        ("device_free_bytes", "device free"),
    ):
        value = reply.get(key)
        measured = f"{value} B" if type(value) is int and value >= 0 else "unknown"
        parts.append(f"{label}: {measured}")
    return "memory at failure: " + ", ".join(parts)


def startup_facts(*, rank: int, world: int, device_kind: str) -> dict[str, Any]:
    """Observe only approved startup facts; never import Torch or initialize CUDA."""
    # Config owns environment reads. Narrow its process snapshot immediately;
    # neither the whole mapping nor unrelated values enter the diagnostic.
    environment = inherited_environment()
    facts: dict[str, Any] = {
        "pid": os.getpid(),
        "rank": rank,
        "world": world,
        "device_kind": device_kind,
        **proctree.own_protection(),
        "environment": {
            name: _printable(environment[name]) if name in environment else None
            for name in STARTUP_ENV
        },
    }
    torch = sys.modules.get("torch")
    facts["cuda_initialized"] = False
    if torch is not None and device_kind == "cuda":
        try:
            facts["cuda_initialized"] = bool(torch.cuda.is_initialized())
            if facts["cuda_initialized"]:
                facts["selected_device"] = int(torch.cuda.current_device())
        except (AttributeError, RuntimeError):
            # Diagnostic reads must not replace the failure being investigated.
            facts["cuda_initialized"] = None
    return facts
