"""Tolerant reading of documents another process, Runtime version or peer wrote.

One member that does not decode costs itself, never the document: `read` drops the deepest
node a validation error names and decodes again. A null takes its member's default anywhere;
any other unreadable node is dropped only inside an advisory member (telemetry, narration).
A load-bearing member (a verdict, a result, a digest) is never guessed, so its error raises.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Collection
from typing import Any

import msgspec

_STEP = re.compile(r"\.([^.\[`]+)|\[(\d+|\.\.\.)\]")


def read[T](doc: object, into: type[T], advisory: Collection[str] = ()) -> tuple[T, list[str]]:
    """`into` decoded from `doc`, and one line per node dropped to get there."""
    try:
        return msgspec.convert(doc, into), []
    except msgspec.ValidationError as first:
        error = first
    work: Any = copy.deepcopy(doc)
    dropped: list[str] = []
    while True:
        message, _, where = str(error).rpartition(" - at `$")
        path: list[str | int] = []
        for key, index in _STEP.findall(where.rstrip("`")):
            if index == "...":
                break  # a dict value names no key: its whole container goes
            path.append(int(index) if index else key)
        parent, node = work, work
        for step in path:
            parent, node = node, _child(node, step)
            if node is _MISSING:
                raise error
        absent = node is None and isinstance(path[-1], str) if path else False
        if not path or not (absent or path[0] in advisory):
            raise error
        del parent[path[-1]]
        dropped.append(f"{'.'.join(map(str, path))}: {message or error}")
        try:
            return msgspec.convert(work, into), dropped
        except msgspec.ValidationError as exc:
            error = exc


_MISSING = object()


def _child(node: Any, step: str | int) -> Any:
    if isinstance(step, int):
        return node[step] if isinstance(node, list) and step < len(node) else _MISSING
    return node.get(step, _MISSING) if isinstance(node, dict) else _MISSING
