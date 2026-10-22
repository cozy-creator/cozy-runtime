"""The machine's kept-layout reply over a real executor seam; bytes come from TensorFS."""

from __future__ import annotations

import os
from collections.abc import Mapping

from cozy_runtime.internal.seam import Channel


def answer_read_grants(
    channel: Channel,
    deposited: Mapping[str, int],
    asked: list[str],
    sealing: dict[str, int] | None = None,
    admitted: frozenset[str] = frozenset(),
) -> None:
    """`read_grant` as TensorD answers it: the layout deposited for it (`deposited`: layout ->
    sealed memfd) as the one descriptor, else none held, sealing it the asker's where `admitted`.
    A layout another call seals answers `layout_sealing` as many times as `sealing` says. `asked`
    records each layout."""
    sealing = {} if sealing is None else sealing
    while (frame := channel.recv()) is not None:
        if frame.get("event") != "request" or frame.get("kind") != "read_grant":
            continue
        layout = str(frame["layout"])
        asked.append(layout)
        answer = {"event": "answer", "seq": frame["seq"], "ok": True}
        if sealing.get(layout, 0) > 0:
            sealing[layout] -= 1
            channel.send(answer | {"ok": False, "code": "layout_sealing", "held": False})
            continue
        if (held := deposited.get(layout)) is None:
            channel.send(answer | {"held": False} | ({"seal": True} if layout in admitted else {}))
            continue
        fd = os.open(f"/proc/self/fd/{held}", os.O_RDONLY)
        try:
            channel.send(answer | {"held": True, "descriptors": 1})
            channel.send_fds([fd])
        finally:
            os.close(fd)
