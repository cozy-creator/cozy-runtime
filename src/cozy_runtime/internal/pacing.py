"""A call on a GPU that also drives the computer's display pauses for it (cl-796).

The desktop's frames need that GPU too (a laptop's external monitor copies through it every
frame), and a saturating call starves them. When the machine says the session's GPU drives a
display (Start's `display_pace`), each block entry and eager step end is a pause point: once
`work_ms` has passed since the last pause, the call waits for its GPU and sleeps `pause_ms`
(a 60 Hz frame by default). A GPU-bound call keeps about work / (work + pause) of its speed.
Never inside a graph capture; a native resident program runs between pause points unpaced.
"""

from __future__ import annotations

import time
from typing import Any

import torch

_work = 0.0
_pause = 0.0
_since = 0.0
#: Pauses taken in this process.
pauses = 0


def configure(pace: Any) -> bool:
    """This process's pace: Start's `display_pace`, None (or a zero pause) for none."""
    global _work, _pause, _since
    on = pace is not None and pace.pause_ms > 0
    _work, _pause = (pace.work_ms / 1e3, pace.pause_ms / 1e3) if on else (0.0, 0.0)
    _since = time.perf_counter()
    return _pause > 0


def tick() -> None:
    """A pause point: wait for the GPU and pause when `work_ms` has passed since the last one."""
    global _since, pauses
    if not _pause or time.perf_counter() - _since < _work:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    torch.cuda.synchronize()
    time.sleep(_pause)
    pauses += 1
    _since = time.perf_counter()
