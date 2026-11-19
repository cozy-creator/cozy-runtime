"""What a call's K/V and state need, as the engine's K/V manager declares them (`engine.kv`, cr-336).

The rung judge that reads it (a step's graph, eager or short of one step, and how much of the
declared K/V the card holds, from the step's measured first run) comes with its caller, the
decoder's fit (cozy-runtime #2301; its code is on `claude/xs-213-rungs-judge`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KvNeed:
    """A call's K/V and state as its manager declared them (`engine.kv`): the exact upper bound
    of the call, and of that, what a step writes (running rows' tail blocks and state slots),
    which never leaves the card."""

    declared: int
    written: int
