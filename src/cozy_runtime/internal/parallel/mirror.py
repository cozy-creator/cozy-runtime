"""The mirrored model call: how a handler on rank 0 drives K replicas.

The package's handler runs on rank 0 alone; a follower runs nothing of it. What a follower
runs is every OUTERMOST component-use method the handler enters, with the same arguments,
on its own replica - the runtime installs this hook on the model beside the residency
plane (`Model._cozy_group`), and the author cannot observe it. Inside that method every
rank's forward issues the same collectives in the same order, which is what makes the
Ulysses all-to-all pair up. A nested component-use call is part of the outer method's
body on every rank already and is not mirrored again.

Rank 0's failure inside a mirrored call breaks the group (the followers may be parked on a
collective rank 0 will never join); a follower's failure or death surfaces on rank 0 when
the outer call closes, typed, and poisons the executor. In both cases the worker rebuilds
the whole cgroup: a group is repaired by replacement, never patched.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from cozy_runtime.internal.parallel import wire
from cozy_runtime.internal.parallel.cp import UngatedShardedForward, gated_call, in_gated_call
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.parallel.plan import GroupRefusal


class LeaderMirror:
    """Rank 0's side: broadcast the call, hold the gate, collect the replicas' returns."""

    def __init__(self, group: RankGroup, model: str, spool: Callable[[], Path | None]) -> None:
        self.group = group
        self.model = model
        self.spool = spool
        self._gate = gated_call()
        self._outer = False
        self._payloads: wire.TensorSpool | None = None

    def enter(self, method: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        if in_gated_call():
            return
        spool = self.spool()
        if spool is None:
            raise GroupRefusal(
                f"{method}: a mirrored call outside an attempt has no spool for its "
                "payloads; component-use methods run inside invoke only under a group",
                code="ungated_sharded_forward",
            )
        self.group.check_alive()
        self._payloads = wire.TensorSpool(spool / "ranks")
        command = wire.run_command(method, args, kwargs, self._payloads)
        self.group.broadcast({**command, "model": self.model})
        self._outer = True
        self._gate.__enter__()

    def leave(self, method: str, failed: bool) -> None:
        if not self._outer:
            return
        self._outer = False
        self._gate.__exit__(None, None, None)
        try:
            if failed:
                self.group._break(f"rank 0 failed inside the mirrored call {method!r}")
                return
            for reply in self.group.collect("run"):
                if not reply.get("ok"):
                    raise GroupRefusal(
                        f"{method}: rank {reply.get('rank')} failed: {reply.get('code')}: "
                        f"{reply.get('detail', '')}"[:600],
                        code=str(reply.get("code") or "group_broken"),
                    )
        finally:
            if self._payloads is not None:
                self._payloads.clear()
                self._payloads = None


class FollowerMirror:
    """A follower's side: a component-use method runs ONLY as a mirrored call."""

    def enter(self, method: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        if not in_gated_call():
            raise UngatedShardedForward(
                f"{method} entered on a follower rank outside a mirrored call: nothing on "
                "rank 0 is in this collective"
            )

    def leave(self, method: str, failed: bool) -> None:
        return None
