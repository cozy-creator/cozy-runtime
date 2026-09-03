"""GroupPlan: rank 0 decides, every rank obeys - the whole doctrine, in one file.

Every adaptive decision that must be identical across a group's ranks is a field here.
Rank 0 authors the plan from ITS resolution and hands it to the followers with the prepare;
a follower resolves the same inputs on its own card and ASSERTS agreement, field by field.
A rank never adapts locally: a group whose ranks would run different plans produces
silently wrong output (a per-tensor activation scale derived from the local shard, a
different resident set, a different route), and the only honest answer is to refuse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.internal import canonical


class GroupRefusal(RuntimeFailure):
    """The group could not form, agree, or stay whole. RUNTIME origin: the request and
    the author are sound; the executor's process group is not. `code` names which."""

    default_code = "group_broken"


class RankDivergence(GroupRefusal):
    """A rank could not honour the group's plan."""

    default_code = "rank_divergence"

    def __init__(self, rank: int, field_name: str, detail: str) -> None:
        super().__init__(f"rank {rank} field {field_name!r}: {detail}", fields=[field_name])
        self.rank = int(rank)
        self.field_name = field_name


def rank_invariant_digest(identity: dict[str, Any]) -> str:
    """A resolved plan's identity with the one rank-LOCAL fact removed: the device INDEX.

    Rank r measures card r of the seal, so `Plan.digest()` differs across the ranks of a
    group by construction; everything else in it - the bytes, every tensor's route, the
    device CLASS, the runtime, the construction contract - must agree, and this is the
    digest that says so.
    """
    device = {k: v for k, v in dict(identity.get("device") or {}).items() if k != "index"}
    return canonical.digest({**identity, "device": device})


@dataclass(frozen=True, slots=True)
class GroupPlan:
    """Every fact the K ranks must agree on before a weight moves."""

    #: the sequence-parallel degree; equals the number of sealed devices
    degree: int
    #: the seal's entries, in order - rank r holds entry r
    devices: tuple[str, ...]
    #: the resolved fill plan's RANK-INVARIANT digest (`rank_invariant_digest`): every
    #: tensor's route, the bytes, the device class - not the card's index
    plan_digest: str
    #: the construction the fill will build: `config_digest/tensor_requirements_digest`
    construction_digest: str
    #: the resident ceiling the worker assigned - the same number on every rank
    authorized_device_limit_bytes: int

    def document(self) -> dict[str, Any]:
        row = asdict(self)
        row["devices"] = list(self.devices)
        return row

    @classmethod
    def read(cls, document: dict[str, Any]) -> GroupPlan:
        return cls(
            degree=int(document["degree"]),
            devices=tuple(str(entry) for entry in document["devices"]),
            plan_digest=str(document["plan_digest"]),
            construction_digest=str(document["construction_digest"]),
            authorized_device_limit_bytes=int(document["authorized_device_limit_bytes"]),
        )

    def assert_agrees(self, broadcast: GroupPlan, *, rank: int) -> None:
        """This rank's locally derived plan against rank 0's delivered one."""
        mine, theirs = asdict(self), asdict(broadcast)
        for key in sorted(mine):
            if mine[key] != theirs[key]:
                raise RankDivergence(
                    rank,
                    key,
                    f"rank 0 broadcast {theirs[key]!r}, this rank derived {mine[key]!r}; the "
                    "group fails and no rank adapts locally",
                )
