"""GroupPlan: rank 0 decides, every rank obeys - the whole doctrine, in one file.

Every adaptive decision that must be identical across a group's ranks is a field here.
Rank 0 authors the plan from ITS resolution and hands it to the followers with the prepare;
a follower resolves the same inputs on its own card and ASSERTS agreement, field by field.
A rank never adapts locally: a group whose ranks would run different plans produces
silently wrong output (a per-tensor activation scale derived from the local shard, a
different resident set, a different route), and the only honest answer is to refuse.
"""

from __future__ import annotations

from collections.abc import Mapping

import msgspec

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.internal import canonical, execution_evidence


class GroupRefusal(RuntimeFailure):
    """The group could not form, agree, or stay whole. RUNTIME origin: the request and
    the author are sound; the executor's process group is not. `code` names which."""

    default_code = "group_broken"


class GpuDivergence(GroupRefusal):
    """One GPU's process could not honour the group's plan. `gpu` names it as people do."""

    default_code = "gpu_divergence"

    def __init__(self, gpu: str, field_name: str, detail: str) -> None:
        super().__init__(f"{gpu} disagrees on {field_name!r}: {detail}", fields=[field_name])
        self.gpu = gpu
        self.field_name = field_name


def rank_invariant_digest(identity: Mapping[str, object]) -> str:
    """A resolved plan's identity with the one rank-LOCAL fact removed: the device INDEX.

    Rank r measures card r of the seal, so `Plan.digest()` differs across the ranks of a
    group by construction; everything else in it - the bytes, every tensor's route, the
    device CLASS, the runtime, the construction contract - must agree, and this is the
    digest that says so.
    """
    device = identity["device"]
    if not isinstance(device, dict):
        raise TypeError("a plan identity names its device as a mapping")
    return canonical.digest(
        {**identity, "device": {k: v for k, v in device.items() if k != "index"}}
    )


class GroupPlan(msgspec.Struct, frozen=True):
    """Every fact the K ranks must agree on before a weight moves."""

    #: the sequence-parallel degree; equals the number of sealed devices
    degree: int
    #: the seal's entries, in order - rank r holds entry r
    devices: tuple[str, ...]
    #: the resolved fill plan's RANK-INVARIANT digest (`rank_invariant_digest`): every
    #: tensor's route, the bytes, the device class - not the card's index
    plan_digest: str
    #: the resident ceiling the worker assigned - the same number on every rank
    authorized_device_limit_bytes: int

    def document(self) -> dict[str, object]:
        return {**msgspec.structs.asdict(self), "devices": list(self.devices)}

    @classmethod
    def read(cls, document: Mapping[str, object]) -> GroupPlan:
        """A delivered plan; a malformed one raises `msgspec.ValidationError`."""
        return msgspec.convert(document, cls)

    def assert_agrees(self, broadcast: GroupPlan, *, rank: int) -> None:
        """This rank's locally derived plan against rank 0's delivered one."""
        for key in sorted(self.__struct_fields__):
            mine, theirs = getattr(self, key), getattr(broadcast, key)
            if mine != theirs:
                gpu = execution_evidence.gpu_name(broadcast.devices, rank)
                leader = execution_evidence.gpu_name(broadcast.devices, 0)
                raise GpuDivergence(
                    gpu,
                    key,
                    f"{leader} decided {theirs!r}, {gpu} derived {mine!r}; the group fails "
                    "and no GPU adapts locally",
                )
