"""The RUNTIME's side of a binding, for live runs: resolved artifacts with real topologies.

Selection produced these — author code never names any part of one (§1.1). Each artifact's
topology table is derived by censusing a REFERENCE construction, which is what a checkpoint
header supplies to the match at exact binding; the dual artifact is the union of both role
graphs plus the shared components, exactly as job-001 packages it.

Not a package: this is fixture plumbing for `scripts/verify.py` and `scripts/redarm.py`.
"""

from __future__ import annotations

from collections.abc import Mapping

from tiny_diffusers import Task, TinySdxlPipeline
from tiny_diffusers import build_h3_pipeline as _build_h3

from cozy_runtime.author import Artifact, Config, TensorSpec
from cozy_runtime.author._loader import topology_of

H3_CONFIG: Mapping[str, object] = {"hidden_size": 32, "pruned": False}
PRUNED_CONFIG: Mapping[str, object] = {"hidden_size": 32, "pruned": True}
SDXL_CONFIG: Mapping[str, object] = {"hidden_size": 32}


def sdxl(snapshot: str = "cozy/tiny-sdxl@v1#a1b2", **config: object) -> Artifact:
    data = {**SDXL_CONFIG, **config}
    return Artifact(
        snapshot=snapshot,
        topology=topology_of(TinySdxlPipeline(Config(data))),
        config=Config(data),
    )


def h3(
    *, task: Task, snapshot: str | None = None, pruned: bool = False, dual: bool = False
) -> Artifact:
    """A narrow role artifact, or the DUAL package carrying both transformer roles.

    STAMPS ARE A CHECKPOINT FACT, NOT A LANE FACT (#567d). The dual package carries both
    task-role transformers, each with its own evidence, so its stamp key holds BOTH values
    and its contract renders `task=fl2va+ref2va`; a class keyword joins that set by
    membership. The previous fixture stamped the dual artifact with whichever single task
    the caller asked for, which is the one shape the hub cannot produce — stamps live per
    snapshot, so every lane over one checkpoint reads the same set.

    `task=` therefore selects the TOPOLOGY of a narrow package and is ignored for the stamp
    of a dual one. Two slots on the dual artifact share a snapshot and config yet keep
    distinct construction identities, because identity includes the model class (§1.1).
    """
    data = dict(PRUNED_CONFIG if pruned else H3_CONFIG)
    config = Config(data)
    topology: dict[str, TensorSpec] = {}
    roles: tuple[Task, ...] = ("fl2va", "ref2va") if dual else (task,)
    for role in roles:
        topology.update(topology_of(_build_h3(config, task=role)))
    label = "h3-dual" if dual else f"h3-{task}"
    lane = "adaln-pruned" if pruned else "full"
    return Artifact(
        snapshot=snapshot or f"cozy/{label}@v1/{lane}#c3d4",
        topology=topology,
        config=config,
        stamps={"task": roles, "family": ("h3",)},
    )


def named(artifact: Artifact, snapshot: str) -> Artifact:
    """The same graph under a different exact snapshot — a DIFFERENT construction identity."""
    return Artifact(
        snapshot=snapshot,
        topology=artifact.topology,
        config=artifact.config,
        stamps=artifact.stamps,
        variant=artifact.variant,
        custody=artifact.custody,
    )
