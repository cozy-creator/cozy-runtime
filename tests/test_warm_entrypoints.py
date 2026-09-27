"""Two entrypoints of ONE construction on ONE executor, and a warm first stage (h3a-018).

The H3 package serves `first_last_frame_to_video` and `reference_media_to_video` over one
`H3Model`. Measured on an H100 pod, switching between them cost 55 s: a rebuilt executor
for a placement that changed nothing it fills, and a cold read of the other DiT's 40 GB at
0.73 GB/s against 23-33 GB/s from a warm page cache. Every arm below drives the REAL worker,
real executor processes and real weights through the real fill plane against
`corpus/tenant`, whose `touch` and `retouch` bind the same model under the same parameter:

* a desired set that ADDS the second entrypoint reuses the prepared executor — same pid,
  same epoch, both bindings ready, no replacement;
* both entrypoints execute through ordinary dispatch on that one process;
* a component the prepare PARKED is read through the store once, in the background, so
  its first stage reads from the page cache and not the block layer.

The arms need what `test_shared_lane` needs, and name the same skip reasons.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import fill, package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import (
    LocalRequest,
    stage_development_placement,
)
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.plan import UNCALIBRATED_ACTIVATION_BYTES
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_shared_lane import (
    DEVICE,
    NO_TENANT,
    PACKAGE,
    Neighbour,
    TenantStore,
    Turn,
    _config,
    _drive,
    _free,
    _workspace,
    measure_context,
    needs_tenant,
    write_tenant_store,
)


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> dict[str, TenantStore]:
    if NO_TENANT:
        pytest.skip(NO_TENANT)
    base = tmp_path_factory.mktemp("twin")
    return {
        "small": write_tenant_store(base / "small", blocks=2, scale=4),
        "large": write_tenant_store(base / "large", blocks=48),
    }


class _Booted:
    """The worker of the direct-prepare arms, with its notes captured."""

    def __init__(self, root: Path, store: TenantStore) -> None:
        config = _config(root / "home")
        found = discover(PACKAGE)
        body = package_interface.build(found)
        self.staged = stage_development_placement(
            config.cozy_home,
            body,
            package="local/development-package",
            release="0.0.0",
            installation_id="local-" + uuid.uuid4().hex,
            artifact_cache=root / "artifact-cache",
            tensorfs_root=store.root,
            project=PACKAGE,
        )
        options = WorkerOptions(
            root=root / "worker",
            devices=DEVICE,
            tensorfs_root=self.staged.tensorfs_root,
            artifact_cache=self.staged.artifact_cache,
            development_project=PACKAGE,
            development_package=self.staged.placement.development.package,
            development_release=self.staged.placement.development.release,
            development_installation_id=self.staged.placement.development.installation_id,
        )
        self.worker = Worker(config, options, InMemoryControlHost())
        self.notes: list[tuple[str, str]] = []
        narrate = self.worker.note

        def note(kind: str, step: str) -> None:
            self.notes.append((kind, step))
            narrate(kind, step)

        self.worker.note = note  # type: ignore[method-assign]
        self.worker.accepted.mode = "serving"
        self.worker.accepted.accepted_desired_state_revision = 1
        document = documents.body(self.staged.placement)
        self.placement = self.worker._placement_from_entry(
            {**document, "placement_id": "plc-p"}, b""
        )
        self.worker.placement = self.placement
        self.worker.accepted.device_pins = {"plc-p": (0,)}
        self.placement.device_pin = (0,)

    def binding(self, entrypoint: str) -> Any:
        return next(b for b in self.staged.declared_bindings.values() if b.entrypoint == entrypoint)

    def prepare(self, *entrypoints: str) -> None:
        self.worker.bindings = {
            digest: binding
            for digest, binding in self.staged.declared_bindings.items()
            if binding.entrypoint in entrypoints
        }
        if self.placement.lane_id == "":
            assert self.worker._assign_lane(
                self.placement, self.worker.bindings, self.worker.supervision
            )
        with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
            self.worker.prepare_executor()
        assert self.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE, (
            self.worker.latched
        )

    def load(self, entrypoint: str) -> None:
        """The attempt path's load and activation of one entrypoint's construction."""
        tenant = self.worker._tenant("plc-p")
        key = self.binding(entrypoint).construction_key()
        group = [b for b in tenant.bindings.values() if b.construction_key() == key]
        executor = self.worker.supervision.current
        assert executor is not None
        with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
            self.worker._load_construction(tenant, executor, key, group)
            self.worker._activate_construction(tenant, executor, key)

    def steps(self, pattern: str) -> list[str]:
        return [step for _kind, step in self.notes if re.search(pattern, step)]

    def close(self) -> None:
        self.worker.supervision.close()


# ----------------------------------------------------------------- one executor, two bindings


@needs_tenant
def test_adding_the_second_entrypoint_reuses_the_prepared_executor(
    stores: dict[str, TenantStore],
) -> None:
    """THE SWITCH ARM. The placement starts for `touch` alone and its construction loads;
    then the desired set adds `retouch` over the same construction. The process is the
    construction's home, not the binding set's: the executor is kept — same pid, same epoch
    — both bindings are dispatchable on it, and nothing is filled twice."""
    store = stores["small"]
    with _workspace() as root:
        booted = _Booted(root, store)
        try:
            booted.prepare("touch")
            executor = booted.worker.supervision.current
            assert executor is not None and executor.alive() and executor.started
            booted.load("touch")
            (loaded,) = executor.loaded.values()
            assert loaded.parameters == {"model"}
            assert len(booted.steps(r"construction \w{12} loaded in .*\(1 binding")) == 1

            booted.prepare("touch", "retouch")
            assert booted.worker.supervision.current is executor and executor.alive()
            assert executor.epoch == 1
            assert executor.ready_bindings == set(booted.staged.declared_bindings)
            assert booted.steps(r"reused started epoch 1 pid \d+ \(2 binding"), [
                step for _k, step in booted.notes
            ]
            booted.load("retouch")
            assert executor.loaded_bindings() == set(booted.staged.declared_bindings)
            assert not booted.steps(r"replacing the device executor"), booted.notes
            # ONE fill for the construction: the second load answered from what was built.
            fills = booted.steps(r"construction \w{12} loaded in")
            assert len(fills) == 2 and "reused" in fills[1] and "reused" not in fills[0], fills
        finally:
            booted.close()


@needs_tenant
def test_both_entrypoints_execute_on_the_one_executor(stores: dict[str, TenantStore]) -> None:
    """THE DISPATCH ARM, through the real Claim/Snapshot/DesiredState/Offer conversation:
    one placement carrying both bindings, an attempt on each, both served by epoch 1."""
    store = stores["small"]
    turns = [
        Turn(
            "plc-p",
            LocalRequest(entrypoint=name, payload={"seed": seed}, outputs=(), request_id=rid),
        )
        for name, seed, rid in (
            ("touch", 1, "t-1"),
            ("retouch", 2, "r-1"),
            ("touch", 3, "t-2"),
            ("retouch", 4, "r-2"),
            ("touch", 5, "t-3"),
        )
    ]
    with _workspace() as root:
        owner, _worker, notes = _drive(store, turns, root)
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    assert [turn.outcome["status"] for turn in owner.turns] == [1] * len(turns), [
        (turn.request.request_id, turn.outcome.get("cause"), turn.outcome.get("safe_message"))
        for turn in owner.turns
    ]
    loads = [step for _k, step in notes if re.search(r"construction \w{12} loaded in", step)]
    assert len(loads) == 1 and "(2 binding(s)" in loads[0], loads
    assert not [step for _k, step in notes if "replacing the device executor" in step]
    for turn in owner.turns:
        assert {row["executor_epoch"] for row in turn.rows.values()} == {1}, turn.rows
    pids = {row["executor_pid"] for turn in owner.turns for row in turn.rows.values()}
    assert len(pids) == 1, pids


# --------------------------------------------------------------------- the warm first stage


def _stored_bytes(store: TenantStore, component: str) -> int:
    """What the store holds for one component, as the fill plane reads it."""
    import tensorfs

    release = tensorfs.Store.open(str(store.root)).resolve_release(
        "cozytest", "tenant", "1", "bf16"
    )
    checkpoint = fill.Checkpoint(store.root, release["manifest_digest"])
    return sum(row.stored_nbytes for row in checkpoint.rows(component))


def _drop_page_cache(store: TenantStore) -> None:
    """Give the store's objects back to the block layer: the test's own files, so their
    first read is measurably cold. `sync` first — a dirty page is not dropped."""
    os.sync()
    for path in Path(store.root).rglob("*"):
        if not path.is_file():
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


@needs_tenant
def test_a_parked_component_is_page_warmed_before_its_first_stage(
    stores: dict[str, TenantStore],
) -> None:
    """THE WARM ARM. The worker reads every desired construction's stored bytes through the
    store in the background once the process starts — measured cold, at the block layer.
    A neighbour leaves the card room for the small components and not for `unet`, so the
    load PARKS it; once the neighbour leaves, the executor's restore of `unet` reads from
    the page cache."""
    store = stores["large"]
    unet = _stored_bytes(store, "unet")
    context = measure_context()
    # Room for the executor's context and the small components, not for the unet: the
    # ceiling the worker assigns is measured free bytes, and the fill parks the first
    # component that does not fit under it.
    target = context + (store.weight_bytes - unet) + unet // 2 + UNCALIBRATED_ACTIVATION_BYTES
    if _free() <= target:
        pytest.skip(f"the card has {_free()} B free, under the {target} B the arm needs")
    _drop_page_cache(store)
    neighbour = Neighbour(target)
    try:
        with _workspace() as root:
            booted = _Booted(root, store)
            try:
                booted.prepare("touch", "retouch")
                warm = next(t for t in threading.enumerate() if t.name == "page-warm:plc-p")
                warm.join()
                booted.load("touch")
                assert booted.steps(r"parked=\['unet'\]"), [s for _k, s in booted.notes]
                warmed = booted.steps(r"page-warmed 'unet'")
                assert len(warmed) == 1, [s for _k, s in booted.notes]
                found = re.search(
                    r": (\d+) B in (\d+) ms .*?, (\d+) B read from the block layer \((.+?)\)",
                    warmed[0],
                )
                assert found is not None, warmed[0]
                moved, read, verdict = int(found.group(1)), int(found.group(3)), found.group(4)
                assert moved >= unet * 0.99, (moved, unet)
                assert verdict == "cold" and read >= moved // 2, warmed[0]
                neighbour.close()
                executor = booted.worker.supervision.current
                assert executor is not None
                reply = executor.call({"cmd": "restore", "names": ["unet"]}, timeout=None)
                restored = reply["restored"]
                assert restored["restored"].get("unet", 0) >= unet * 0.99, restored
                assert 0 <= restored["read_bytes"] <= restored["restored_bytes"] // 2, restored
            finally:
                booted.close()
    finally:
        with contextlib.suppress(Exception):
            neighbour.close()
