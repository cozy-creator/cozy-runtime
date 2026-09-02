"""Two model-bearing placements on one card (cr-022, over cr-066's lanes).

The owner's ask: "multiple executors sharing a GPU, i.e., one gets to use it first, then
the next gets to use it for a different request". Every arm below drives the REAL worker —
real trampoline, real executor processes, real weights through the real fill plane, the
real Claim/Snapshot/DesiredState/Offer conversation — against `corpus/tenant`, a package
whose model is the synthetic checkpoint TensorFS's own `tfs checkpoint write` produces.
Nothing is patched. Memory is read where the design reads it: the driver, through NVML,
from outside every CUDA context (cr-066's `accel.device_memory`).

WHAT IS REAL AND WHAT IS SIMULATED. This box has one card, which is exactly the SHARING
case: two placements assigned to lane 0 by measured fit. The co-fit arm holds two small
tenants resident at once and proves nothing moves; the swap arm makes the card too small
for both with a NEIGHBOUR process holding memory (a harness actor, decision #249, not a
plant in the runtime) and proves that each tenant's attempt evicts the other and re-stages,
with the driver's free bytes moving by the weights. The different-lanes arm can only be
simulated: lane 1 names a card that does not exist, so the second tenant's prepare is
refused `device_capacity_unreadable` by the driver's honest answer — which is itself the
proof that the assignment went to lane 1 and that lane 0's tenant was not touched.

The arms need `tfs` on PATH (to write the checkpoint), torch and tensorfs importable in
THIS interpreter (the executors are spawned from it), a driver-readable device 0, and an
executor-containing host; each absence names itself as the skip reason.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import accel, canonical, child_env
from cozy_runtime.internal import descriptor as desc
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import (
    LocalRecordOwner,
    LocalRequest,
    development_source_digest,
    stage_development_placement,
)
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.plan import UNCALIBRATED_ACTIVATION_BYTES
from cozy_runtime.internal.worker.session import HostedPlacement, Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "corpus" / "tenant"
DEVICE = "0"


def _tenant_host() -> str:
    """Why this host cannot run a model-bearing tenant, or "" when it can."""
    if NO_EXECUTOR:
        return NO_EXECUTOR
    if shutil.which("tfs") is None:
        return "the `tfs` CLI is not on PATH: no synthetic checkpoint can be written"
    try:
        import tensorfs  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        return f"torch and tensorfs must be importable in the test interpreter: {exc}"
    if accel.device_memory(DEVICE, accel.host_backend_family()).state != "measured":
        return "no driver-readable device 0 on this host"
    return ""


NO_TENANT = _tenant_host()
needs_tenant = pytest.mark.skipif(bool(NO_TENANT), reason=NO_TENANT or "")


# --------------------------------------------------------------------------- the fixture


@dataclasses.dataclass(frozen=True, slots=True)
class TenantStore:
    """One TensorFS store holding `cozytest/tenant@1` on lane `bf16`, and its size."""

    root: Path
    weight_bytes: int
    config: dict[str, int]


def write_tenant_store(root: Path, *, blocks: int, scale: int = 1) -> TenantStore:
    """Write one synthetic checkpoint and publish it as a repository release.

    `tfs checkpoint write --plain` objectizes a synthetic tree (tfs-015: `unet` blocks,
    `text_encoder`, `vae`, noise bytes) with a tokenizer ASSET the development lane does not
    stream, so the header is re-emitted without it — and with ONE construction config the
    package reads its shapes from — through `tfs checkpoint reproduce`, over the exact same
    tensor blobs. The release is then committed through the facade the worker itself reads
    with. `blocks` sets the size: 48 MiB per block at scale 1, plus a 128 MiB embedding.
    """
    import tensorfs

    store = root / "store"
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)

    def tfs(*args: str) -> str:
        return subprocess.run(
            ["nice", "-n", "19", "tfs", *args], capture_output=True, text=True, check=True
        ).stdout

    written = tfs(
        "checkpoint", "write", str(store), "--plain", "--blocks", str(blocks), "--scale", str(scale)
    )
    header_hex = re.search(r"header_digest\s+sha256:([0-9a-f]{64})", written).group(1)  # type: ignore[union-attr]
    tfs("get", str(store), header_hex, "--out", str(work / "header.cbor"))
    header = json.loads(tfs("cbor", "decode", str(work / "header.cbor")))
    components = {name: {row[0]: row for row in rows} for name, rows in header["components"]}
    unet = components["unet"]
    config = {
        "blocks": len({key.split(".")[1] for key in unet if key.startswith("blocks.")}),
        "hidden": int(unet["blocks.0.norm1.weight"][2][0]),
        "ff": int(unet["blocks.0.ff.net.0.proj.weight"][2][1]),
        "norm_q": int(unet["blocks.0.attn1.norm_q.weight"][2][0]),
        "text_embed": int(components["text_encoder"]["embeddings.weight"][2][0]),
        "text_norm": int(components["text_encoder"]["final_norm.weight"][2][0]),
        "vae": int(components["vae"]["decoder.conv_in.weight"][2][0]),
    }
    config_path = work / "config.json"
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config_path.write_bytes(config_bytes)
    config_hex = hashlib.sha256(config_bytes).hexdigest()
    tfs("put", str(store), str(config_path))
    header["assets"] = []
    header["configs"] = [["unet", f"sha256:{config_hex}", len(config_bytes), "application/json"]]
    (work / "header.json").write_text(json.dumps(header))
    (work / "order.json").write_text(
        json.dumps([[name, row[0]] for name, rows in header["components"] for row in rows])
    )
    (work / "entries.json").write_text(
        json.dumps(
            [
                ["model.cozytensors", "cozytensors"],
                ["unet/config.json", "file", {"sha256": config_hex, "length": len(config_bytes)}],
            ]
        )
    )
    reproduced = tfs(
        "checkpoint",
        "reproduce",
        str(store),
        str(work / "header.json"),
        str(work / "entries.json"),
        "--order",
        str(work / "order.json"),
    )
    manifest_id, manifest_length = re.search(  # type: ignore[union-attr]
        r"manifest\s+(sha256:[0-9a-f]{64}) length=(\d+)", reproduced
    ).groups()
    opened = tensorfs.Store.open(str(store))
    manifest = opened.manifest(manifest_id)
    topology = tensorfs.topology_digest(bytes(manifest["header"]))
    evidence = json.dumps(
        {"classification_digest": topology}, separators=(",", ":"), sort_keys=True
    ).encode()
    operation = opened.begin_operation("op-tenant", "cozytest", "tenant")
    operation.hold_manifest(manifest_id, int(manifest_length))
    operation.commit_release(None, "1", "bf16", manifest_id, int(manifest_length), evidence)
    weight_bytes = sum(int(row["length"]) for row in opened.walk(manifest_id))
    return TenantStore(root=store, weight_bytes=weight_bytes, config=config)


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> dict[str, TenantStore]:
    """Two tenant releases: one the card holds two of, one it does not (beside a neighbour)."""
    if NO_TENANT:
        pytest.skip(NO_TENANT)
    base = tmp_path_factory.mktemp("tenant")
    return {
        "small": write_tenant_store(base / "small", blocks=2, scale=4),
        "large": write_tenant_store(base / "large", blocks=48),
    }


def _config(home: Path) -> RuntimeConfig:
    base = {
        k: v
        for k, v in os.environ.items()
        if not child_env.erased(k) and k not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
    }
    return RuntimeConfig(
        cozy_home=home,
        credentials=Credentials(),
        child_base_env=tuple(sorted(base.items())),
    )


@contextlib.contextmanager
def _workspace() -> Iterator[Path]:
    # Short: the executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-share.", dir="/tmp") as raw:
        yield Path(raw)


def _free() -> int:
    memory = accel.device_memory(DEVICE, accel.host_backend_family())
    assert memory.state == "measured", memory
    return memory.free_bytes


# --------------------------------------------------------------------------- the neighbour

_NEIGHBOUR = """
import sys, torch
torch.cuda.init()
free, _ = torch.cuda.mem_get_info()  # the context exists now; the parent read free before
print("context", free, flush=True)
target = int(sys.argv[1])
held = []
while True:
    free, _ = torch.cuda.mem_get_info()
    if free <= target:
        break
    held.append(torch.empty(min(free - target, 64 << 20), dtype=torch.uint8, device="cuda"))
torch.cuda.synchronize()
print("held", torch.cuda.mem_get_info()[0], flush=True)
sys.stdin.readline()
"""


class Neighbour:
    """Another process holding device memory, so the card is SMALLER for the worker.

    It is what the runtime's own prepare path calls "a neighbour holding the card": a fact
    about the device the driver reports and the lane measures, never a number handed to
    the worker. It allocates until the driver reports `target` bytes free, and holds them
    until closed. Its CUDA context size is reported so the test can derive the target
    from measurements rather than from a guess.
    """

    def __init__(self, target: int) -> None:
        before = _free()
        self.process = subprocess.Popen(
            ["nice", "-n", "19", sys.executable, "-c", _NEIGHBOUR, str(target)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": DEVICE},
        )
        assert self.process.stdout is not None
        line = self.process.stdout.readline().split()
        assert line and line[0] == "context", line
        # what one process's CUDA context costs the DEVICE: the driver's free bytes before
        # the process existed, less what the process sees once its context is up
        self.context_bytes = before - int(line[1])
        line = self.process.stdout.readline().split()
        assert line and line[0] == "held", line
        self.free_after = int(line[1])

    def close(self) -> None:
        with contextlib.suppress(OSError):
            assert self.process.stdin is not None
            self.process.stdin.write("done\n")
            self.process.stdin.flush()
        self.process.wait(timeout=30)


def measure_context() -> int:
    """One executor-shaped CUDA context, measured: the neighbour's own, then released."""
    probe = Neighbour(target=1 << 62)
    try:
        return probe.context_bytes
    finally:
        probe.close()


# --------------------------------------------------------------------------- the owner


@dataclasses.dataclass(slots=True)
class Turn:
    """One attempt the owner dispatched, and what the worker and the driver said."""

    placement_id: str
    request: LocalRequest
    outcome: dict[str, Any] = dataclasses.field(default_factory=dict)
    free_bytes: int = -1
    rows: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)


class TwoTenantOwner(LocalRecordOwner):
    """The in-memory RecordOwner, hosting TWO placements and dispatching a SEQUENCE.

    Exactly `LocalRecordOwner`'s conversation — the same claim, snapshot ack, canonical
    desired state, spec/grant construction and terminal ack — over a two-placement set,
    one request at a time, each against the placement the turn names. A refused offer
    (`no_capacity` while the previous outcome's ack is in flight, a stale admission epoch)
    is re-dispatched on the next report with the next ordinal, exactly as a hub would.
    """

    def __init__(
        self,
        turns: list[Turn],
        bindings: dict[str, str],
        workspace: Path,
        *,
        package_revision_digest: str,
        placements: list[pb.Placement],
        worker: Callable[[], Worker],
    ) -> None:
        super().__init__(
            turns[0].request,
            bindings,
            workspace,
            package_revision_digest=package_revision_digest,
            placement=placements[0],
        )
        self.placements = placements
        self.pending = list(turns)
        self.turns: list[Turn] = []
        self.current: Turn | None = None
        self.worker_ref = worker
        self.refusals: list[str] = []

    def _desired_state(self) -> pb.DesiredWorkerState:
        directive = self._env(
            pb.DesiredWorkerState(
                revision=1, posture=pb.Posture.POSTURE_ACCEPTING, wire_minor=WIRE_MINOR
            )
        )
        placement_set = pb.PlacementSet(placements=self.placements)
        canonical_bytes, set_digest = documents.identity(placement_set)
        self.placement_set_digest = set_digest
        self.placement_set_canonical_bytes = canonical_bytes
        directive.placement_set.CopyFrom(
            pb.DesiredPlacementSet(
                placement_set_digest=set_digest, placement_set_canonical_bytes=canonical_bytes
            )
        )
        return directive

    def _on_observed_state(self, report: pb.ObservedWorkerState) -> None:
        self.reports.append(report)
        self.admission_epoch = report.admission_epoch
        if report.worker_phase == pb.WorkerPhase.WORKER_PHASE_FAILED:
            self.ended = "worker_error"
            self.error = "the worker failed"
            self.done.set()
            self.host.close()
            return
        refusal = self._refusal(report)
        if refusal:
            self.ended = "worker_error"
            self.error = f"the worker refused to become ready: {refusal}"[:400]
            self.done.set()
            self.host.close()
            return
        ready = report.admission_state == pb.AdmissionState.ADMISSION_STATE_OPEN and all(
            any(
                status.placement_id == placement.placement_id
                and status.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
                for status in report.placements
            )
            for placement in self.placements
        )
        if ready and self.current is None and self.pending:
            self.dispatched = True
            self._next()

    def _next(self) -> None:
        turn = self.pending.pop(0)
        self.current = turn
        self.request = turn.request
        self.accepted = None
        self.terminal = None
        self._dispatch_to(turn.placement_id)

    def _dispatch_to(self, placement_id: str) -> None:
        binding_digest = self._binding_digest()
        self.grant_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        body = canonical.write(self.request.payload)
        spec = pb.InvocationSpec(
            package_revision_digest=self.package_revision_digest,
            environment_digest="",
            config_digest="",
            payload_digest="sha256:" + hashlib.sha256(body).hexdigest(),
            inputs=self._input_bindings(body),
            outputs=self._output_bindings(),
            deadline_unix_ms=0,
        )
        spec.serving.entrypoint_binding_digest = binding_digest
        spec.serving.attempt_binding_id = binding_digest
        canonical_bytes, digest = documents.identity(spec)
        self.spec_digest = digest
        self.spec_canonical_bytes = canonical_bytes
        self.spec_document = documents.read(canonical_bytes, pb.InvocationSpec)
        self._send(
            pb.RecordOwnerFrame(
                attempt_offer=self._env(
                    pb.AttemptOffer(
                        request_id=self.request.request_id,
                        attempt_ordinal=self.request.attempt,
                        invocation_spec_digest=digest,
                        invocation_spec_canonical_bytes=canonical_bytes,
                        grant=self._grant(body, digest),
                        placement_id=placement_id,
                        admission_epoch=self.admission_epoch,
                    )
                )
            )
        )

    def _on_accepted(self, accepted: pb.AttemptAccepted) -> None:
        assert self.current is not None
        assert accepted.request_id == self.request.request_id, accepted.request_id
        assert accepted.placement_id == self.current.placement_id, accepted.placement_id
        self.accepted = accepted

    def _on_outcome(self, outcome: pb.AttemptOutcome) -> None:
        self._send(
            pb.RecordOwnerFrame(
                outcome_ack=self._env(
                    pb.AttemptOutcomeAck(
                        request_id=outcome.request_id,
                        attempt_ordinal=outcome.attempt_ordinal,
                        invocation_spec_digest=outcome.invocation_spec_digest,
                        outcome_id=outcome.outcome_id,
                        outcome_digest=outcome.outcome_digest,
                    )
                )
            )
        )
        turn = self.current
        assert turn is not None
        body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        if int(body.get("status", 0)) == pb.OutcomeStatus.OUTCOME_STATUS_REFUSED:
            self.refusals.append(str(body.get("safe_message", "")))
            self.current = None
            self.pending.insert(
                0,
                dataclasses.replace(
                    turn,
                    request=dataclasses.replace(turn.request, attempt=turn.request.attempt + 1),
                ),
            )
            return
        turn.outcome = body
        # THE MEASUREMENTS THIS TURN IS JUDGED ON: the driver's free bytes now, and each
        # tenant's row on the lane as the worker holds it.
        turn.free_bytes = _free()
        worker = self.worker_ref()
        for lane in worker.lanes.lanes:
            for slot, row in lane.rows.items():
                turn.rows[slot] = {
                    "resident": dict(row.ledger.resident),
                    "evicted": dict(row.ledger.evicted),
                    "baseline_notes": list(row.ledger.baseline_notes),
                    "executor_pid": row.ledger.executor_pid,
                    "executor_epoch": (
                        row.supervision.epoch if row.supervision is not None else -1
                    ),
                }
        self.turns.append(turn)
        self.current = None
        if self.pending:
            self._next()
            return
        self.ended = "terminal"
        self.done.set()
        self.host.close()


def _drive(
    store: TenantStore, turns: list[Turn], root: Path
) -> tuple[TwoTenantOwner, Worker, list[tuple[str, str]]]:
    """Two placements of `corpus/tenant` on the one-card worker, `turns` dispatched in order."""
    config = _config(root / "home")
    found = discover(PACKAGE)
    body = desc.build(found)
    workspace = root / "run"
    workspace.mkdir(parents=True, exist_ok=True)
    source_digest = development_source_digest(PACKAGE)
    staged = stage_development_placement(
        config.cozy_home,
        body,
        package="local/development-package",
        release="0.0.0",
        source_digest=source_digest,
        found=found,
        artifact_store=store.root,
    )
    placements = []
    for placement_id in dict.fromkeys(turn.placement_id for turn in turns):
        placement = pb.Placement()
        placement.CopyFrom(staged.placement)
        placement.placement_id = placement_id
        placements.append(placement)
    holder: list[Worker] = []
    owner = TwoTenantOwner(
        turns,
        dict(staged.bindings),
        workspace,
        package_revision_digest=documents.spell(staged.placement.development.source_digest),
        placements=placements,
        worker=lambda: holder[0],
    )
    options = WorkerOptions(
        root=workspace / "worker",
        devices=DEVICE,
        no_warm=True,
        instance_id="ins-shared-lane",
        artifact_store=staged.artifact_store,
        artifact_cache=staged.artifact_cache,
        development_project=PACKAGE,
        development_package=staged.placement.development.package,
        development_release=staged.placement.development.release,
        development_source_digest=documents.spell(staged.placement.development.source_digest),
    )
    worker = Worker(config, options, owner.host)
    holder.append(worker)
    notes: list[tuple[str, str]] = []
    narrate = worker.note

    def note(kind: str, step: str) -> None:
        notes.append((kind, step))
        narrate(kind, step)

    worker.note = note  # type: ignore[method-assign]
    worker.watches.tap = owner.tap_progress
    owner.watch(worker)
    with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
        worker.run()
    return owner, worker, notes


def _turn(placement_id: str, request_id: str, seed: int) -> Turn:
    return Turn(
        placement_id,
        LocalRequest(entrypoint="touch", payload={"seed": seed}, outputs=(), request_id=request_id),
    )


def _vacates(notes: list[tuple[str, str]]) -> list[tuple[str, str, int, int, int]]:
    """(victim, for-whom, freed, driver free before, driver free after) per vacate note."""
    found = []
    pattern = re.compile(
        r"lane-0: '([^']+)' \(epoch \d+ pid \d+\) vacated (\d+) B for '([^']+)' on .*?"
        r"the driver reports (\d+) -> (\d+) B free"
    )
    for kind, step in notes:
        if kind != "residency":
            continue
        match = pattern.search(step)
        if match:
            victim, freed, whom, before, after = match.groups()
            found.append((victim, whom, int(freed), int(before), int(after)))
    return found


# --------------------------------------------------------------------------- the co-fit arm


@needs_tenant
def test_two_tenants_that_fit_hold_the_card_together_and_nothing_moves(
    stores: dict[str, TenantStore],
) -> None:
    """THE CO-FIT ARM. Two small tenants on lane 0: both prepare resident, three attempts
    alternate between them, no vacate is ever issued, both rows keep their weights, and the
    driver's free bytes stay where two resident tenants leave them."""
    store = stores["small"]
    turns = [_turn("plc-p", "p-1", 1), _turn("plc-q", "q-1", 2), _turn("plc-p", "p-2", 3)]
    with _workspace() as root:
        owner, worker, notes = _drive(store, turns, root)
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    assert [turn.outcome["status"] for turn in owner.turns] == [1, 1, 1], [
        turn.outcome for turn in owner.turns
    ]
    assert _vacates(notes) == [], _vacates(notes)
    for turn in owner.turns:
        assert set(turn.rows) == {"plc-p", "plc-q"}, turn.rows
        for slot, row in turn.rows.items():
            assert sum(row["resident"].values()) >= store.weight_bytes * 0.99, (slot, row)
            assert row["evicted"] == {}, (slot, row)
    assert worker.lanes.lanes[0].placements == {"plc-p", "plc-q"}
    assigned = [step for kind, step in notes if kind == "materialization" and "assigned" in step]
    assert any(
        "'plc-q' assigned to lane-0" in step and "shares the device" in step for step in assigned
    ), assigned


# --------------------------------------------------------------------------- the swap arm


@needs_tenant
def test_two_tenants_that_do_not_fit_take_turns_on_the_card(
    stores: dict[str, TenantStore],
) -> None:
    """THE SWAP ARM — the owner's words made a proof. Beside a neighbour that leaves the
    card room for ONE tenant, an attempt for P finds Q resident: Q vacates (the driver's
    free bytes grow by Q's weights), P re-stages and runs; an attempt for Q does the
    reverse; a second attempt for P the reverse again. Neither executor is respawned —
    the epochs do not move — and no timer is anywhere in the decision."""
    store = stores["large"]
    weights = store.weight_bytes
    context = measure_context()
    # THE TARGET, DERIVED. With F free before either executor exists and each executor's
    # context costing about `context`:
    #   the second tenant's prepare must NOT fit beside the first (so residency has to be
    #   arbitrated at all):        F - context - weights < weights            => F < 2W + c
    #   a vacated tenant's re-stage must fit once the other has left, under the plan's
    #   uncalibrated activation bound H (the executor enforces it at admission):
    #                              F - 2*context >= weights + H               => F >= W + 2c + H
    # The window is non-empty only when W > c + H, which is why the fixture is this size;
    # the target is its midpoint, so a context a few hundred MiB off either way still lands.
    lower = weights + 2 * context + UNCALIBRATED_ACTIVATION_BYTES
    upper = 2 * weights + context
    if lower >= upper:
        pytest.skip(
            f"the fixture ({weights} B) is not larger than a context ({context} B) plus "
            "the activation bound"
        )
    target = (lower + upper) // 2
    if _free() <= target:
        pytest.skip(f"the card has {_free()} B free, under the {target} B the arm needs")
    neighbour = Neighbour(target)
    try:
        assert abs(neighbour.free_after - target) <= (64 << 20), (neighbour.free_after, target)
        turns = [_turn("plc-p", "p-1", 1), _turn("plc-q", "q-1", 2), _turn("plc-p", "p-2", 3)]
        with _workspace() as root:
            owner, _worker, notes = _drive(store, turns, root)
    finally:
        neighbour.close()
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    assert [turn.outcome["status"] for turn in owner.turns] == [1, 1, 1], [
        (turn.request.request_id, turn.outcome.get("cause"), turn.outcome.get("safe_message"))
        for turn in owner.turns
    ]
    vacates = _vacates(notes)
    # Q's prepare found P resident with no room beside it: P vacated for the prepare.
    # Then each attempt evicted the other tenant: Q for p-1, P for q-1, Q for p-2.
    assert [(victim, whom) for victim, whom, *_ in vacates] == [
        ("plc-p", "plc-q"),
        ("plc-q", "plc-p"),
        ("plc-p", "plc-q"),
        ("plc-q", "plc-p"),
    ], vacates
    for victim, _whom, freed, before, after in vacates:
        # MEMORY ACTUALLY MOVED: the executor freed the weights, and the driver — read from
        # outside every context — saw the free bytes grow by them (the allocator returns
        # its segments to the driver; a few MiB of segment rounding is the slack).
        assert freed >= weights * 0.99, (victim, freed, weights)
        assert after - before >= freed - (64 << 20), (victim, before, after, freed)
    # After each attempt exactly ONE tenant holds its weights: the one that just ran.
    for turn in owner.turns:
        resident = {slot: sum(row["resident"].values()) for slot, row in turn.rows.items()}
        evicted = {slot: sum(row["evicted"].values()) for slot, row in turn.rows.items()}
        other = "plc-q" if turn.placement_id == "plc-p" else "plc-p"
        assert resident[turn.placement_id] >= weights * 0.99, (turn.request.request_id, resident)
        assert resident[other] == 0 and evicted[other] >= weights * 0.99, (
            turn.request.request_id,
            resident,
            evicted,
        )
        # and the driver agrees: one tenant's weights are on the card (two could not be,
        # the target was chosen so — and had both stayed, free would be under zero)
        assert turn.free_bytes <= target - weights, (turn.free_bytes, target, weights)
    # NO RESPAWN: a swap changes residency, never the process. Both epochs are 1 throughout.
    for turn in owner.turns:
        assert {row["executor_epoch"] for row in turn.rows.values()} == {1}, turn.rows
        pids = {slot: row["executor_pid"] for slot, row in turn.rows.items()}
        assert pids == {slot: row["executor_pid"] for slot, row in owner.turns[0].rows.items()}, (
            pids
        )
    # the re-stage was DECLARED and reconciled, never called a leak
    for turn in owner.turns[1:]:
        notes_for = turn.rows[turn.placement_id]["baseline_notes"]
        assert any("declared residency transition" in note for note in notes_for), notes_for
    assert all("ledger_unreconciled" not in step for _kind, step in notes), [
        step for _kind, step in notes if "ledger_unreconciled" in step
    ]


# --------------------------------------------------------------------------- different lanes


@needs_tenant
def test_a_second_tenant_takes_another_lane_when_one_has_room(
    stores: dict[str, TenantStore],
) -> None:
    """DIFFERENT LANES (cr-066's item, lifted here). On a two-lane worker the primary
    tenant prepares for real on lane 0; a second model-bearing placement is assigned to
    lane 1 — the lane with no tenant — and its executor is sealed to `CUDA_VISIBLE_DEVICES=1`.
    On this box lane 1 names a card that does not exist, so its prepare is refused
    `device_capacity_unreadable` by the driver's honest answer; lane 0's tenant is never
    asked for anything, keeps its weights, and stays dispatchable. The assignment step is
    fed a measurement for the absent card by the harness (there is no other way to have two
    readable lanes here); everything after it is real."""
    store = stores["small"]
    with _workspace() as root:
        config = _config(root / "home")
        found = discover(PACKAGE)
        body = desc.build(found)
        source_digest = development_source_digest(PACKAGE)
        staged = stage_development_placement(
            config.cozy_home,
            body,
            package="local/development-package",
            release="0.0.0",
            source_digest=source_digest,
            found=found,
            artifact_store=store.root,
        )
        options = WorkerOptions(
            root=root / "worker",
            devices="0,1",
            no_warm=True,
            artifact_store=staged.artifact_store,
            artifact_cache=staged.artifact_cache,
            development_project=PACKAGE,
            development_package=staged.placement.development.package,
            development_release=staged.placement.development.release,
            development_source_digest=documents.spell(staged.placement.development.source_digest),
        )
        worker = Worker(config, options, InMemoryControlHost())
        seals: list[str] = []
        try:
            worker.accepted.mode = "serving"
            worker.accepted.accepted_desired_state_revision = 1
            document = documents.body(staged.placement)
            primary = worker._placement_from_entry({**document, "placement_id": "plc-p"}, b"")
            worker.placement = primary
            worker.bindings = dict(staged.declared_bindings)
            # a WIDE envelope: the RecordOwner pins (cr-068); the worker validates the pin
            # against the driver's reading of that card and never picks one itself
            worker.accepted.device_pins = {"plc-p": (0,)}
            assert worker._assign_lane(primary, worker.bindings, worker.supervision)
            assert primary.lane_id == "lane-0"
            with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
                worker.prepare_executor()
            assert primary.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE, worker.latched
            row_p = worker.lanes.lanes[0].rows["plc-p"]
            assert sum(row_p.ledger.resident.values()) >= store.weight_bytes * 0.99
            executor_p = worker.supervision.current
            assert executor_p is not None and executor_p.alive()

            hosted_placement = worker._placement_from_entry(
                {**document, "placement_id": "plc-q"}, b""
            )
            hosted = HostedPlacement(
                placement=hosted_placement, supervision=worker._new_placement_supervision("plc-q")
            )
            hosted.bindings = dict(staged.declared_bindings)
            worker.hosted["plc-q"] = hosted
            publish = hosted.supervision.on_change

            def on_change(executor: Any) -> None:
                if executor is not None:
                    seals.append(str(executor.hello["sealed"]["CUDA_VISIBLE_DEVICES"]))
                if publish is not None:
                    publish(executor)

            hosted.supervision.on_change = on_change
            # THE PLANT, named: lane 1's card does not exist, so the driver cannot measure
            # it and the fit rule would (rightly) never choose it. The assignment step alone
            # is handed a reading for it — what a second real card would report — so the
            # step-1 rule can be shown choosing the empty lane over sharing lane 0.
            # PINNED to ordinal 1 (cr-068): on a wide envelope the RecordOwner names the
            # device; the worker validates the pin against the reading and never chooses.
            lane_q = worker.lanes.assign(
                "plc-q",
                model_bearing=True,
                declared_bytes=store.weight_bytes,
                measured={
                    0: accel.device_memory("0", accel.host_backend_family()),
                    1: accel.DeviceMemory("measured", 8 << 30, 8 << 30),
                },
                pin=(1,),
            )
            assert lane_q.lane_id == "lane-1", lane_q
            hosted_placement.lane_id = "lane-1"
            row_q = lane_q.row("plc-q")
            row_q.model_bearing = True
            row_q.supervision = hosted.supervision
            with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
                worker._prepare_hosted_executor(hosted)
            assert seals == ["1"], seals
            assert hosted.latched == "prepare refused: every binding", hosted.latched
            reasons = [fault.reason for fault in worker.engine.faults]
            assert "device_capacity_unreadable" in reasons, reasons
            assert hosted_placement.serving == pb.ServingState.SERVING_STATE_OFFLINE
            # lane 0's tenant was never touched: same process, same weights, still serving
            assert worker.supervision.current is executor_p and executor_p.alive()
            assert sum(row_p.ledger.resident.values()) >= store.weight_bytes * 0.99
            assert primary.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
            assert worker.lanes.lane_of("plc-p") is worker.lanes.lanes[0]
            assert worker.lanes.lane_of("plc-q") is worker.lanes.lanes[1]
        finally:
            for slot in worker.hosted.values():
                slot.supervision.close()
            worker.supervision.close()
