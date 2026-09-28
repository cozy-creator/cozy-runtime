"""The startup warm (h3a-087): a serving call's executor starts, and proves its attention
kernels, while its weights download; a request that is ready takes the started process without
waiting on warm work it does not need; a started process never sits on another root's GPU.

Real Worker, real durable journal, a real TensorFS pull from an in-process origin whose
object GETs are held open (the slow download), a real package environment the worker opens
like any installed one, and real executor processes on this machine's GPU. The one fake is
`accel.device_memory`, the hardware memory seam, where a test needs cards this box lacks.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import signed_claims
from cozy_runtime import canonical_json
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.gpu_scheduler import Demand
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.plan import ModelBinding as WorkerSlot
from cozy_runtime.internal.worker.session import HostedPlacement, Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config
from test_end_to_end import NO_EXECUTOR
from test_gpu_scheduler import MARCO, Machine
from test_model_runtime_closure import _ASSET, _HEADER, _snapshot

OWNER = "owner"
INSTALLATION = "release-startup-warm-fixture"


def _gpu() -> str:
    if NO_EXECUTOR:
        return NO_EXECUTOR
    try:
        import torch
    except ImportError as exc:
        return f"torch is absent: {exc}"
    return "" if torch.cuda.is_available() else "no CUDA device"


NO_GPU = _gpu()
needs_gpu = pytest.mark.skipif(bool(NO_GPU), reason=NO_GPU or "")


def _interface(degrees: tuple[int, ...] = ()) -> bytes:
    body = copy.deepcopy(package_interface.build(discover(MARCO)))
    entry = next(row for row in body["entrypoints"] if row["name"] == "marco")
    slot: dict[str, Any] = {"path": "marco.models.model", "class": "M", "component_use": {}}
    if degrees:
        slot["sequence_parallel"] = {"degrees": list(degrees)}
    entry["models"] = [slot]
    return package_interface.canonical_bytes(body)


def _install(root: Path) -> None:
    """A package environment as the worker records one: its own venv, whose path reaches this
    interpreter's packages (torch, the runtime) and the marco-polo package it serves."""
    home = root / "installations" / INSTALLATION
    venv = home / "venv"
    subprocess.run(
        [sys._base_executable, "-m", "venv", "--without-pip", str(venv)],  # type: ignore[attr-defined]
        check=True,
    )
    (site,) = (venv / "lib").glob("python*/site-packages")
    (site / "startup_warm_fixture.pth").write_text(
        f"import site; site.addsitedir({sysconfig.get_paths()['purelib']!r})\n{MARCO}\n"
    )
    (home / "installation.json").write_text(
        json.dumps({"python": str(venv / "bin" / "python"), "package": "t/m", "release": "1"})
    )


class Origin:
    """A TensorFS public origin serving one model; every object GET waits for `release`."""

    def __init__(self, root: Path) -> None:
        source = root / "origin"
        source.mkdir()
        _, manifest, length, store = _snapshot(source, include_asset=True, checkpoint_only=True)
        self.checkpoint: dict[str, Any] = {"digest": manifest, "length": length}
        bodies = {hashlib.sha256(body).hexdigest(): body for body in (_HEADER, _ASSET)}
        objects = [{"length": len(b), "sha256": d} for d, b in sorted(bodies.items())]
        bodies[manifest[7:]] = store.manifest(manifest)["manifest"]
        self.release = threading.Event()
        self.started = threading.Event()
        self.landed = 0.0
        origin = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def answer(self, body: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                port = origin.server.server_port
                if self.path == "/v1/tensorfs/closure":
                    response: dict[str, Any] = {
                        "complete": True,
                        "lane": "bf16",
                        "model": "proof/model",
                        "manifest": {"sha256": manifest[7:], "length": length},
                        "objects": objects,
                        "presign_max_digests": 10,
                        "release": "1.0.0",
                        "scope": "runtime",
                        "server_time_unix": int(time.time()),
                    }
                else:
                    response = {
                        "expires_at_unix": int(time.time()) + 3600,
                        "server_time_unix": int(time.time()),
                        "urls": {d: f"http://127.0.0.1:{port}/{d}" for d in request["digests"]},
                    }
                self.answer(json.dumps(response, sort_keys=True, separators=(",", ":")).encode())

            def do_GET(self) -> None:
                origin.started.set()
                assert origin.release.wait(600)
                self.answer(bodies[self.path.removeprefix("/")])
                origin.landed = time.time()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://localhost:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class Warm(Machine):
    """The scheduler harness's worker, with a package environment, a store and an artifact
    cache: what a machine preparing a serving child actually holds."""

    def __init__(self, root: Path, devices: str) -> None:
        self.worker = Worker(
            replace(
                _config(root / "home"),
                record_owner_public_key=signed_claims.PUBLIC_KEY,
                object_storage_hosts=("127.0.0.1",),
            ),
            WorkerOptions(
                **signed_claims.IDENTITY,
                root=root / "worker",
                tensorfs_root=root / "store",
                install_root=root / "install",
                artifact_cache=root / "artifacts",
                devices=devices,
            ),
            InMemoryControlHost(),
        )
        claim = signed_claims.claim(OWNER)
        assert self.worker.accept_claim(claim, lambda frame: None)[0] >= 1
        assert self.worker.executions is not None and self.worker.machine_calls is not None
        self.executions = self.worker.executions
        self.calls = self.worker.machine_calls
        self.children: dict[str, int] = {}
        _install(root / "install")

    def captured_root(self, name: str, origin: Origin) -> None:
        """A running orchestration root whose capture names the served model's default."""
        capture = pb.MachineExecutionCapture(
            model_defaults=[
                pb.MachineModelDefault(
                    callee_installation_id=INSTALLATION,
                    entrypoint="marco",
                    parameter="model",
                    public_origin=origin.url,
                    rungs=[
                        pb.MachineModelDefaultRung(
                            gpu="*",
                            repository="proof/model",
                            manifest=pb.Ref(
                                digest=documents.raw(origin.checkpoint["digest"]),
                                length=origin.checkpoint["length"],
                            ),
                        )
                    ],
                )
            ]
        )
        raw, _ = documents.identity(capture)
        self.submit_root(name, raw)

    def submit_root(self, name: str, capture: bytes = b"") -> None:
        spec = pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                installation_id="inst-root", job_descriptor_id="sha256:" + "12" * 32
            )
        )
        raw, digest = documents.identity(spec)
        self.executions.submit(
            OWNER,
            name,
            hashlib.sha256(name.encode()).digest(),
            pb.AttemptOffer(
                request_id=name,
                attempt_ordinal=1,
                invocation_spec_canonical_bytes=raw,
                invocation_spec_digest=digest,
                grant=pb.DeliveryGrant(invocation_spec_digest=digest),
            ),
            expected_execution_workspace_id=self.executions.workspace_id,
            worker_boot=self.worker.fence.worker_boot_id,
            capture_document=capture,
        )
        accepted = pb.AttemptAccepted(
            request_id=name, attempt_ordinal=1, invocation_spec_digest=digest
        )
        self.executions.dispatched(OWNER, accepted.request_id, accepted.attempt_ordinal)

    def target(self, checkpoint: dict[str, Any], degrees: tuple[int, ...] = ()) -> Target:
        digest = "sha256:" + hashlib.sha256(b"warm-binding").hexdigest()
        placement = {
            "placement_id": "tmpl-warm",
            "installation_id": INSTALLATION,
            "package_interface": base64.b64encode(_interface(degrees)).decode(),
            "bindings_digest": digest,
            "models": [{"id": "model", "manifest": checkpoint}],
            "entrypoints": [
                {
                    "name": "marco",
                    "entrypoint_binding_digest": digest,
                    "slots": [
                        {
                            "slot": "model",
                            "reference_model_id": "model",
                            "components": [{"component": "unet", "model_id": "model"}],
                        }
                    ],
                }
            ],
        }
        prepared = self.worker._placement_from_entry(
            documents.from_body(placement, pb.Placement), b""
        )
        self.worker.prepared_installations[prepared.prepared_key] = prepared
        models = [{"path": "marco.models.model"}]
        if degrees:
            models[0]["sequence_parallel"] = {"degrees": list(degrees)}  # type: ignore[assignment]
        return Target(INSTALLATION, "marco", {"models": models}, {"placement": placement}, None, "")

    def direct_root(self, name: str) -> None:
        """A serving root as Creator submits a rented direct root once the Host's preparation
        answers: its template names the installed fixture, and no call asked for it before."""
        placement = {
            **self.target({"digest": "sha256:" + "0" * 64, "length": 1}).prepared_installation[
                "placement"
            ],
            "package": {"package": "t/m", "release": "1"},
            "models": [
                {
                    "id": "model",
                    "repo": "proof/model",
                    "manifest": {"digest": "sha256:" + "0" * 64, "length": 1},
                }
            ],
        }
        raw = canonical_json.encode(
            {"format": "cozy.worker.v1.PlacementSet/1", "placements": [placement]}
        )
        state = pb.DesiredWorkerState(
            wire_minor=WIRE_MINOR,
            posture=pb.POSTURE_ACCEPTING,
            placement_set=pb.DesiredPlacementSet(
                placement_set_digest=hashlib.sha256(raw).digest(),
                placement_set_canonical_bytes=raw,
            ),
        )
        digest = placement["bindings_digest"]
        self.submit(
            name,
            pb.InvocationSpec(
                installation_id=INSTALLATION,
                serving=pb.ServingInvocationSpec(
                    entrypoint_binding_digest=digest,
                    attempt_binding_id=digest,
                    bindings_digest=digest,
                ),
            ),
            {"installations": {}, "state": base64.b64encode(state.SerializeToString()).decode()},
            placement=placement["placement_id"],
        )

    def phases(self, root: str) -> dict[str, dict[str, Any]]:
        """The root's journaled phase records, by name: what `cozy run show` renders."""
        found: dict[str, dict[str, Any]] = {}
        for event in self.executions.events(OWNER, root).events:
            body = canonical_json.decode(event.body)
            fields = (body.get("payload") or {}).get("fields") if isinstance(body, dict) else None
            if isinstance(fields, dict) and "phase" in fields:
                found[str(fields["phase"])] = fields
        return found


@pytest.fixture
def warm() -> Iterator[Callable[..., Warm]]:
    made: list[Warm] = []
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-warm.", dir="/tmp") as raw:

        def make(devices: str = "0") -> Warm:
            made.append(Warm(Path(raw) / str(len(made)), devices))
            return made[-1]

        try:
            yield make
        finally:
            for machine in made:
                machine.worker.shutdown()


def _wait(condition: Callable[[], Any], what: str) -> Any:
    """Poll a real process's progress; the bound is the test's patience, not a product rule."""
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"never observed: {what}")


def _call(parent: str) -> tuple[Call, pb.ChildCallRequest]:
    call = Call(parent, 0, 1, 1, b"", b"", "child-warm", b"", False, b"", "", "")
    return call, pb.ChildCallRequest(parent_attempt_ordinal=1)


@needs_gpu
def test_a_slow_download_overlaps_the_executor_start_and_its_probes(
    warm: Callable[..., Warm],
) -> None:
    machine = warm()
    origin = Origin(Path(tempfile.mkdtemp(prefix="cz-origin.", dir="/tmp")))
    try:
        machine.captured_root("A", origin)
        target = machine.target(origin.checkpoint)
        call, request = _call("A")
        serving = machine.worker.machine_calls.serving  # type: ignore[union-attr]
        future = serving._shared(OWNER, call, target, request, {}, speculative=False)
        assert future is not None
        assert origin.started.wait(60), f"the download never began: {future.exception(0.1)!r}"
        # The weights are still in flight: every object GET is held open.
        phases = _wait(
            lambda: (p := machine.phases("A")).get("Starting kernel compiles") and p,
            "the prespawned executor starting its compiles",
        )
        assert not future.done() and origin.landed == 0.0
        started = phases["Starting model executor"]
        probes = phases["Starting kernel compiles"]
        assert started["completed"] and probes["completed"], phases
        assert started["ordinals"] == [0] and "torch_import" in started["detail"]
        assert "GPU 0: " in probes["detail"] and "sdpa ready" in probes["detail"], probes
        # The download still holds; now it lands, after the start and the probes.
        origin.release.set()
        prepared = future.result(120)
        assert prepared.placement["installation_id"] == INSTALLATION
        assert origin.landed * 1000 > probes["started_unix_ms"] + probes["elapsed_ms"]
        # The started process waits on its GPU, sealed there, at the lowest priority.
        (slot,) = machine.worker.prespawns.slots.values()
        assert slot.executor is not None and slot.executor.alive() and slot.executor.started
        assert slot.executor.sealed_devices == "0"
        assert machine.worker._warm((INSTALLATION, "any")) == (0,)
        assert os.getpriority(os.PRIO_PROCESS, slot.executor.pid) == 19
    finally:
        origin.close()


@needs_gpu
def test_a_ready_request_takes_the_started_process_and_skips_the_probes_it_does_not_need(
    warm: Callable[..., Warm],
) -> None:
    """The grant arrives while the process is still starting: the replica waits out only the
    start it would pay itself, takes that very process at ordinary priority, and no probe is
    sent to it — its own construction proves exactly the kernels it selects."""
    machine = warm()
    machine.submit_root("A")
    target = machine.target({"digest": "sha256:" + "0" * 64, "length": 1})
    emitted: list[dict[str, Any]] = []
    machine.worker.prespawns.request(OWNER, "A", target, emitted.append)
    slot = machine.worker.prespawns.claim(INSTALLATION, (0,))
    assert slot is not None and not slot.started.is_set(), "the claim came mid-start"
    machine.worker.prespawns.settle(slot, "machine-adopter")
    assert slot.finished.wait(60)
    executor = slot.executor
    assert executor is not None and executor.started and executor.alive()
    # Adopted at the worker's own priority: the warm's lowest priority never applied.
    assert os.getpriority(os.PRIO_PROCESS, executor.pid) == os.getpriority(os.PRIO_PROCESS, 0)
    assert [frame["fields"]["phase"] for frame in emitted] == ["Starting model executor"]

    # The replica's own prepare reuses the process: no second spawn, no second start.
    placement = machine.worker._placement_from_entry(
        documents.from_body(
            {**target.prepared_installation["placement"], "placement_id": "machine-adopter"},
            pb.Placement,
        ),
        b"",
    )
    placement.device_pin = (0,)
    binding = DeclaredBinding(
        entrypoint_binding_digest="sha256:" + hashlib.sha256(b"warm-binding").hexdigest(),
        entrypoint="marco",
        model_class="M",
        model_binding_path="marco.models.model",
        model_parameter_name="model",
        release="r",
        logical_weight_bytes=1 << 30,
        models=(
            WorkerSlot(
                *("M", "marco.models.model", "model", "", "", ""),
                components=("unet",),
                snapshots={},
            ),
        ),
    )
    hosted = HostedPlacement(
        placement, slot.supervision, {binding.entrypoint_binding_digest: binding}
    )
    machine.worker.hosted[placement.placement_id] = hosted
    with machine.worker.control_lock:
        assert machine.worker._assign_lane(placement, hosted.bindings, hosted.supervision)
    latch, reused = machine.worker._prepare_generation(machine.worker._tenant("machine-adopter"))
    assert (latch, reused) == ("", True)
    assert hosted.supervision.current is executor and executor.alive()


@needs_gpu
def test_a_prespawn_never_sits_on_another_roots_gpu_and_yields_when_one_is_granted_it(
    warm: Callable[..., Warm],
) -> None:
    """Root A holds the card: B's download starts no process there. Once A ends the card is
    free and B's process starts on it; when C's call is then granted that card, B's process
    leaves it in the same scheduling pass, before C's replica binds."""
    machine = warm()
    machine.submit_root("A")
    machine.submit_root("B")
    target = machine.target({"digest": "sha256:" + "0" * 64, "length": 1})
    a1 = machine.child("A", "qwen")
    machine.tick()
    assert machine.granted(a1) == [0]
    machine.worker.prespawns.request(OWNER, "B", target, lambda frame: None)
    assert machine.worker.prespawns.slots == {}, "B started a process on A's card"

    machine.finish(a1)
    machine.finish("A")
    machine.tick()
    assert machine.worker.gpu.view()["leases"].get("A") is None
    machine.worker.prespawns.request(OWNER, "B", target, lambda frame: None)
    (slot,) = machine.worker.prespawns.slots.values()
    assert slot.root == "B" and slot.ordinals == (0,)
    assert slot.started.wait(120) and slot.executor is not None and slot.executor.alive()
    executor = slot.executor

    machine.submit_root("C")
    c1 = machine.child("C", "qwen")
    machine.tick()
    assert machine.granted(c1) == [0]
    assert machine.worker.prespawns.slots == {}
    _wait(
        lambda: not executor.alive() and slot.supervision.current is None,
        "B's prespawned executor reclaimed off C's card",
    )


@needs_gpu
def test_the_warm_starts_every_chain_kernel_and_selection_reads_what_is_ready() -> None:
    """The warm returns at once with a row per kernel of the chain and the ranked table: ready,
    compiling, absent here, or unsupported on this card, each with why. A construction then
    selects the first ready one."""
    import torch
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    from cozy_runtime.internal import attention
    from cozy_runtime.internal.encoding import measure_device

    device = measure_device(torch, 0)
    preferred = ("sol-attn", "sageattention", "flash-attn3", "sdpa")
    rows = {row["kernel"]: row for row in attention.expect(device, preferred)}
    assert set(rows) >= set(preferred)
    assert rows["sdpa"]["status"] == "ready"
    assert all(row["status"] == "ready" or row.get("detail") for row in rows.values()), rows
    site = WanAttention(dim=1024, heads=8, dim_head=128, processor=WanAttnProcessor())
    roots = {"dit": site.to("cuda", torch.bfloat16)}
    applied = attention.select(device, roots, choose=lambda context: preferred)
    [served] = applied.totals()
    assert rows[served]["status"] == "ready"


@needs_gpu
def test_probes_wait_while_a_call_runs_on_the_card_and_follow_when_it_leaves(
    warm: Callable[..., Warm],
) -> None:
    """The root's own call holds the card: its process may start there (the root holds it),
    but no probe runs beside a call on the device; the pass that sees the card released
    starts them. The scheduler is driven directly, so the call's grant lasts exactly as
    long as the test says."""
    machine = warm()
    machine.submit_root("A")
    gpu, prespawns = machine.worker.gpu, machine.worker.prespawns
    roots = {"A": 1}
    assert gpu.sync(roots, [Demand("A-call#1", "A", 1)]) == {"A-call#1": (0,)}
    emitted: list[dict[str, Any]] = []
    target = machine.target({"digest": "sha256:" + "0" * 64, "length": 1})
    prespawns.request(OWNER, "A", target, emitted.append)
    (slot,) = prespawns.slots.values()
    assert slot.started.wait(120) and slot.executor is not None and slot.executor.started
    for _ in range(20):
        prespawns.reconcile(roots, gpu.view())
        time.sleep(0.05)
    assert [frame["fields"]["phase"] for frame in emitted] == ["Starting model executor"]
    gpu.release("A-call#1")
    gpu.sync(roots, [])
    released = time.time()
    prespawns.reconcile(roots, gpu.view())
    assert slot.finished.wait(120)
    probes = emitted[-1]["fields"]
    assert probes["phase"] == "Starting kernel compiles" and probes["completed"], probes
    assert probes["started_unix_ms"] >= int(released * 1000)


@needs_gpu
def test_a_host_started_warm_is_adopted_by_the_direct_root_granted_its_card(
    warm: Callable[..., Warm],
) -> None:
    """h3a-089: the Host's preparation starts the installation's executor while it lands the
    weights of a root nobody has submitted yet. The process starts and proves its kernels on
    a free card with its rows waiting in the slot; the direct root submitted afterwards is
    granted that card, adopts that very process, and its journal carries both phases with
    the times they actually ran."""
    machine = warm()
    prespawns = machine.worker.prespawns
    prespawns.request_landing(INSTALLATION, _interface(), [{"path": "marco.models.model"}])
    (slot,) = prespawns.slots.values()
    assert slot.root == "" and slot.ordinals == (0,)
    assert slot.finished.wait(300)
    assert [frame["fields"]["phase"] for frame in slot.frames] == [
        "Starting model executor",
        "Starting kernel compiles",
    ]
    assert all(frame["fields"]["completed"] for frame in slot.frames), slot.frames
    # A second landing preparation of the same installation starts nothing more.
    prespawns.request_landing(INSTALLATION, _interface(), [{"path": "marco.models.model"}])
    assert list(prespawns.slots.values()) == [slot]
    executor = slot.executor
    assert executor is not None and executor.alive()

    submitted = time.time()
    machine.direct_root("D")
    machine.tick()
    assert machine.granted("D") == [0]
    (hosted,) = machine.worker.hosted.values()
    assert hosted.supervision is slot.supervision and slot.root == "D"
    assert prespawns.slots == {} and slot.frames == []
    phases = machine.phases("D")
    started = phases["Starting model executor"]
    probes = phases["Starting kernel compiles"]
    assert started["completed"] and "sdpa ready" in probes["detail"], phases
    assert started["ordinals"] == probes["ordinals"] == [0]
    assert probes["started_unix_ms"] + probes["elapsed_ms"] < submitted * 1000


@needs_gpu
def test_a_host_started_warm_yields_to_a_root_of_another_installation(
    warm: Callable[..., Warm],
) -> None:
    """A process nobody adopted is not a reservation: another installation's root granted
    its card takes it in the same scheduling pass."""
    machine = warm()
    prespawns = machine.worker.prespawns
    prespawns.request_landing(INSTALLATION, _interface(), [{"path": "marco.models.model"}])
    (slot,) = prespawns.slots.values()
    assert slot.started.wait(300) and slot.executor is not None
    executor = slot.executor
    machine.serving("B", "qwen")
    machine.tick()
    assert machine.granted("B") == [0]
    assert prespawns.slots == {} and slot.root == ""
    _wait(
        lambda: not executor.alive() and slot.supervision.current is None,
        "the unadopted prespawn reclaimed off B's card",
    )


@needs_gpu
def test_a_dead_replica_starts_its_successor_beside_the_next_download(
    warm: Callable[..., Warm],
) -> None:
    """Run 1516: a dead replica's cards were prespawned in a fresh slot whose root the
    replica's own slot still owned, so every prefetch failed and the group start was paid
    at the grant. The successor now starts early in the replica's own slot."""
    machine = warm()
    machine.submit_root("A")
    target = machine.target({"digest": "sha256:" + "0" * 64, "length": 1})
    placement = machine.worker._placement_from_entry(
        documents.from_body(
            {**target.prepared_installation["placement"], "placement_id": "machine-dead"},
            pb.Placement,
        ),
        b"",
    )
    placement.device_pin = (0,)
    supervision = machine.worker._new_placement_supervision("machine-dead")
    hosted = HostedPlacement(placement, supervision)
    machine.worker.hosted[placement.placement_id] = hosted
    notes = len(machine.worker.activity)
    machine.worker.prespawns.request(OWNER, "A", target, lambda frame: None)
    (slot,) = machine.worker.prespawns.slots.values()
    assert slot.replica == "machine-dead" and slot.supervision is supervision
    assert slot.started.wait(60)
    executor = supervision.current
    assert executor is not None and executor.alive() and executor.started
    failed = [e.step for e in list(machine.worker.activity)[notes:] if "no prespawn" in e.step]
    assert failed == [], failed
    assert machine.worker.prespawns.claim(INSTALLATION, (0,)) is slot
