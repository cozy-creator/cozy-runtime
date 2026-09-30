"""The startup warm (h3a-087): a serving installation's executor imports torch, the Runtime and
its package before its GPU grant, whoever holds the card meanwhile, and the grant adopts that
process and only initializes its device.

Real Worker, real durable journal, a real package environment the worker opens like any
installed one, and real executor processes, on no GPU: the envelope's cards are GPU UUIDs no
driver knows, so every executor is sealed to no visible device, and the device start that a
card would initialize refuses `accelerator_unavailable` here. The one fake is
`accel.device_memory`, the hardware memory seam, which reads those cards as measured. A
download still in flight is a real TensorFS pull from an in-process origin whose object GETs
are held open.
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
from cozy_runtime.internal import accel, package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor_commands import Probe
from cozy_runtime.internal.worker import prespawn
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.child import Executor
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.plan import ModelBinding as WorkerSlot
from cozy_runtime.internal.worker.session import HostedPlacement, Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_cross_version_seam import _installation
from test_device_lanes import _config
from test_end_to_end import NO_EXECUTOR
from test_gpu_scheduler import MARCO, Machine
from test_model_runtime_closure import _ASSET, _HEADER, _snapshot

pytestmark = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")

OWNER = "owner"
INSTALLATION = "release-startup-warm-fixture"
#: Cards no driver knows: an executor sealed to one sees no device and holds none.
CARDS = tuple(f"GPU-00000000-0000-0000-0000-{n:012d}" for n in range(4))
GiB = 1 << 30


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
                threads=1,
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
        """A CPU orchestration root, already running its Python, as `Machine.root` holds one."""
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
        attempt = AttemptRecord(name, 1, digest, {}, state="running")
        self.worker.engine.history[attempt.key()] = attempt
        self.worker.engine.live[name] = attempt
        self.worker.execution(OWNER, name)

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

    def direct_root(self, name: str, checkpoint: dict[str, Any] | None = None) -> None:
        """A serving root as Creator submits a rented direct root once the Host's preparation
        answers: its template names the installed fixture, and no call asked for it before.
        `checkpoint` is its model's landed manifest; none names weights never landed."""
        checkpoint = checkpoint or {"digest": "sha256:" + "0" * 64, "length": 1}
        placement = {
            **self.target(checkpoint).prepared_installation["placement"],
            "package": {"package": "t/m", "release": "1"},
            "models": [{"id": "model", "repo": "proof/model", "manifest": checkpoint}],
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
def warm(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Warm]]:
    measured = accel.DeviceMemory("measured", 80 * GiB, 80 * GiB)
    monkeypatch.setattr(accel, "device_memory", lambda entry, kind: measured)
    made: list[Warm] = []
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-warm.", dir="/tmp") as raw:

        def make(cards: int = 1) -> Warm:
            made.append(Warm(Path(raw) / str(len(made)), ",".join(CARDS[:cards])))
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


def _imported(executor: Executor) -> bool:
    """Torch and the package are loaded in the process, and its Runtime holds no device:
    `probe` reports torch only once a device start bound it."""
    maps = Path(f"/proc/{executor.pid}/maps").read_text()
    return "libtorch" in maps and executor.call(Probe())["torch"] is False


def _binding(machine: Warm) -> DeclaredBinding:
    """The fixture's binding as acquisition resolves it: its application and the interface
    bytes it published for the executor."""
    raw = _interface()
    path = Path(machine.worker.options.artifact_cache or "") / "selections" / INSTALLATION
    package_interface.publish(path / "package-interface.json", raw)
    return DeclaredBinding(
        entrypoint_binding_digest="sha256:" + hashlib.sha256(b"warm-binding").hexdigest(),
        application=package_interface.parse(raw, "fixture").application,
        interface_path=str(path / "package-interface.json"),
        entrypoint="marco",
        model_class="M",
        model_binding_path="marco.models.model",
        model_parameter_name="model",
        release="r",
        logical_weight_bytes=GiB,
        models=(
            WorkerSlot(
                *("M", "marco.models.model", "model", "", "", ""),
                components=("unet",),
                snapshots={},
            ),
        ),
    )


def _replica(machine: Warm, slot: prespawn.Slot | None, name: str) -> HostedPlacement:
    """A replica of the fixture on card 0 in the slot's supervision, or a fresh one: what a
    grant's `_activate_replica` binds, with the bindings its acquisition would have read."""
    target = machine.target({"digest": "sha256:" + "0" * 64, "length": 1})
    placement = machine.worker._placement_from_entry(
        documents.from_body(
            {**target.prepared_installation["placement"], "placement_id": name}, pb.Placement
        ),
        b"",
    )
    placement.device_pin = (0,)
    supervision = (
        slot.supervision if slot is not None else machine.worker._new_placement_supervision(name)
    )
    installed = machine.worker.options.install_root / "installations" / INSTALLATION  # type: ignore[operator]
    supervision.use_environment(str(installed / "venv" / "bin" / "python"), INSTALLATION)
    binding = _binding(machine)
    hosted = HostedPlacement(placement, supervision, {binding.entrypoint_binding_digest: binding})
    machine.worker.hosted[name] = hosted
    with machine.worker.control_lock:
        assert machine.worker._assign_lane(placement, hosted.bindings, hosted.supervision)
    return hosted


def test_a_slow_download_overlaps_the_executor_imports(warm: Callable[..., Warm]) -> None:
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
            lambda: (p := machine.phases("A")).get("Preparing model executor") and p,
            "the prespawned executor's imports",
        )
        assert not future.done() and origin.landed == 0.0
        prepared = phases["Preparing model executor"]
        assert prepared["completed"] and prepared["ordinals"] == [0], prepared
        for leg in ("torch_import", "runtime_imports", "package_import"):
            assert leg in prepared["detail"], prepared
        origin.release.set()
        assert future.result(120).placement["installation_id"] == INSTALLATION
        assert origin.landed * 1000 > prepared["started_unix_ms"] + prepared["elapsed_ms"]
        # Imported, sealed to its card, holding no device, and preferred by the grant.
        (slot,) = machine.worker.prespawns.slots.values()
        executor = slot.executor
        assert executor is not None and executor.alive() and not executor.started
        assert executor.sealed_devices == CARDS[0] and _imported(executor)
        assert machine.worker._warm((INSTALLATION, "any")) == (0,)
        if prespawn._RESTORABLE:
            assert os.getpriority(os.PRIO_PROCESS, executor.pid) == 19
    finally:
        origin.close()


def test_a_waiting_root_imports_while_another_root_holds_its_card_and_its_grant_adopts_it(
    warm: Callable[..., Warm],
) -> None:
    """Root A's call holds the only card. D, a direct serving root of the installed package,
    waits for it and meanwhile its executor spawns and imports on that card's seal, with no
    device touched. When A ends, D is granted the card and its replica adopts that very
    process: no second spawn, and its start is the device start alone."""
    machine = warm()
    origin = Origin(Path(tempfile.mkdtemp(prefix="cz-origin.", dir="/tmp")))
    try:
        machine.captured_root("A", origin)
        # D's weights land first (the Host's preparation), so its acquisition finds them.
        call, request = _call("A")
        serving = machine.worker.machine_calls.serving  # type: ignore[union-attr]
        origin.release.set()
        target = machine.target(origin.checkpoint)
        landing = serving._shared(OWNER, call, target, request, {}, speculative=False)
        assert landing is not None and landing.result(120)
        (landed,) = machine.worker.prespawns.slots.values()
        machine.worker.prespawns.forget(INSTALLATION)
        _wait(lambda: landed.supervision._closed, "the landing's own prespawn retired")

        a1 = machine.child("A", "qwen")
        machine.tick()
        assert machine.granted(a1) == [0]
        machine.direct_root("D", origin.checkpoint)
        slot = _wait(
            lambda: machine.worker.prespawns.slots.get((INSTALLATION, (0,))), "D's prespawn"
        )
        assert slot.roots == {"D"} and slot.started.wait(300) and slot.ready
        executor = slot.executor
        assert executor is not None and executor.alive() and _imported(executor)
        assert machine.granted("D") == [] and machine.worker.gpu.view()["leases"]["A"] == [0]
        prepared = _wait(
            lambda: machine.phases("D").get("Preparing model executor"), "D's prespawn phase"
        )
        assert prepared["completed"] and "package_import" in prepared["detail"], prepared

        notes = len(machine.worker.activity)
        machine.finish(a1)
        machine.finish("A")
        machine.tick()
        assert machine.granted("D") == [0]
        assert slot.supervision.spawns == 1, "the grant spawned another process"
        steps = [e.step for e in list(machine.worker.activity)[notes:]]
        assert not [s for s in steps if "spawning the device executor" in s], steps
        assert [s for s in steps if f"starting epoch {executor.epoch} pid {executor.pid}" in s]
        # On a card the device start initializes it; here none is visible, and nothing else
        # was left to do.
        refused = [s for s in steps if "start REFUSED" in s]
        assert refused and "accelerator_unavailable" in refused[0], steps
        started = machine.phases("D")["Starting model executor"]
        assert started["started_unix_ms"] >= prepared["started_unix_ms"] + prepared["elapsed_ms"]
    finally:
        origin.close()


def test_the_adopting_start_is_the_device_start_alone(warm: Callable[..., Warm]) -> None:
    """A grant that arrives while the process is still importing waits out only those imports
    (it would pay them itself), takes that very process at the worker's priority, and sends
    it one start: no spawn and no import is repeated. A replica with no prespawn pays both."""
    machine = warm()
    machine.submit_root("A")
    target = machine.target({"digest": "sha256:" + "0" * 64, "length": 1})
    emitted: list[dict[str, Any]] = []
    machine.worker.prespawns.request(OWNER, "A", target, emitted.append)
    slot = machine.worker.prespawns.claim(INSTALLATION, (0,))
    assert slot is not None and not slot.started.is_set(), "the claim came mid-import"
    machine.worker.prespawns.settle(slot, "machine-adopter")
    executor = slot.executor
    assert executor is not None and slot.ready and executor.alive() and not executor.started
    assert os.getpriority(os.PRIO_PROCESS, executor.pid) == os.getpriority(os.PRIO_PROCESS, 0)
    assert [frame["fields"]["phase"] for frame in emitted] == ["Preparing model executor"]

    hosted = _replica(machine, slot, "machine-adopter")
    notes = len(machine.worker.activity)
    latch, _ = machine.worker._prepare_generation(machine.worker._tenant("machine-adopter"))
    steps = [e.step for e in list(machine.worker.activity)[notes:]]
    assert hosted.supervision.spawns == 1 and executor.epoch == 1
    assert [s for s in steps if f"starting epoch 1 pid {executor.pid}" in s], steps
    assert not [s for s in steps if "spawning the device executor" in s], steps
    # The device start is all that was sent; with no card visible it refuses there.
    assert latch and [s for s in steps if "accelerator_unavailable" in s], steps

    fresh = _replica(machine, None, "machine-fresh")
    notes = len(machine.worker.activity)
    machine.worker._prepare_generation(machine.worker._tenant("machine-fresh"))
    steps = [e.step for e in list(machine.worker.activity)[notes:]]
    assert fresh.supervision.spawns == 1
    assert [s for s in steps if "spawning the device executor" in s], steps


def test_a_mismatched_or_failed_prespawn_is_a_missed_saving(warm: Callable[..., Warm]) -> None:
    """A grant on other cards than the prespawn's leaves it: its process is reclaimed and the
    replica spawns its own. A prespawn that cannot start says why and leaves nothing."""
    machine = warm(2)
    machine.submit_root("A")
    target = machine.target({"digest": "sha256:" + "0" * 64, "length": 1})
    machine.worker.prespawns.request(OWNER, "A", target, lambda frame: None)
    (slot,) = machine.worker.prespawns.slots.values()
    assert slot.ordinals == (0,) and slot.started.wait(300) and slot.ready
    executor = slot.executor
    assert executor is not None and executor.alive()
    assert machine.worker.prespawns.claim(INSTALLATION, (1,)) is None
    _wait(
        lambda: not executor.alive() and slot.supervision.current is None,
        "the mismatched prespawn reclaimed",
    )
    assert machine.worker.prespawns.slots == {}

    emitted: list[dict[str, Any]] = []
    machine.worker.prespawns.request(
        OWNER, "A", replace(target, installation_id="absent-installation"), emitted.append
    )
    _wait(lambda: emitted, "the failed prespawn's phase")
    (row,) = emitted
    assert row["fields"]["completed"] is False, row
    assert "package_installation_absent" in row["fields"]["detail"], row
    _wait(lambda: not machine.worker.prespawns.slots, "the failed slot dropped")


def test_a_prespawn_waits_for_cards_its_grant_will_take_and_otherwise_starts_nothing(
    warm: Callable[..., Warm],
) -> None:
    """On two cards, a width-1 root is foreseen on a free one; with both held by other roots
    its grant depends on who leaves first, so nothing starts. On one card the grant is that
    card whoever holds it."""
    machine = warm(2)
    for name in ("A", "B", "C"):
        machine.submit_root(name)
    a1, b1 = machine.child("A", "qwen"), machine.child("B", "qwen")
    machine.tick()
    assert sorted(machine.granted(a1) + machine.granted(b1)) == [0, 1]
    target = machine.target({"digest": "sha256:" + "0" * 64, "length": 1})
    machine.worker.prespawns.request(OWNER, "C", target, lambda frame: None)
    assert machine.worker.prespawns.slots == {}
    machine.finish(b1)
    machine.finish("B")
    machine.tick()
    free = machine.granted(b1)
    machine.worker.prespawns.request(OWNER, "C", target, lambda frame: None)
    (slot,) = machine.worker.prespawns.slots.values()
    assert list(slot.ordinals) == free and slot.roots == {"C"}
    assert slot.started.wait(300) and slot.ready


def test_a_host_started_warm_is_adopted_by_the_direct_root_granted_its_card(
    warm: Callable[..., Warm],
) -> None:
    """h3a-089: the Host's preparation starts the installation's executor while it lands the
    weights of a root nobody has submitted yet; its rows wait in the slot. The direct root
    submitted afterwards adopts it while waiting, is granted its card, takes that very
    process, and its journal carries the phase with the times it actually ran."""
    machine = warm()
    prespawns = machine.worker.prespawns
    prespawns.request_landing(INSTALLATION, _interface(), [{"path": "marco.models.model"}])
    (slot,) = prespawns.slots.values()
    assert not slot.roots and slot.ordinals == (0,)
    assert slot.started.wait(300) and slot.ready
    assert [frame["fields"]["phase"] for frame in slot.frames] == ["Preparing model executor"]
    # A second landing preparation of the same installation starts nothing more.
    prespawns.request_landing(INSTALLATION, _interface(), [{"path": "marco.models.model"}])
    assert list(prespawns.slots.values()) == [slot]
    executor = slot.executor
    assert executor is not None and executor.alive()

    submitted = time.time()
    machine.direct_root("D")
    machine.tick()
    assert machine.granted("D") == [0]
    assert slot.roots == {"D"} and prespawns.slots == {} and slot.frames == []
    assert slot.supervision.spawns == 1
    prepared = machine.phases("D")["Preparing model executor"]
    assert prepared["completed"] and prepared["ordinals"] == [0]
    assert prepared["started_unix_ms"] + prepared["elapsed_ms"] < submitted * 1000


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
    assert (INSTALLATION, (0,)) not in prespawns.slots and not slot.roots
    _wait(
        lambda: not executor.alive() and slot.supervision.current is None,
        "the unadopted prespawn reclaimed off B's card",
    )


def test_a_dead_replica_starts_its_successor_while_its_root_waits(
    warm: Callable[..., Warm],
) -> None:
    """Run 1516: a dead replica's successor starts early in the replica's own slot, so the
    grant revives it without a spawn."""
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
    assert slot.started.wait(300) and slot.ready
    executor = supervision.current
    assert executor is not None and executor.alive() and not executor.started
    failed = [e.step for e in list(machine.worker.activity)[notes:] if "no prespawn" in e.step]
    assert failed == [], failed
    assert machine.worker.prespawns.claim(INSTALLATION, (0,)) is slot


def test_an_executor_before_import_only_starts_is_only_spawned(
    warm: Callable[..., Warm],
) -> None:
    """Version skew: an installed package locked to a Runtime before import-only starts gets
    its process spawned ahead of the grant and nothing more. Sent the start it does not know,
    it would initialize its device; its grant sends it the ordinary one."""
    installation, _wheel, _lock, version = _installation("0.18.67", "==0.18.67")
    machine = warm()
    installed = machine.worker.options.install_root / "installations" / installation  # type: ignore[operator]
    installed.symlink_to(
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "cozy-runtime-tests/cross-version-seam/installs/installations"
        / installation
    )
    prespawns = machine.worker.prespawns
    prespawns.request_landing(installation, _interface(), [{"path": "marco.models.model"}])
    (slot,) = prespawns.slots.values()
    assert slot.started.wait(300) and slot.ready, slot.frames
    executor = slot.executor
    assert executor is not None and executor.alive()
    assert executor.hello["runtime_version"] == version
    assert "import_only" not in executor.hello.get("memory", ())
    (row,) = slot.frames
    assert row["fields"]["detail"] == f"executor cozy-runtime {version} imports at its grant"
    assert executor.call(Probe())["torch"] is False
    assert prespawns.claim(installation, (0,)) is slot
    prespawns.settle(slot, "machine-older")
    assert slot.supervision.current is executor and executor.alive()
