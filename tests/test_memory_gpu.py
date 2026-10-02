"""The memory manager on a real card, through the machine path (runs 1458/1459, 1507, 1514).

Real: the Worker on device 0, SubmitMachineExecution, each root's execution unit and GPU
grant, replica activation, real executor processes with Torch on the card, real weights from
a TensorFS store through the real fill plane, and the driver's own free bytes. Fakes: the uv
install (a retained venv whose `.pth` names this checkout's environment), the census derive
and the interface describe. A `Neighbour` process stands in for memory this worker does not
own. Each test holds the shared `gpu.lock` for the whole card.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

import pytest

import signed_claims
from cozy_runtime import canonical_json
from cozy_runtime.internal import (
    accel,
    canonical,
    census_cache,
    child_env,
    package_environment,
    package_installation,
    package_interface,
)
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.worker import package_prepare
from cozy_runtime.internal.worker.attempts import AttemptEngine, AttemptRecord
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.machine_execution_rpc import MachineExecutionRPC
from cozy_runtime.internal.worker.session import Worker, WorkerOptions, read_placement_set
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR

OWNER = "owner"
DEVICE = "0"
PACKAGE = "cozytest/tenant"
RELEASE = "1"
LANE = "bf16"
TENANT = Path(__file__).resolve().parent.parent / "corpus" / "tenant"
LOCKED = b"--index-url https://pypi.org/simple\ntenant==1.0.0 --hash=sha256:" + b"a" * 64 + b"\n"
BURST = '''

class BurstInput(msgspec.Struct, forbid_unknown_fields=True):
    scratch_bytes: int = 0


@app.entrypoint
def burst(ctx: Context, payload: BurstInput, model: TenantModel, tel: Telemetry) -> TouchOutput:
    """Touch the resident model, then hold request-sized device scratch: an activation."""
    import torch

    view = model.for_request(ctx, seed=1)
    checksum = model.touch(view._seed)
    scratch = torch.empty(payload.scratch_bytes, dtype=torch.uint8, device="cuda")
    scratch.fill_(1)
    torch.cuda.synchronize()
    del scratch
    return TouchOutput(checksum=checksum, components=sorted(model.pipe.components))


@app.entrypoint
def cycle(ctx: Context, payload: BurstInput, model: TenantModel, tel: Telemetry) -> TouchOutput:
    """Leave cyclic garbage that holds device memory, old enough that only a full collection
    finds it."""
    import gc

    import torch

    view = model.for_request(ctx, seed=1)
    checksum = model.touch(view._seed)
    held: list = [torch.ones(payload.scratch_bytes, dtype=torch.uint8, device="cuda")]
    held.append(held)
    gc.collect()
    del held
    return TouchOutput(checksum=checksum, components=sorted(model.pipe.components))


from typing import Annotated

from cozy_runtime.author import Shape, invocable


class SegmentInput(msgspec.Struct, forbid_unknown_fields=True):
    seconds: Annotated[int, msgspec.Meta(ge=1, le=4), Shape(frames={1: 24, 2: 48, 3: 72, 4: 96})]
    scratch_per_second: int = 0


@invocable
async def segment(
    ctx: Context, *, payload: SegmentInput, model: TenantModel, tel: Telemetry
) -> TouchOutput:
    """`burst` as H3's `long_form` calls a segment: its length is one level down, in `payload`."""
    import torch

    view = model.for_request(ctx, seed=1)
    checksum = model.touch(view._seed)
    size = payload.seconds * payload.scratch_per_second
    scratch = torch.empty(size, dtype=torch.uint8, device="cuda")
    scratch.fill_(1)
    torch.cuda.synchronize()
    del scratch
    return TouchOutput(checksum=checksum, components=sorted(model.pipe.components))


app.entrypoint(segment)
'''
BINDING = "".join(
    f'\n[bindings."{entrypoint}.models.model"]\nmodel = "cozytest/tenant"\nrelease = "1"\n'
    'lane = "bf16"\n'
    for entrypoint in ("burst", "cycle", "segment")
)


def _host() -> str:
    """Why this host cannot run the arms, or ""."""
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


#: Asked only when the run opted in (`--real-gpu`), never at import: importing reaches no driver.
needs_card = pytest.mark.real_gpu


def free() -> int:
    memory = accel.device_memory(DEVICE, accel.host_backend_family())
    assert memory.state == "measured", memory
    return memory.free_bytes


# ------------------------------------------------------------------------------ weights


@dataclass(frozen=True, slots=True)
class Store:
    root: Path
    weight_bytes: int
    manifests: dict[str, str]


def write_store(root: Path, *, blocks: int, models: tuple[str, ...]) -> Store:
    """One synthetic `tfs` checkpoint published as `cozytest/<model>@1` for each model, over
    the SAME tensor blobs; each differs by one config marker, so each is its own
    construction (its own executor and bytes on the card) for no second copy on disk."""
    import tensorfs

    store, work = root / "store", root / "work"
    work.mkdir(parents=True, exist_ok=True)

    def tfs(*args: str) -> str:
        return subprocess.run(
            ["nice", "-n", "19", "tfs", *args], capture_output=True, text=True, check=True
        ).stdout

    written = tfs("checkpoint", "write", str(store), "--plain", "--blocks", str(blocks))
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
    header["assets"] = []
    (work / "order.json").write_text(
        json.dumps([[name, row[0]] for name, rows in header["components"] for row in rows])
    )
    (work / "entries.json").write_text(json.dumps([["model.cozytensors", "cozytensors"]]))
    manifests: dict[str, str] = {}
    for index, model in enumerate(models):
        marked = json.dumps({**config, "twin": index}, sort_keys=True, separators=(",", ":"))
        (work / f"header-{model}.json").write_text(
            json.dumps({**header, "configs": [["unet", marked]]})
        )
        reproduced = tfs(
            "checkpoint",
            "reproduce",
            str(store),
            str(work / f"header-{model}.json"),
            str(work / "entries.json"),
            "--order",
            str(work / "order.json"),
        )
        manifest, length = re.search(  # type: ignore[union-attr]
            r"manifest\s+(sha256:[0-9a-f]{64}) length=(\d+)", reproduced
        ).groups()
        opened = tensorfs.Store.open(str(store))
        operation = opened.begin_operation(f"op-{model}", "cozytest", model)
        operation.hold_manifest(manifest, int(length))
        operation.commit_release(None, RELEASE, LANE, manifest, int(length))
        manifests[model] = manifest
    opened = tensorfs.Store.open(str(store))
    first = manifests[models[0]]
    weight_bytes = sum(int(row["length"]) for row in opened.walk_cozytensors(first))
    return Store(root=store, weight_bytes=weight_bytes, manifests=manifests)


# ------------------------------------------------------------------------------ a neighbour

_NEIGHBOUR = """
import sys, torch
torch.cuda.init()
print("context", torch.cuda.mem_get_info()[0], flush=True)
target = int(sys.argv[1])
held = []
while (free := torch.cuda.mem_get_info()[0]) > target:
    held.append(torch.empty(min(free - target, 64 << 20), dtype=torch.uint8, device="cuda"))
torch.cuda.synchronize()
print("held", torch.cuda.mem_get_info()[0], flush=True)
sys.stdin.readline()
"""


class Neighbour:
    """Memory this worker does not own: a process that allocates until the driver reports
    `target` bytes free, and holds them until closed."""

    def __init__(self, target: int) -> None:
        before = free()
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
        self.context_bytes = before - int(line[1])
        line = self.process.stdout.readline().split()
        assert line and line[0] == "held", line
        self.free_after = int(line[1])

    def close(self) -> None:
        with contextlib.suppress(OSError):
            assert self.process.stdin is not None
            self.process.stdin.write("done\n")
            self.process.stdin.flush()
        self.process.wait()


# ------------------------------------------------------------------------------ the machine


class _Context:
    def abort(self, code: object, detail: str) -> NoReturn:
        raise AssertionError(f"{code}: {detail}")

    def is_active(self) -> bool:
        return True

    def set_trailing_metadata(self, metadata: tuple[tuple[str, str], ...]) -> None:
        pass

    def add_callback(self, _callback: Callable[[], None]) -> bool:
        return True


class Card:
    """One worker on device 0 serving the tenant package through the machine path."""

    def __init__(self, root: Path, store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
        self.store = store
        # A request id starts alphanumeric; a temporary name may start with "_".
        self.tag = "c" + root.name.rsplit(".", 1)[-1]
        self.install_root = root / "installs"
        #: request -> the plan's placement rung, and the replica that served it
        self.plans: dict[str, str] = {}
        self.replicas: dict[str, str] = {}
        project = root / "project"
        project.mkdir()
        source = (TENANT / "tiny_tenant.py").read_text() + BURST
        (project / "gpu_tenant.py").write_text(source)
        (project / "package.toml").write_text(
            (TENANT / "package.toml").read_text().replace("tiny_tenant:app", "gpu_tenant:app")
            + BINDING
        )
        self.interface = package_interface.canonical_bytes(
            package_interface.build(discover(project))
        )
        generation = root / "generation"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(generation)],
            check=True,
            capture_output=True,
        )
        (site,) = generation.glob("lib/python*/site-packages")
        (site / "tenant-1.0.0.dist-info").mkdir()
        (site / "tenant-1.0.0.dist-info" / "METADATA").write_text(
            "Metadata-Version: 2.3\nName: tenant\nVersion: 1.0.0\n"
        )
        (site / "gpu_tenant.py").write_text(source)
        # This checkout's environment (Torch, TensorFS, the Runtime under test), as a worker
        # image's closure would be; a `.pth` survives the executor's isolated mode.
        purelib = sysconfig.get_paths()["purelib"]
        (site / "checkout.pth").write_text(f"import site; site.addsitedir({purelib!r})\n")
        installed = package_installation.retain_environment(
            self.install_root, generation / "bin/python", package=PACKAGE, release=RELEASE
        )

        def install(request: pb.PreparePackageSetRequest, **seams: Any) -> Any:
            models = documents.read(request.download_delegation, pb.DownloadDelegation)["models"]
            return package_prepare._prepare_published(
                package_name=PACKAGE,
                release="1.0.0",
                locked=package_environment.read_locked_requirements(LOCKED),
                models=package_prepare.selections(models),
                artifact_cache=seams["artifact_cache"],
                tensorfs_root=seams["tensorfs_root"],
                install_root=seams["install_root"],
                python=installed.python,
                interface=lambda _installed, _distribution: self.interface,
                verified=seams["verified"],
                job_plan_root=seams["job_plan_root"],
                base=None,
                installed_environment=installed,
                materialized=seams["materialized"],
            )

        def census(*_: Any) -> Any:
            return lambda wanted: {
                path: census_cache.Census(("text_encoder", "unet", "vae"), (), True, "")
                for path in wanted
            }

        monkeypatch.setattr(package_prepare, "prepare_package_set", install)
        enter = AttemptEngine.enter

        def entered(engine: AttemptEngine, attempt: AttemptRecord, **options: Any) -> None:
            enter(engine, attempt, **options)
            if attempt.plan is not None:
                name = attempt.request_id.removeprefix(self.tag + "-")
                self.plans[name] = attempt.plan.placement
                self.replicas[name] = attempt.placement_id

        monkeypatch.setattr(AttemptEngine, "enter", entered)
        #: every step the worker narrated, whole (its activity keeps the last 64, capped)
        self.notes: list[str] = []
        note = Worker.note

        def noted(worker: Worker, kind: str, step: str) -> None:
            self.notes.append(step)
            note(worker, kind, step)

        monkeypatch.setattr(Worker, "note", noted)
        monkeypatch.setattr(package_prepare, "_census", census)
        base = {
            k: v for k, v in os.environ.items() if not child_env.erased(k) and k != "PYTHONPATH"
        }
        self.worker = Worker(
            replace(
                RuntimeConfig(
                    cozy_home=root / "home",
                    credentials=Credentials(),
                    child_base_env=tuple(sorted(base.items())),
                ),
                record_owner_public_key=signed_claims.PUBLIC_KEY,
            ),
            WorkerOptions(
                **signed_claims.IDENTITY,
                root=root / "worker",
                tensorfs_root=store.root,
                install_root=self.install_root,
                artifact_cache=root / "artifacts",
                devices=DEVICE,
            ),
            InMemoryControlHost(),
        )
        describe = self.worker._describe_installed

        def described(installed: Any, distribution: str, **options: Any) -> bytes:
            if distribution == "tenant":
                return self.interface
            return describe(installed, distribution, **options)

        monkeypatch.setattr(self.worker, "_describe_installed", described)
        self.claim = signed_claims.claim(OWNER)
        assert self.worker.accept_claim(self.claim, lambda frame: None)[0] >= 1
        assert self.worker.executions is not None
        self.executions = self.worker.executions
        self.rpc = MachineExecutionRPC(self.worker)
        self.installed: pb.InstalledPackage | None = None
        self.prepared: dict[tuple[str, str], pb.DesiredPlacementSet] = {}

    def prepare(self, slot: str, model: str) -> pb.DesiredPlacementSet:
        """The release prepared for one model slot bound to `model`'s weights."""
        if (slot, model) not in self.prepared:
            row = {
                "lane": LANE,
                "manifest": self.store.manifests[model],
                "model": "cozytest/" + model,
                "package": PACKAGE,
                "release": RELEASE,
                "slot": slot,
            }
            result = self.worker.prepare_package_set(
                pb.PreparePackageSetRequest(
                    download_delegation=canonical.write(
                        {
                            "format": "cozy.worker.v1.DownloadDelegation/1",
                            "packages": [{"package": PACKAGE, "release": "1.0.0"}],
                            "models": [row],
                        }
                    ),
                    install_root=str(self.install_root),
                    application="gpu_tenant:app",
                    locked_requirements=LOCKED,
                    package_interface=self.interface,
                )
            )
            self.installed = result.installed_package
            self.prepared[(slot, model)] = result.placement_set
        return self.prepared[(slot, model)]

    def call(self, name: str, entrypoint: str, model: str, **payload: Any) -> dict[str, Any]:
        """One serving root: `entrypoint` on `model`'s weights, run to its terminal. The
        execution journal lives beside the module's shared store, so ids are per card."""
        request = f"{self.tag}-{name}"
        state = self.prepare(f"{entrypoint}.models.model", model)
        (placement,) = read_placement_set(state)
        row = next(e for e in placement.entrypoints if e.name == entrypoint)
        assert self.installed is not None
        installation = self.installed.installation_id
        capture, capture_digest = documents.identity(
            pb.MachineExecutionCapture(
                root_installation_id=installation, installed_packages=[self.installed]
            )
        )
        body = canonical_json.encode(payload)
        digest = documents.spell(documents.digest_of(body))
        raw, spec_digest = documents.identity(
            pb.InvocationSpec(
                installation_id=installation,
                payload_digest=digest,
                inputs=[
                    pb.InputBinding(
                        input_id="payload",
                        digest=digest,
                        length=len(body),
                        kind_mime="application/json",
                    )
                ],
                deadline_unix_ms=int(time.time() * 1000) + 600_000,
                serving=pb.ServingInvocationSpec(
                    entrypoint_binding_digest=documents.spell(row.entrypoint_binding_digest),
                    attempt_binding_id=documents.spell(row.entrypoint_binding_digest),
                    bindings_digest=documents.spell(placement.bindings_digest),
                ),
            )
        )
        desired = pb.DesiredWorkerState(
            placement_set=state, revision=1, wire_minor=WIRE_MINOR, posture=pb.POSTURE_ACCEPTING
        )
        self.rpc.SubmitMachineExecution(
            pb.MachineExecutionSubmit(
                claim=self.claim,
                submission_id=request,
                capture_digest=capture_digest,
                capture_canonical_bytes=capture,
                offer=pb.AttemptOffer(
                    request_id=request,
                    attempt_ordinal=1,
                    placement_id=placement.placement_id,
                    invocation_spec_canonical_bytes=raw,
                    invocation_spec_digest=spec_digest,
                    grant=pb.DeliveryGrant(
                        invocation_spec_digest=spec_digest,
                        inputs=[
                            pb.InputAccess(
                                input_id="payload",
                                url="data:application/json;base64,"
                                + base64.b64encode(body).decode(),
                            )
                        ],
                    ),
                ),
                prepared_state=desired,
                payload_canonical_bytes=body,
                expected_execution_workspace_id=self.executions.workspace_id,
            ),
            _Context(),
        )
        bound = time.monotonic() + 600  # a hang bound on a loaded shared box, not a budget
        while self.worker.execution_status(self.claim, request).state not in (
            "succeeded",
            "failed",
            "canceled",
        ):
            assert time.monotonic() < bound, self.steps()[-20:]
            time.sleep(0.05)
        outcome = self.worker.collect_execution(self.claim, request)
        return documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)

    def steps(self) -> list[str]:
        return list(self.notes)

    def vacated(self) -> list[tuple[str, str]]:
        """(victim, for whom) per eviction the memory manager made."""
        pattern = re.compile(r"'([^']+)' \(epoch \d+ pid \d+\) vacated for '([^']+)'")
        return [(m.group(1), m.group(2)) for s in self.steps() if (m := pattern.search(s))]

    def epochs(self) -> dict[str, int]:
        live = {p: h.supervision.current for p, h in self.worker.hosted.items()}
        return {p: e.epoch for p, e in live.items() if e is not None}


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Store:
    if why := _host():
        pytest.skip(why)
    return write_store(tmp_path_factory.mktemp("gpu-store"), blocks=24, models=("p", "q", "r"))


@contextlib.contextmanager
def serving(store: Store, monkeypatch: pytest.MonkeyPatch) -> Iterator[Card]:
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-card.", dir="/tmp") as root:
        made = Card(Path(root), store, monkeypatch)
        try:
            yield made
        finally:
            made.worker.shutdown()


@pytest.fixture
def card(store: Store, monkeypatch: pytest.MonkeyPatch) -> Iterator[Card]:
    with serving(store, monkeypatch) as made:
        yield made


def succeeded(body: dict[str, Any]) -> bool:
    return int(body.get("status", 0)) == pb.OUTCOME_STATUS_SUCCEEDED


# ------------------------------------------------------------------------------ the arms


@needs_card
def test_models_that_fit_together_stay_resident_once_measured(card: Card) -> None:
    """1507: two tenants that fit together. Once each has measured its call, alternating
    calls move nothing: every plan is all-resident and nobody is evicted."""
    for index in range(1, 4):
        assert succeeded(card.call(f"p-{index}", "touch", "p", seed=index))
        assert succeeded(card.call(f"q-{index}", "retouch", "q", seed=index))
        if index == 1:
            measured = len(card.vacated())
    assert card.vacated()[measured:] == [], card.vacated()
    assert {card.plans[f"{t}-{i}"] for t in "pq" for i in (2, 3)} == {"all_resident"}, card.plans
    assert set(card.epochs().values()) == {1}


@needs_card
def test_a_call_that_needs_room_evicts_the_idle_neighbour_instead_of_running_out(
    card: Card, store: Store
) -> None:
    """1514: P's burst needs more than fits beside Q. Measured once, every later burst
    evicts Q first and runs; nothing is killed and nothing runs out of memory."""
    assert succeeded(card.call("p-touch", "touch", "p"))
    assert succeeded(card.call("q-touch", "retouch", "q"))
    scratch = free() - store.weight_bytes // 2
    assert scratch > store.weight_bytes, (scratch, store.weight_bytes)
    for index in range(1, 3):
        body = card.call(f"p-burst-{index}", "burst", "p", scratch_bytes=scratch)
        assert succeeded(body), body
        assert succeeded(card.call(f"q-{index}", "retouch", "q"))
    burst = card.replicas["p-burst-1"]
    evicted_for_burst = [victim for victim, whom in card.vacated() if whom == burst]
    assert len(evicted_for_burst) >= 2, card.vacated()
    assert set(card.epochs().values()) == {1}, "vacated, not killed"
    assert not any("out of memory" in step.lower() for step in card.steps())


@needs_card
def test_a_longer_segment_is_not_admitted_on_a_shorter_ones_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run 1560: H3's 12 s segment ran on its 6 s segment's measured peak, because the length
    sat in an invocable's `payload` and both were one shape. A 4 s call after a 1 s one of the
    same entrypoint is unmeasured, so the idle neighbour goes instead of the call running out.
    Small weights, so both tenants stay resident beside another process's memory."""
    small = write_store(tmp_path, blocks=2, models=("p", "q"))
    with serving(small, monkeypatch) as card:
        assert succeeded(card.call("q-touch", "touch", "q"))
        assert succeeded(card.call("s-1", "segment", "p", payload={"seconds": 1}))
        assert succeeded(card.call("q-again", "touch", "q"))
        # Beside Q it does not fit; with Q vacated it does.
        per_second = (free() + small.weight_bytes // 2) // 4
        body = card.call(
            "s-4", "segment", "p", payload={"seconds": 4, "scratch_per_second": per_second}
        )
        assert succeeded(body), body
        whom = card.replicas["s-4"]
        assert card.replicas["q-again"] in [v for v, by in card.vacated() if by == whom]
        assert set(card.epochs().values()) == {1}


@needs_card
def test_a_load_gets_room_from_idle_tenants_of_other_executors(card: Card, store: Store) -> None:
    """1458/1459: memory this worker does not own leaves R's load room only if idle tenants
    of other executors give theirs back. They are vacated, not killed, and R serves."""
    assert succeeded(card.call("p-1", "touch", "p"))
    assert succeeded(card.call("q-1", "retouch", "q"))
    neighbour = Neighbour(store.weight_bytes * 3 // 2)
    try:
        assert succeeded(card.call("r-1", "burst", "r", scratch_bytes=1 << 20))
    finally:
        neighbour.close()
    whom = card.replicas["r-1"]
    victims = [victim for victim, for_whom in card.vacated() if for_whom == whom]
    assert victims and set(victims) <= {card.replicas["p-1"], card.replicas["q-1"]}, card.vacated()
    assert set(card.epochs().values()) == {1}


@needs_card
def test_a_capacity_failure_is_not_inherited_and_poisons_no_binding(
    card: Card, store: Store
) -> None:
    """Memory held outside the worker leaves no room for P's burst even alone: that call
    fails on the device. Once the memory is back the same call serves, and so does P."""
    assert succeeded(card.call("p-1", "touch", "p"))
    scratch = free() - (256 << 20)
    neighbour = Neighbour(store.weight_bytes)
    try:
        refused = card.call("p-big-1", "burst", "p", scratch_bytes=scratch)
    finally:
        neighbour.close()
    assert not succeeded(refused), refused
    # The same call, sized to fit the card alone beside a fresh executor's context.
    body = card.call("p-big-2", "burst", "p", scratch_bytes=free() - 3 * store.weight_bytes)
    assert succeeded(body), body
    assert succeeded(card.call("p-2", "touch", "p"))


@needs_card
def test_cyclic_garbage_holding_device_memory_costs_no_executor(
    card: Card, caplog: pytest.LogCaptureFixture
) -> None:
    """A package that leaves a reference cycle around a device tensor has leaked nothing: the
    terminal ledger collects and reads again, and the same executor keeps serving."""
    assert succeeded(card.call("p-1", "touch", "p"))
    with caplog.at_level("WARNING", logger="cozy_runtime.internal.worker.attempts"):
        assert succeeded(card.call("p-cycle", "cycle", "p", scratch_bytes=8 << 20))
    assert f"{8 << 20} B of device memory were held by cyclic garbage" in caplog.text
    assert succeeded(card.call("p-2", "touch", "p"))
    assert set(card.epochs().values()) == {1}
