"""The worker at HEAD serves the executors of older Runtimes, and a package locked to an older
Runtime runs the machine's own.

An installed package's executor runs in the package's own environment and speaks the seam to
whatever worker the pod runs. Its lock freezes its dependencies, not its Runtime: a package
that declares a lower bound gets the machine's Runtime, and one that pins a Runtime exactly
(like every environment installed before this rule) keeps that executor. Runtime 0.18.69 broke
SDXL and Anima conversions because its worker refused a record an older peer wrote. Here the
worker is this checkout and every executor is a real PyPI release.

Each Runtime is locked the way a package locked it on release day (`--exclude-newer` at its
upload) and installed by the product installer into `CACHE`, with its lock and the package
wheel, so a rerun installs nothing and needs no network; delete that directory to rebuild.
PyPI unreachable on a cold cache is a skip. Each worker session gets its own install root with
the cached installation linked in, and prepares it through its ordinary reuse path.
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import grpc
import msgspec
import pytest
from packaging.version import Version

import signed_claims
from cozy_runtime import canonical_json
from cozy_runtime.internal import child_env, derive_child, package_installation
from cozy_runtime.internal.config import Credentials, RuntimeConfig, package_install_environment
from cozy_runtime.internal.worker import activity
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from local_owner import LocalRecordOwner, LocalRequest
from test_end_to_end import NO_EXECUTOR, _development, _wheel_rows

#: The executor floor (`child.require_executor_protocol`), the Runtime SDXL 2.3.23 pins, and the
#: Runtime the owner's H3/Qwen environments pin.
VERSIONS = ("0.18.51", "0.18.67", "0.18.69")

#: (the Runtime a lock holds, the package's own requirement on it). An exact pin keeps that
#: executor; the owner's H3/Qwen packages declare `>=0.18.67` and so run the machine's own.
CASES = (*((version, f"=={version}") for version in VERSIONS), ("0.18.67", ">=0.18.67"))

CACHE = (
    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    / "cozy-runtime-tests"
    / "cross-version-seam"
)

#: The interpreter cached environments are built on; it outlives any one session.
PYTHON = Path(getattr(sys, "_base_executable", sys.executable))

PACKAGE, RELEASE, MODULE = "local/seam-package", "1.0.0", "seam_package"

#: Every callable reports the Runtime it imported, which is the executor serving it.
SOURCE = """import cozy_runtime
import msgspec
import tensorfs
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

from cozy_runtime.author import (
    App, Context, Model, ModelArtifact, Telemetry, WeightsOutput, invocable,
)

app = App()
PLAIN = dict(tensorfs.seed_digests())["plain/1"]


class Request(msgspec.Struct):
    n: int = 0


class Result(msgspec.Struct):
    value: int
    runtime: str


class Source(Model[object], encoded_leaves="accept"):
    def load(self, loader):
        del loader


@invocable
async def double(ctx: Context, tel: Telemetry, *, n: int) -> Result:
    # Older executors forward these fields raw; the worker's lossy lane must not fail on them.
    tel.log("noisy", loss=float("nan"), seed=2**60)
    return Result(n * 2, cozy_runtime.__version__)


app.entrypoint(double)


@invocable(memoize=True)
async def triple(ctx: Context, *, n: int) -> Result:
    return Result(n * 3, cozy_runtime.__version__)


app.job(triple)


@invocable(memoize=True)
async def derive(ctx: Context, *, n: int) -> ModelArtifact:
    definition = Derivation(
        sources={},
        targets={"body": Target(add={
            "layer.weight": Tensor("f16", (4, n), PLAIN, {"value": Part("f16", (4, n))}),
        })},
        configs={"pipeline": Config("add")},
        order=(("body", "layer.weight"),),
    )
    with tensorfs.derive(ctx.output("model"), definition) as output:
        if output.receipt is None:
            output.add_part("body", "layer.weight", "value", bytes(8 * n))
            output.add_config("pipeline", b"{}")
        receipt = output.receipt or output.commit()
    return ctx.adopt_model(receipt)


app.job(derive, weights=(WeightsOutput("model", max_new_bytes=65536),))


@invocable
async def inspect(ctx: Context, *, model: Source) -> Result:
    view = ctx.tensorfs_source(model).inspect()
    return Result(len(view.components["body"]), cozy_runtime.__version__)


app.job(inspect)


@app.job
async def served(ctx: Context, payload: Request) -> Result:
    return await double(n=payload.n)


@app.job
async def called(ctx: Context, payload: Request) -> Result:
    first = await triple(n=payload.n)
    assert await triple(n=payload.n) == first
    return first


@app.job
async def derived(ctx: Context, payload: Request) -> Result:
    first = await derive(n=payload.n)
    assert (await derive(n=payload.n)).manifest == first.manifest
    return await inspect(model=first)
"""

#: What a root job may call, as a captured Python composition binds it.
CALLABLES = ("derive", "double", "inspect", "triple")

pytestmark = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")


class _File(msgspec.Struct):
    upload_time_iso_8601: datetime.datetime


class _Release(msgspec.Struct):
    urls: list[_File]


def _uv(*argv: str, cwd: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    done = subprocess.run(
        [uv, *argv],
        cwd=cwd,
        env=package_install_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, f"uv {' '.join(argv)}\n{done.stderr[-4000:]}"


def _publish(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}")
    staged.write_bytes(data)
    os.replace(staged, path)


def _lock(version: str) -> bytes:
    """`cozy-runtime==version` and its closure, as a package locked them on release day."""
    path = CACHE / "locks" / f"cozy-runtime-{version}.txt"
    if path.is_file():
        return path.read_bytes()
    url = f"https://pypi.org/pypi/cozy-runtime/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            release = msgspec.json.decode(response.read(), type=_Release)
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"PyPI is unreachable, so cozy-runtime {version} cannot be locked: {exc}")
    uploaded = max(row.upload_time_iso_8601 for row in release.urls)
    with tempfile.TemporaryDirectory(prefix="cz-seam-lock.") as raw:
        work = Path(raw)
        (work / "requirements.in").write_text(f"cozy-runtime=={version}\n")
        _uv(
            "pip",
            "compile",
            "--no-config",
            "--python",
            str(PYTHON),
            "--exclude-newer",
            (uploaded + datetime.timedelta(minutes=1)).isoformat(),
            "--generate-hashes",
            "--no-header",
            "--no-annotate",
            "--output-file",
            str(work / "requirements.txt"),
            str(work / "requirements.in"),
            cwd=work,
        )
        _publish(path, (work / "requirements.txt").read_bytes())
    return path.read_bytes()


def _wheel(requirement: str) -> Path:
    """The seam package's wheel requiring `cozy-runtime<requirement>`, built once per SOURCE."""
    wheels = CACHE / "wheels" / hashlib.sha256((SOURCE + requirement).encode()).hexdigest()[:16]
    for built in wheels.glob("*.whl"):
        return built
    with tempfile.TemporaryDirectory(prefix="cz-seam-wheel.") as raw:
        project = Path(raw) / "project"
        project.mkdir()
        (project / f"{MODULE}.py").write_text(SOURCE)
        (project / "pyproject.toml").write_text(
            f'[project]\nname="{PACKAGE.split("/")[1]}"\nversion="{RELEASE}"\n'
            f'requires-python=">=3.12,<3.13"\ndependencies=["cozy-runtime{requirement}"]\n'
            f'[project.entry-points."cozy.application"]\ndefault="{MODULE}:app"\n'
            '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
            f'[tool.hatch.build.targets.wheel]\nonly-include=["{MODULE}.py"]\n'
        )
        _uv("build", "--wheel", "--out-dir", str(Path(raw) / "out"), str(project), cwd=project)
        (built,) = (Path(raw) / "out").glob("*.whl")
        _publish(wheels / built.name, built.read_bytes())
    return wheels / built.name


def _installation(version: str, requirement: str) -> tuple[str, Path, bytes, str]:
    """The cached package environment locked to cozy-runtime `version`, its wheel and lock,
    and the Runtime the installer chose for it."""
    lock, wheel = _lock(version), _wheel(requirement)
    identity = b"\0".join((lock, wheel.read_bytes(), str(PYTHON).encode()))
    installation = f"seam-{version}-{hashlib.sha256(identity).hexdigest()[:16]}"
    installed = package_installation.install(
        CACHE / "installs",
        package=PACKAGE,
        release=RELEASE,
        python=PYTHON,
        installation_id=installation,
        requirements=lock,
        wheels=(wheel,),
    )
    record = json.loads((installed.generation.parent / "installation.json").read_text())
    return installation, wheel, lock, record["sdk"]["cozy-runtime"]


@dataclass
class Harness:
    """One HEAD worker serving one installed package whose executors run `version`."""

    version: str
    requirement: str
    root: Path
    worker: Worker
    thread: threading.Thread
    client: rpc.WorkerControlStub
    claim: pb.Claim
    prepared: pb.PreparePackageSetResult
    capture: bytes
    capture_digest: bytes
    #: what the Host sends with every submission of this package
    preparation: pb.PrepareLocalPackageRequest

    def settle(self, request_id: str) -> dict[str, object]:
        """The succeeded result. A hang is stillness, never elapsed time."""
        still = ("", time.monotonic())
        while self.worker.execution_status(self.claim, request_id).state not in (
            "succeeded",
            "failed",
        ):
            reading = str(activity.meter(self.worker))
            if reading != still[0]:
                still = (reading, time.monotonic())
            assert time.monotonic() - still[1] < 180, "the worker stopped moving"
            assert self.thread.is_alive()
            time.sleep(0.02)
        outcome = self.worker.collect_execution(self.claim, request_id)
        body = documents.parse(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        assert body.status == pb.OUTCOME_STATUS_SUCCEEDED, (self.version, body)
        result = json.loads(body.result.inline_result)
        assert isinstance(result, dict)
        return result

    def submit(self, offer: pb.AttemptOffer, state: pb.DesiredWorkerState, raw: bytes) -> None:
        self.client.SubmitMachineExecution(
            pb.MachineExecutionSubmit(
                claim=self.claim,
                submission_id=offer.request_id,
                capture_digest=self.capture_digest,
                capture_canonical_bytes=self.capture,
                offer=offer,
                payload_canonical_bytes=raw,
                prepared_state=state,
                expected_execution_workspace_id=self.client.GetMachineExecutionWorkspace(
                    pb.MachineExecutionWorkspaceQuery(claim=self.claim)
                ).execution_workspace_id,
            ),
            timeout=30,
        )

    def serve(self, request_id: str, entrypoint: str, payload: dict[str, int]) -> dict[str, object]:
        """One inference on the prepared placement: `prepare_request`, then `invoke`."""
        placements = self.prepared.placement_set
        placement = documents.parse(
            placements.placement_set_canonical_bytes, pb.PlacementSet
        ).placements[0]
        binding = documents.spell(
            next(
                row for row in placement.entrypoints if row.name == entrypoint
            ).entrypoint_binding_digest
        )
        raw = canonical_json.encode(payload)
        digest = documents.spell(documents.digest_of(raw))
        spec, spec_digest = documents.identity(
            pb.InvocationSpec(
                installation_id=placement.installation_id,
                payload_digest=digest,
                inputs=[
                    pb.InputBinding(
                        input_id="payload",
                        digest=digest,
                        length=len(raw),
                        kind_mime="application/json",
                    )
                ],
                serving=pb.ServingInvocationSpec(
                    entrypoint_binding_digest=binding,
                    attempt_binding_id=binding,
                    bindings_digest=documents.spell(placement.bindings_digest),
                ),
            )
        )
        self.submit(
            pb.AttemptOffer(
                request_id=request_id,
                attempt_ordinal=1,
                placement_id=placement.placement_id,
                invocation_spec_canonical_bytes=spec,
                invocation_spec_digest=spec_digest,
                grant=pb.DeliveryGrant(
                    invocation_spec_digest=spec_digest,
                    inputs=[
                        pb.InputAccess(
                            input_id="payload",
                            url="data:application/json;base64," + base64.b64encode(raw).decode(),
                        )
                    ],
                ),
            ),
            pb.DesiredWorkerState(
                revision=1,
                wire_minor=WIRE_MINOR,
                posture=pb.POSTURE_ACCEPTING,
                placement_set=placements,
            ),
            raw,
        )
        return self.settle(request_id)

    def run(self, request_id: str, job: str, payload: dict[str, int]) -> dict[str, object]:
        """One root job, orchestrated on the worker: `run_job` and its durable requests."""
        plans = (self.root / "home" / "job-plans").glob("*/*.json")
        binding = JobBinding.read(
            next(row for row in map(json.loads, map(Path.read_bytes, plans)) if row["job"] == job)
        )
        owner = LocalRecordOwner(
            LocalRequest(
                entrypoint=job,
                payload=payload,
                outputs=(),
                kind="job",
                request_id=request_id,
                job_descriptor_id=binding.job_descriptor_id,
            ),
            {},
            self.root / ("grants-" + request_id),
            package_installation_id=binding.installation_id,
        )
        offered = owner.offer()
        state = owner.desired_state()
        state.job.orchestration = True
        self.submit(offered, state, canonical_json.encode(payload))
        return self.settle(request_id)


@pytest.fixture(scope="module", params=CASES, ids=lambda case: case[1])
def harness(request: pytest.FixtureRequest) -> Iterator[Harness]:
    locked, requirement = request.param
    installation, wheel, lock, version = _installation(locked, requirement)

    def link(install_root: Path) -> None:
        (install_root / "installations" / installation).symlink_to(
            CACHE / "installs" / "installations" / installation
        )

    with _session(installation, wheel, lock, version, requirement, link) as serving:
        yield serving


@contextlib.contextmanager
def _session(
    installation: str,
    wheel: Path,
    lock: bytes,
    version: str,
    requirement: str,
    place: Callable[[Path], None],
) -> Iterator[Harness]:
    """One worker serving `installation`, which `place` puts into its install root."""
    # Short: the executors' control sockets live under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-seam.") as raw:
        root = Path(raw)
        install_root = root / "installs"
        (install_root / "installations").mkdir(parents=True)
        place(install_root)
        staged = install_root / ".stage" / installation / "wheels"
        staged.mkdir(parents=True)
        shutil.copyfile(wheel, staged / wheel.name)
        (root / "artifacts").mkdir()
        worker = Worker(
            RuntimeConfig(
                cozy_home=root / "home",
                credentials=Credentials(),
                record_owner_public_key=signed_claims.PUBLIC_KEY,
                child_base_env=tuple(
                    sorted(
                        (key, value)
                        for key, value in os.environ.items()
                        if not child_env.erased(key) and key != "PYTHONPATH"
                    )
                ),
            ),
            WorkerOptions(
                **signed_claims.IDENTITY,
                root=root / "worker",
                devices="",
                accelerator_backend="none",
                python=str(PYTHON),
                install_root=install_root,
                artifact_cache=root / "artifacts",
                tensorfs_root=root / "store",
                grant_roots=(str(root),),
            ),
            GrpcControlHost("127.0.0.1:0", root / "address"),
        )
        thread = threading.Thread(target=worker.run, daemon=True)
        channel: grpc.Channel | None = None
        try:
            preparation = pb.PrepareLocalPackageRequest(
                operation_id=installation,
                package=_development(PACKAGE, RELEASE, installation),
                files=_wheel_rows([staged / wheel.name]),
                dependency_requirements=lock,
                install_root=str(install_root),
            )
            prepared = worker.prepare_local_package(preparation)
            capture, capture_digest = documents.identity(
                pb.MachineExecutionCapture(
                    root_installation_id=installation,
                    installed_packages=[prepared.installed_package],
                    bindings=[
                        pb.MachineCallableBinding(
                            caller_installation_id=installation,
                            callee_installation_id=installation,
                            module=MODULE,
                            export=export,
                            entrypoint=export,
                        )
                        for export in CALLABLES
                    ],
                )
            )
            if worker.supervision.current is not None:
                worker.supervision.retire_current(worker.supervision.current, "begin lifecycle")
            thread.start()
            while not (root / "address").exists():
                assert thread.is_alive()
                time.sleep(0.02)
            claim = signed_claims.claim()
            worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda _: None)
            channel = grpc.insecure_channel((root / "address").read_text().strip())
            yield Harness(
                version,
                requirement,
                root,
                worker,
                thread,
                rpc.WorkerControlStub(channel),
                claim,
                prepared,
                capture,
                capture_digest,
                preparation,
            )
        finally:
            if channel is not None:
                channel.close()
            worker.request_stop()
            worker.host.stop()
            if thread.ident is not None:
                thread.join(30)
            assert not thread.is_alive()


def test_serving_invoke(harness: Harness) -> None:
    """A root inference: `prepare_request`, then `invoke` and its reply. Its NaN and 2^60 log
    fields, which no JSON journal holds exactly, cost only their live frame."""
    assert harness.serve("inference", "double", {"n": 21}) == {
        "value": 42,
        "runtime": harness.version,
    }


def test_job_calls_a_serving_child(harness: Harness) -> None:
    """A root job's durable `child_call`/`child_poll` reach a served child's `invoke`."""
    assert harness.run("served", "served", {"n": 5}) == {"value": 10, "runtime": harness.version}


def test_job_polls_a_finished_job_child(harness: Harness) -> None:
    """A job child runs, settles, and its parent polls the result; the repeat is a memo hit."""
    assert harness.run("called", "called", {"n": 5}) == {"value": 15, "runtime": harness.version}


def test_job_derives_adopts_and_reads_weights(harness: Harness) -> None:
    """ExecutionStorage's `weights_writer`: output and adopt in one child, source in another."""
    assert harness.run("derived", "derived", {"n": 4}) == {"value": 1, "runtime": harness.version}


def test_prepared_adapters_refuse_only_the_unsupported_installed_sdk(harness: Harness) -> None:
    """The same real old SDK still answers base-only derivation; adapter work refuses alone."""
    installation = package_installation.open_installation(
        harness.root / "installs", harness.prepared.installed_package.installation_id
    )
    bare = derive_child.SlotRequest("inspect.models.model", b"{}")
    request = derive_child.DeriveRequest((bare,), application=MODULE + ":app")
    assert derive_child.derive_in(installation.python, request) == (
        derive_child.SourceSlot(bare.slot),
    )
    if not harness.requirement.startswith("=="):
        return
    graph = canonical_json.encode({"format": "cozy.model.lora/1", "layers": [], "adapters": []})
    adapted = derive_child.DeriveRequest(
        (derive_child.SlotRequest(bare.slot, b"{}", adapters=graph),),
        application=MODULE + ":app",
    )
    with pytest.raises(derive_child.DeriveRefusal) as refused:
        derive_child.derive_in(installation.python, adapted)
    assert refused.value.code == "adapter_composition_unsupported"
    assert "update its Runtime dependency" in refused.value.detail


def test_an_exact_pin_keeps_its_runtime_and_a_bound_runs_the_machines(harness: Harness) -> None:
    """Execute the SDK selected when this environment was materialized. A later PyPI
    publication cannot change that captured choice or an already running executor."""
    assert harness.run("selected-sdk", "called", {"n": 2}) == {
        "value": 6,
        "runtime": harness.version,
    }
    if harness.requirement.startswith("=="):
        assert harness.version == harness.requirement.removeprefix("==")
        return
    assert Version(harness.version) > Version("0.18.67")


def test_a_runtime_update_retires_the_warm_executor_and_the_next_run_reports_the_new_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs 2575 and 2710: after `cozy rental update --runtime-wheel`, a request was served by
    an executor of the Runtime the machine ran before. Here the machine ran 0.18.67 when the
    package was installed and its executor is warm. The machine is updated; the next
    submission's preparation rebuilds the installation's SDK generation, and that run is
    served by an executor of the new generation, never by the warm one. Nothing is refused."""
    old = "0.18.67"
    lock, wheel = _lock(old), _wheel(f">={old}")

    def install(install_root: Path) -> None:
        monkeypatch.setattr(package_installation, "machine_sdk", lambda: {"cozy-runtime": old})
        package_installation.install(
            install_root,
            package=PACKAGE,
            release=RELEASE,
            python=PYTHON,
            installation_id="updated",
            requirements=lock,
            wheels=(wheel,),
        )

    with _session("updated", wheel, lock, old, f">={old}", install) as serving:
        assert serving.serve("before", "double", {"n": 2}) == {"value": 4, "runtime": old}
        record = serving.root / "installs" / "installations" / "updated" / "installation.json"
        before = json.loads(record.read_text())["python"]
        monkeypatch.undo()  # the Runtime update: this machine now runs another SDK
        serving.prepared = serving.worker.prepare_local_package(serving.preparation)
        rebuilt = json.loads(record.read_text())
        assert rebuilt["python"] != before and rebuilt["sdk"]["cozy-runtime"] != old
        after = serving.serve("after", "double", {"n": 3})
        assert after == {"value": 6, "runtime": rebuilt["sdk"]["cozy-runtime"]}
