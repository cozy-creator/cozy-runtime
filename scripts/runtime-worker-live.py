#!/usr/bin/env python3
"""Narrow live contract proof for the fixed environment-neutral Runtime worker."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict, cast
from unittest.mock import patch

from cozy_runtime.cli.runtime_worker import _worker_options
from cozy_runtime.internal import (
    canonical,
    contract,
    fill,
    host_paths,
    hostfacts,
    local,
    provisioned,
    pull,
    readiness,
)
from cozy_runtime.internal.config import (
    ConfigError,
    Credentials,
    RuntimeConfig,
    read_worker_host_config,
)
from cozy_runtime.internal.placement_materialization import (
    EMPTY_ARTIFACT_OBJECT_SET,
    PreparedBoot,
    _publish_exact,
    materialize_objects,
)
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.grants import Authorizer, GrantRefusal
from cozy_runtime.internal.worker.plan import DeclaredBinding, PlanRefusal
from cozy_runtime.internal.worker.session import Placement, Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR
from cozy_runtime.protocol import worker_pb2 as pb

ROOT = Path(__file__).resolve().parent.parent


class Ref(TypedDict):
    digest: str
    length: int


class ObjectSetVector(TypedDict):
    name: str
    document: dict[str, object]
    canonical: str
    digest: str


def ref(raw: bytes) -> Ref:
    return {"digest": "sha256:" + hashlib.sha256(raw).hexdigest(), "length": len(raw)}


class PolicyExecutor:
    """A live worker-session seam double: every executor command is counted, never hidden."""

    def __init__(self) -> None:
        self.ready_plans = {"plan-live"}
        self.calls: list[str] = []
        self.prepare_reply: dict[str, Any] | None = None
        self.incarnation = 1
        self.pid = 999_999
        self.poisoned = ""
        self.construction_key = ""
        self.prepared_binding_set: frozenset[str] = frozenset()
        self.prepared_model: object | None = None
        self.reserved_for_job = False
        self.spawned_at = 1.0
        self.ready_at = 1.001
        self.hello = {"no_new_privs": "1", "oom_score_adj": "0", "torch_loaded": True}

    def alive(self) -> bool:
        return True

    def call(self, command: dict[str, Any], **_options: Any) -> dict[str, Any]:
        name = str(command["cmd"])
        self.calls.append(name)
        if name == "prepare":
            assert self.prepare_reply is not None
            return self.prepare_reply
        if name == "warm":
            return {
                "ok": True,
                "outcome": "skipped",
                "reason": "policy proof fixture",
                "boot_warmup_observed": False,
            }
        assert name == "probe"
        return {}


class PolicySupervision:
    def __init__(self, executor: PolicyExecutor) -> None:
        self.current = executor

    def owns(self, executor: object) -> bool:
        return executor is self.current


def policy_worker(
    root: Path, *, environment_root: Path | None, no_warm: bool, accelerator_backend: str = ""
) -> tuple[Worker, PolicyExecutor]:
    """A real Worker carrying controlled executor ownership and one ready binding."""

    options = WorkerOptions(
        root=root,
        environment_root=environment_root,
        artifact_store=root / "store",
        artifact_object_set=root / "object-set.json",
        accelerator_backend=accelerator_backend,
        no_warm=no_warm,
    )
    worker = Worker(
        RuntimeConfig(cozy_home=root.parent / "home", credentials=Credentials()),
        options,
        InMemoryControlHost(),
    )
    executor = PolicyExecutor()
    worker.supervision = cast(Any, PolicySupervision(executor))
    worker.bindings = {"plan-live": cast(Any, SimpleNamespace(entrypoint="probe"))}
    worker.placement = Placement(
        placement_id="placement-live",
        spec_digest=b"s" * 32,
        package_release_id="cozy/probe@live",
        environment_spec_digest=b"e" * 32,
        installed_environment_receipt_digest=b"i" * 32,
        package_descriptor_digest=b"d" * 32,
        model_object_set=("sha256:" + "0" * 64, 53),
        plan_ids=("plan-live",),
        plan_refs={},
        materialization=pb.MaterializationState.MATERIALIZATION_STATE_STAGED,
        serving=pb.ServingState.SERVING_STATE_ACTIVATING,
    )
    worker.accepted.mode = "serving"
    worker.accepted.accepted_desired_state_revision = 7
    return worker, executor


def prove_worker_warm_policy(root: Path) -> None:
    """The worker-owned hardcut, plus both unchanged local policy arms."""

    dynamic, dynamic_executor = policy_worker(
        root / "dynamic", environment_root=root / "environments", no_warm=False
    )
    dynamic._finish_activation()
    assert dynamic_executor.calls == []
    assert dynamic.options.no_warm is False  # the worker hardcut is independent of argv
    assert dynamic.placement is not None
    assert dynamic.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
    assert dynamic.accepted.converged_revision == 7 and dynamic.settled.is_set()
    assert any(
        row.kind == "warm" and "dynamic placement worker" in row.step for row in dynamic.activity
    )

    local, local_executor = policy_worker(root / "local", environment_root=None, no_warm=False)
    assert local.run_warm_pass() is True
    assert local_executor.calls == ["warm", "probe"]

    explicit_skip, explicit_executor = policy_worker(
        root / "explicit-skip", environment_root=None, no_warm=True
    )
    assert explicit_skip.run_warm_pass() is True
    assert explicit_executor.calls == []


def prove_pre_ready_claim(root: Path) -> None:
    """Authentication precedes the state-inert UNDURABLE readiness gate."""

    worker = Worker(
        RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            bootstrap_credential="owner-proof",
        ),
        WorkerOptions(root=root / "worker", worker_id="worker-live", claim_ready=False),
        InMemoryControlHost(),
    )
    frames: list[pb.WorkerFrame] = []
    bad = pb.Claim(
        record_owner_epoch=1,
        record_owner_id="owner-live",
        worker_id="worker-live",
        wire_minor=WIRE_MINOR,
        proof=b"wrong",
    )
    generation, _ = worker.accept_claim(bad, frames.append)
    assert generation == 0
    assert frames[-1].claim_ack.rejection == pb.ClaimRejection.CLAIM_REJECTION_UNAUTHENTICATED

    good = pb.Claim()
    good.CopyFrom(bad)
    good.proof = b"owner-proof"
    generation, _ = worker.accept_claim(good, frames.append)
    assert generation == 0
    assert frames[-1].claim_ack.rejection == pb.ClaimRejection.CLAIM_REJECTION_UNDURABLE
    assert worker.fence.control_stream_generation == 0

    worker.mark_claim_ready()
    generation, _ = worker.accept_claim(good, frames.append)
    assert generation == 1 and worker.fence.control_stream_generation == 1
    worker.shutdown()


def prepared_binding() -> DeclaredBinding:
    return DeclaredBinding(
        entrypoint_binding_plan_id="plan-live",
        entrypoint="probe",
        model_class="ProbeModel",
        model_binding_path="probe.models.model",
        model_parameter_name="model",
        release="cozy/probe-model@live",
        application="probe_package:app",
        descriptor_path="/verified/package.descriptor.json",
        store="/verified/store",
        config="/verified/model.json",
        variant="plain",
        reference_snapshot="sha256:" + "a" * 64,
        components=("model",),
        snapshots={"model": "sha256:" + "a" * 64},
    )


def prepare_reply() -> dict[str, Any]:
    """A nonempty current Plan.wire_document projection, including the plural censuses."""

    return {
        "ok": True,
        "model_construction_digest": "sha256:" + "b" * 64,
        "constructed_model_digest": "sha256:" + "c" * 64,
        "facts": {
            "filled": 1,
            "filled_bytes": 4096,
            "fill_ms": 1.0,
            "allocator_bytes": 4096,
            "reserved_bytes": 4096,
            "resident": {"model": 4096},
            "parked": [],
            "gpu_name": "proof-device",
        },
        "delivery": {
            "rung": "verbatim",
            "variant": "plain",
            "delivery": "verbatim",
            "materialization": "",
            "label": "verbatim",
            "plan_digest": "sha256:" + "d" * 64,
            "routes": {"verbatim": 1},
            "kernels": {"library_float": 1},
            "objective": "latency",
            "calibrated": False,
            "confession": "",
        },
    }


def prove_delivery_activation(root: Path) -> None:
    """A real prepare reply crosses delivery narration, dynamic skip, and readiness."""

    worker, executor = policy_worker(
        root / "plain-delivery", environment_root=root / "environments", no_warm=False
    )
    binding = prepared_binding()
    worker.bindings = {binding.entrypoint_binding_plan_id: binding}
    worker.engine.bindings = worker.bindings
    executor.ready_plans.clear()
    executor.prepare_reply = prepare_reply()
    worker.activate_executor()
    assert executor.calls == ["prepare"]
    assert worker.placement is not None
    assert worker.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
    assert worker.accepted.converged_revision == 7
    assert any(
        row.kind == "delivery"
        and "kernels library_float x1" in row.step
        and "uncalibrated" in row.step
        for row in worker.activity
    )
    assert any(
        row.kind == "warm" and "dynamic placement worker" in row.step for row in worker.activity
    )

    # Any later schema/observer defect is a typed OFFLINE state, not a dead convergence
    # thread silently marooning the placement in ACTIVATING.
    broken, broken_executor = policy_worker(
        root / "broken-observation", environment_root=root / "environments", no_warm=False
    )
    broken_binding = prepared_binding()
    broken.bindings = {broken_binding.entrypoint_binding_plan_id: broken_binding}
    broken.engine.bindings = broken.bindings
    broken_executor.ready_plans.clear()
    broken_executor.prepare_reply = {"ok": True}
    broken.activate_executor()
    assert broken.placement is not None
    assert broken.placement.serving == pb.ServingState.SERVING_STATE_OFFLINE
    assert broken.latched == "activation_exception"
    assert broken.engine.faults[-1].reason == "activation_exception"


def prove_wire_minor_compatibility(root: Path) -> None:
    """Both additive-skew directions converge on the lower implemented minor."""

    for name, advertised, effective in (
        ("older-owner", WIRE_MINOR - 1, WIRE_MINOR - 1),
        ("newer-owner", WIRE_MINOR + 1, WIRE_MINOR),
    ):
        worker, _ = policy_worker(root / name, environment_root=None, no_warm=True)
        worker.supervision.current = None
        generation, _ = worker.accept_claim(
            pb.Claim(
                record_owner_epoch=1,
                record_owner_id=name,
                wire_minor=advertised,
            ),
            lambda _frame: None,
        )
        assert generation != 0
        assert worker.fence.wire_minor == effective
        worker._apply_desired_state(
            pb.DesiredWorkerState(
                revision=8,
                posture=pb.Posture.POSTURE_ACCEPTING,
                wire_minor=advertised,
            )
        )
        assert worker.fence.wire_minor == effective
        assert worker.observed_state().applied_wire_minor == effective


def prove_backend_version(root: Path) -> None:
    """The ClaimAck resource source carries measured CUDA version, and invents none."""

    worker, _ = policy_worker(root / "worker", environment_root=root / "env", no_warm=False)
    measured = hostfacts.HostFacts(
        gpu_name="NVIDIA proof GPU",
        vram_total_bytes=24 << 30,
        driver_version="999.1",
        cuda_version="13.2",
        platform="proof-linux",
        host_ram_total_bytes=64 << 30,
        vcpu_count=16,
        backend="cuda",
    )
    with patch.object(hostfacts, "measure", return_value=measured):
        resources = worker.resources()
    assert resources.backend == "cuda"
    assert resources.backend_version == "13.2"
    assert resources.driver_version == "999.1"  # distinct measured fields stay distinct

    unmeasured = hostfacts.HostFacts(
        gpu_name="NVIDIA proof GPU",
        driver_version="999.1",
        cuda_version="",
        backend="cuda",
        unreadable=("cuda_version",),
    )
    with patch.object(hostfacts, "measure", return_value=unmeasured):
        absent = worker.resources()
    assert absent.backend_version == ""
    assert "cuda_version" in absent.unreadable

    cpu_worker, _ = policy_worker(
        root / "cpu-worker",
        environment_root=root / "cpu-env",
        no_warm=False,
        accelerator_backend="none",
    )
    cpu = hostfacts.HostFacts(
        backend="none",
        platform="proof-linux",
        host_ram_total_bytes=32 << 30,
        vcpu_count=8,
    )
    with patch.object(hostfacts, "measure", return_value=cpu) as measured_cpu:
        resources = cpu_worker.resources()
    measured_cpu.assert_called_once_with("none")
    assert resources.backend == "none"
    assert resources.memory_model == "host"
    assert resources.device_count == 0
    assert resources.host_ram_total_bytes > 0 and resources.vcpu_count > 0

    with (
        patch.object(hostfacts, "_installed_cuda_version", return_value=("", False)),
        patch("cozy_runtime.internal.hostfacts.platform.platform", return_value="proof-linux"),
        patch("subprocess.run", side_effect=AssertionError("CPU profile probed nvidia-smi")),
    ):
        measured_host = hostfacts.measure("none")
    assert measured_host.backend == "none" and measured_host.gpu_count == 0
    assert not ({"gpu_name", "driver_version"} & set(measured_host.unreadable))


def prove_installed_backend_authority(root: Path) -> None:
    """Image metadata beats the driver's maximum; bad metadata invents no version."""

    purelib = root / "purelib"
    version_file = purelib / "torch" / "version.py"
    version_file.parent.mkdir(parents=True)
    version_file.write_text(
        "from typing import Optional\n"
        "__version__ = '2.13.0+cu126'\n"
        "debug = False\n"
        "cuda: Optional[str] = '12.6'\n"
        "git_version = 'final-image-proof'\n"
        "hip: Optional[str] = None\n",
    )
    # A driver advertising CUDA 13.0 is deliberately present as the tempting wrong fact.
    # The installed-toolkit reader never asks for it and returns torch's exact 12.6 literal.
    driver_max = SimpleNamespace(stdout="CUDA Version : 13.0\n")
    with patch("subprocess.run", return_value=driver_max) as queried:
        version, unreadable = hostfacts._installed_cuda_version(
            {"purelib": str(purelib), "platlib": str(purelib)}
        )
    assert (version, unreadable) == ("12.6", False)
    assert queried.call_count == 0

    missing, missing_unreadable = hostfacts._installed_cuda_version(
        {"purelib": str(root / "missing"), "platlib": str(root / "missing")}
    )
    assert missing == "" and missing_unreadable is True

    for index, malformed in enumerate(
        (
            "cuda = build_cuda_version()\n",
            "cuda = 'driver maximum 13.0'\n",
            "cuda = '12.6'\ncuda = '13.0'\n",
        )
    ):
        bad_root = root / f"malformed-{index}"
        bad = bad_root / "torch" / "version.py"
        bad.parent.mkdir(parents=True)
        bad.write_text(malformed)
        value, bad_unreadable = hostfacts._installed_cuda_version(
            {"purelib": str(bad_root), "platlib": str(bad_root)}
        )
        assert value == "" and bad_unreadable is True

    cpu_root = root / "cpu"
    cpu = cpu_root / "torch" / "version.py"
    cpu.parent.mkdir(parents=True)
    cpu.write_text("cuda = None\n")
    assert hostfacts._installed_cuda_version(
        {"purelib": str(cpu_root), "platlib": str(cpu_root)}
    ) == ("", False)

    # Measurement propagates unreadability under the WIRE field's name. It never fills the
    # empty version from the driver query that still measures device/driver facts.
    with (
        patch.object(hostfacts, "_installed_cuda_version", return_value=("", True)),
        patch("subprocess.run", side_effect=OSError("nvidia-smi absent")),
    ):
        facts = hostfacts.measure()
    assert facts.cuda_version == "" and "backend_version" in facts.unreadable


def mint_certificate(root: Path, name: str, san: str) -> Path:
    certificate, key = root / f"{name}.crt", root / f"{name}.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=cozy-worker",
            "-addext",
            f"subjectAltName={san}",
        ],
        check=True,
        capture_output=True,
    )
    return certificate


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cozy-runtime-worker-") as raw_root:
        root = Path(raw_root)
        vectors: dict[str, tuple[bytes, provisioned.ArtifactObjectSet]] = {}
        for line in (ROOT / "vectors" / "artifact-object-sets.jsonl").read_text().splitlines():
            vector = cast(ObjectSetVector, json.loads(line))
            raw = canonical.write(vector["document"])
            assert raw == vector["canonical"].encode()
            assert "sha256:" + hashlib.sha256(raw).hexdigest() == vector["digest"]
            vectors[vector["name"]] = (raw, provisioned.read_artifact_object_set(raw))
        empty_bytes, empty_object_set = vectors["empty"]

        token = "1" * 64
        control_key = base64.urlsafe_b64encode(b"c" * 32).rstrip(b"=").decode()
        env = {
            "AWS_SECRET_ACCESS_KEY": "must-not-cross",
            "HF_TOKEN": "must-not-cross",
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin",
            "RUNPOD_API_KEY": "must-not-cross",
            "COZY_WORKER_ID": "worker-live",
            "COZY_WORKER_INTERNAL_PORT": "43100",
            "COZY_MEDIA_INTERNAL_PORT": "43101",
            "COZY_WORKER_ARTIFACT_ORIGIN": "https://worker.tensorhub.com",
            "COZY_RECORD_OWNER_AUTH_JSON": json.dumps(
                {
                    "control_public_key_ed25519_b64url": control_key,
                    "media_token_sha256": [token],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "COZY_WORKER_LEASE_EXPIRY_UNIX": str(int(time.time()) + 60),
        }
        try:
            read_worker_host_config({**env, "PYTHONPATH": "/injected/source"})
        except ConfigError as exc:
            assert exc.name == "worker_host_config"
        else:
            raise AssertionError("PYTHONPATH source injection was accepted")
        try:
            read_worker_host_config({**env, "COZY_UNKNOWN": "retired"})
        except ConfigError as exc:
            assert exc.name == "worker_host_config" and "unknown" in exc.message
        else:
            raise AssertionError("an unknown COZY name entered the fixed worker handoff")
        auth = json.loads(env["COZY_RECORD_OWNER_AUTH_JSON"])
        for planted in (
            {**auth, "extra": "ambiguous"},
            {**auth, "control_public_key_ed25519_b64url": control_key + "="},
            {**auth, "media_token_sha256": ["2" * 64, "1" * 64]},
        ):
            try:
                read_worker_host_config(
                    {
                        **env,
                        "COZY_RECORD_OWNER_AUTH_JSON": json.dumps(planted),
                    }
                )
            except ConfigError as exc:
                assert exc.name == "worker_host_config"
            else:
                raise AssertionError("malformed RecordOwner auth was accepted")
        for name in (
            "COZY_WORKER_ID",
            "COZY_WORKER_INTERNAL_PORT",
            "COZY_MEDIA_INTERNAL_PORT",
            "COZY_RECORD_OWNER_AUTH_JSON",
            "COZY_WORKER_ARTIFACT_ORIGIN",
            "COZY_WORKER_LEASE_EXPIRY_UNIX",
        ):
            try:
                read_worker_host_config({k: v for k, v in env.items() if k != name})
            except ConfigError as exc:
                assert exc.name == "worker_host_config" and "absent" in exc.message
            else:
                raise AssertionError(f"{name} is not required by the worker handoff")
        try:
            read_worker_host_config({**env, "COZY_WORKER_ID": " untrimmed"})
        except ConfigError as exc:
            assert exc.name == "worker_host_config"
        else:
            raise AssertionError("an untrimmed worker id was accepted")
        for bad_origin in (
            "http://worker.tensorhub.com",
            "https://worker.tensorhub.com/",
            "https://WORKER.tensorhub.com",
            "https://worker.tensorhub.com/path",
            "https://user@worker.tensorhub.com",
            "https://worker.tensorhub.com?",
            "https://worker.tensorhub.com#",
        ):
            try:
                read_worker_host_config({**env, "COZY_WORKER_ARTIFACT_ORIGIN": bad_origin})
            except ConfigError as exc:
                assert exc.name == "worker_host_config"
            else:
                raise AssertionError(f"invalid worker artifact origin was accepted: {bad_origin}")
        config = read_worker_host_config(env)
        assert dict(config.child_base_env) == {"LANG": "C.UTF-8"}
        assert config.worker_id == "worker-live"
        assert config.artifact_origin.value == "https://worker.tensorhub.com"
        assert config.artifact_origin.host == "worker.tensorhub.com"
        assert config.artifact_origin.port == 443
        worker_options = _worker_options(
            config,
            PreparedBoot(root / "store", root / "object-set.json", root / "cache"),
            "boot-live",
            [
                {
                    "device_index": 0,
                    "device_name": "NVIDIA RTX 4090",
                    "device_uuid": "GPU-live",
                    "driver_version": "999.0",
                    "memory_bytes": 24 << 30,
                    "pci_bus_id": "00000000:01:00.0",
                }
            ],
            certificate_der=b"certificate-der",
            accelerator_backend="cuda",
            control_runtime_digest_value="sha256:" + "3" * 64,
        )
        assert worker_options.no_warm is True
        assert worker_options.control_runtime_digest == "sha256:" + "3" * 64
        assert worker_options.accelerator_backend == "cuda"
        assert worker_options.executor_uid == 65533
        assert worker_options.claim_ready is False
        grant_root = worker_options.grant_roots[0]
        assert grant_root == str(host_paths.MEDIA_ROOT) == "/var/lib/cozy/media"
        authorizer = Authorizer(roots=(grant_root,))
        # podmedia PutInput returns a payload path below its fixed mediaRoot. Exercise the
        # worker's real closed call table and root selector against that returned shape.
        payload_url = "file:///var/lib/cozy/media/inputs/invocation-live/payload.json"
        authorizer.authorize("grant_input_read", "GET", payload_url, {"input_id": "payload"})
        assert authorizer.allowed == 1 and authorizer.refusals == []
        for outside in (
            "file:///run/cozy/media/inputs/invocation-live/payload.json",
            "file:///var/lib/cozy/media-other/inputs/invocation-live/payload.json",
        ):
            try:
                authorizer.authorize("grant_input_read", "GET", outside, {"input_id": "payload"})
            except GrantRefusal as exc:
                assert exc.code == "grant_path_outside_roots"
            else:
                raise AssertionError(f"outside podmedia path {outside!r} was authorized")

        with patch("subprocess.run", side_effect=AssertionError("CPU boot probed nvidia-smi")):
            cpu_gpus = readiness.runtime_gpus(
                config.child_base_env,
                deadline=config.lease_expiry_unix,
                accelerator_backend="none",
            )
        assert cpu_gpus == []
        cpu_options = _worker_options(
            config,
            PreparedBoot(root / "cpu-store", root / "cpu-object-set.json", root / "cpu-cache"),
            "cpu-boot-live",
            cpu_gpus,
            certificate_der=b"certificate-der",
            accelerator_backend="none",
            control_runtime_digest_value="sha256:" + "3" * 64,
        )
        assert cpu_options.accelerator_backend == "none"
        assert cpu_options.devices == ""
        try:
            _worker_options(
                config,
                PreparedBoot(root / "bad-store", root / "bad-set.json", root / "bad-cache"),
                "bad-cuda-boot",
                [],
                certificate_der=b"certificate-der",
                accelerator_backend="cuda",
                control_runtime_digest_value="sha256:" + "3" * 64,
            )
        except LaunchRefusal as exc:
            assert exc.code == "runtime_gpus_invalid"
        else:
            raise AssertionError("CUDA base with no GPU was admitted")
        # The hardcut is specific to fixed base worker images. A hand-started/local worker
        # keeps the existing opt-in behavior and may still choose to run its warm pass.
        assert WorkerOptions(root / "local-worker").no_warm is False
        prove_worker_warm_policy(root / "worker-policy")
        prove_pre_ready_claim(root / "pre-ready-claim")
        prove_wire_minor_compatibility(root / "wire-minor")
        prove_delivery_activation(root / "delivery-activation")
        prove_backend_version(root / "backend-version")
        prove_installed_backend_authority(root / "installed-backend")

        assert empty_bytes == b'{"kind":"tensorhub.resolved_object_set/1","roots":[]}'
        assert empty_object_set.roots == () and empty_object_set.refs == {}
        local_weightless = local.weightless_release_documents(
            {
                "application": "weightless:app",
                "entrypoints": [{"models": [], "name": "health"}],
            }
        )
        assert local_weightless.artifact_object_set == empty_bytes
        assert ref(local_weightless.artifact_object_set) == {
            "digest": "sha256:d60755ff474871fce1e9705850245780895c1d2b798750f7692962c2d64d8856",
            "length": 53,
        }
        assert local_weightless.artifact_object_set == EMPTY_ARTIFACT_OBJECT_SET
        boot_root = root / "boot"
        boot_root.mkdir()
        published = boot_root / "artifact-object-set.json"
        _publish_exact(published, EMPTY_ARTIFACT_OBJECT_SET, boot_root)
        _publish_exact(published, EMPTY_ARTIFACT_OBJECT_SET, boot_root)
        assert published.read_bytes() == EMPTY_ARTIFACT_OBJECT_SET
        try:
            _publish_exact(published, EMPTY_ARTIFACT_OBJECT_SET + b"\n", boot_root)
        except LaunchRefusal as exc:
            assert exc.code == "boot_document_conflict"
        else:
            raise AssertionError("a different boot document overwrote the published bytes")

        empty_store = materialize_objects(
            {},
            empty_object_set,
            cache=None,
            deadline=None,
            store_root=root / "empty-local-store",
        )
        assert empty_store.store == root / "empty-local-store" and empty_store.store.is_dir()
        assert not (root / "empty-cache" / "tensorfs").exists()
        assert empty_store.sources == {
            "cloud": {"bytes": 0, "objects": 0},
            "local": {"bytes": 0, "objects": 0},
            "network_cache": {"bytes": 0, "objects": 0},
        }

        snapshot = "a" * 64

        class FakeStore:
            def __init__(self, path: Path) -> None:
                self.root = path

        class FakeTensorFS:
            def __init__(self) -> None:
                self.completed = False

            def fetch_plan(
                self,
                _store: FakeStore,
                _session: str,
                root_digest: str,
                root_length: int,
                declared: list[tuple[str, int]],
            ) -> dict[str, object]:
                assert root_digest == snapshot and root_length == 17
                assert declared == [("sha256:" + "b" * 64, 19)]
                return {
                    "document": b"tensorfs-plan",
                    "held": [("sha256:" + snapshot, 17), *declared],
                    "held_bytes": 36,
                    "presence": {},
                    "wanted": [],
                    "wanted_bytes": 0,
                }

            def fetch_complete(self, _store: FakeStore, document: bytes) -> None:
                assert document == b"tensorfs-plan"
                self.completed = True

        tensorfs = FakeTensorFS()
        outcome = pull.pull_granted_snapshot(
            tensorfs,
            FakeStore(root / "store"),
            "sha256:" + snapshot,
            17,
            [
                pull.Read("sha256:" + snapshot, 17, "https://delivery.invalid/root"),
                pull.Read("sha256:" + "b" * 64, 19, "https://delivery.invalid/object"),
            ],
        )
        assert tensorfs.completed and outcome.held == 2 and outcome.fetched == 0
        try:
            pull.pull_granted_snapshot(
                tensorfs,
                FakeStore(root / "store"),
                snapshot,
                17,
                [pull.Read("sha256:" + snapshot, 17, "http://delivery.invalid/root")],
            )
        except pull.PullRefusal as exc:
            assert exc.code == "artifact_incomplete"
        else:
            raise AssertionError("plaintext direct object grant was accepted")
        try:
            pull.open_guarded(
                "https://127.0.0.1/grant",
                pull.Deadline.after(1),
                no_proxy=True,
            )
        except pull.PullRefusal as exc:
            assert exc.code == "source_not_allowed"
        else:
            raise AssertionError("private small-artifact grant bypassed the shared guard")

        construction = contract.ModelConstructionContract(
            candidate=contract.Candidate(
                release="package@release",
                application="package:app",
                model_class="package.models:Model",
                factory="sha256:" + "c" * 64,
                snapshot="sha256:" + snapshot,
                config_digest="sha256:" + "d" * 64,
                hardware_variant="cuda-h200",
            ),
            logical_tensors=(
                contract.LogicalTensor(
                    key="transformer.weight",
                    component="transformer",
                    shape=(16, 16),
                    kind="parameter",
                    module_class="Linear",
                    alias_group=0,
                    dtype=None,
                ),
            ),
            destination_sets=(
                contract.DestinationSet(
                    component="transformer",
                    module_class="Transformer",
                    training=False,
                    compiled=False,
                    keys=("transformer.weight",),
                ),
            ),
            derived=(
                contract.DerivedTensor(
                    key="transformer.rotary",
                    component="transformer",
                    shape=(8,),
                    dtype="float32",
                ),
            ),
            caches=(),
        )
        contract_bytes = canonical.write(construction.render())
        parsed_contract = contract.read_contract(contract_bytes)
        assert parsed_contract == construction
        assert contract.logical_model_digest(parsed_contract).startswith("sha256:")
        malformed_contract = dict(construction.render())
        malformed_contract["asserted_digest"] = "sha256:" + "e" * 64
        try:
            contract.read_contract(canonical.write(malformed_contract))
        except contract.ContractError as exc:
            assert exc.code == "contract_document_invalid"
        else:
            raise AssertionError("open ModelConstructionContract field set was accepted")

        binding_documents = root / "binding-documents"
        binding_documents.mkdir()
        config_bytes = b"{}"
        config_ref = ref(config_bytes)
        config_asset_bytes = b"tokenizer-asset"
        config_asset_ref = ref(config_asset_bytes)
        bound_contract = contract.ModelConstructionContract(
            candidate=contract.Candidate(
                release="package@release",
                application="package:app",
                model_class="package.models:Model",
                factory="factory-identity",
                snapshot="sha256:" + snapshot,
                config_digest=str(config_ref["digest"]),
                hardware_variant="cuda-h200",
            ),
            logical_tensors=construction.logical_tensors,
            destination_sets=construction.destination_sets,
            derived=construction.derived,
            caches=(),
        )
        contract_bytes = canonical.write(bound_contract.render())
        contract_ref = ref(contract_bytes)
        descriptor_bytes = canonical.write(
            {
                "application": "package:app",
                "entrypoints": [
                    {
                        "models": [
                            {
                                "class": "Model",
                                "component_use": {"run": ["transformer"]},
                                "path": "generate.models.model",
                                "stamps": {},
                            }
                        ],
                        "name": "generate",
                        "request": {"fields": []},
                        "result": {"fields": []},
                    },
                    {
                        "models": [],
                        "name": "health",
                        "request": {"fields": []},
                        "result": {"fields": []},
                    },
                ],
                "format": "cozy.package.descriptor/1",
                "jobs": [],
            }
        )
        descriptor_ref = ref(descriptor_bytes)
        layout_ref: Ref = {"digest": "sha256:" + "e" * 64, "length": 23}
        fp8_weights_ref: Ref = {"digest": "sha256:" + "f" * 64, "length": 256}
        model_root_ref: Ref = {"digest": "sha256:" + snapshot, "length": 17}
        binding_bytes = canonical.write(
            {
                "checkpoint": {
                    "snapshot_root": model_root_ref,
                    "tensor_layout": layout_ref,
                },
                "config": {
                    "assets": [{"name": "tokenizer", "ref": config_asset_ref}],
                    "document": config_ref,
                },
                "execution_layout": [
                    {
                        "component": "transformer",
                        "snapshot_root": model_root_ref,
                        "tensor_layout": layout_ref,
                    }
                ],
                "format": "cozy.package.ResolvedModelBinding/1",
                "model_construction_contract": contract_ref,
                "stamps": [],
            }
        )
        binding_ref = ref(binding_bytes)
        plan_bytes = canonical.write(
            {
                "bindings": [{"binding": binding_ref, "slot": "model"}],
                "package_descriptor": descriptor_ref,
                "entrypoint": "generate",
                "format": "cozy.package.EntrypointBindingPlan/1",
            }
        )
        weightless_plan_bytes = canonical.write(
            {
                "bindings": [],
                "package_descriptor": descriptor_ref,
                "entrypoint": "health",
                "format": "cozy.package.EntrypointBindingPlan/1",
            }
        )
        for artifact_ref, raw in (
            (config_ref, config_bytes),
            (contract_ref, contract_bytes),
            (descriptor_ref, descriptor_bytes),
            (binding_ref, binding_bytes),
        ):
            (binding_documents / f"{str(artifact_ref['digest'])[7:]}.json").write_bytes(raw)
        object_set_bytes = canonical.write(
            {
                "kind": "tensorhub.resolved_object_set/1",
                "roots": [
                    {
                        "objects": sorted(
                            [config_asset_ref, fp8_weights_ref, layout_ref],
                            key=lambda row: str(row["digest"]),
                        ),
                        "root": model_root_ref,
                    },
                ],
            }
        )
        object_set = provisioned.read_artifact_object_set(object_set_bytes)
        assert [root.root.digest for root in object_set.roots] == [model_root_ref["digest"]]
        legacy_role_bytes = canonical.write(
            {
                "kind": "tensorhub.resolved_object_set/1",
                "roots": [
                    {
                        "objects": [layout_ref],
                        "role": "package",
                        "root": model_root_ref,
                    }
                ],
            }
        )
        try:
            provisioned.read_artifact_object_set(legacy_role_bytes)
        except PlanRefusal as exc:
            assert exc.code == "artifact_object_set.roots[0]_invalid"
        else:
            raise AssertionError("deleted ArtifactObjectSet role field was accepted")
        plan_ref = ref(plan_bytes)
        weightless_plan_ref = ref(weightless_plan_bytes)
        plan_path = root / "plan.json"
        weightless_plan_path = root / "weightless-plan.json"
        plan_path.write_bytes(plan_bytes)
        weightless_plan_path.write_bytes(weightless_plan_bytes)
        plan_documents = {
            str(plan_ref["digest"]): provisioned.ExactDocument(
                provisioned.ArtifactRef(str(plan_ref["digest"]), int(plan_ref["length"])),
                plan_path,
                plan_bytes,
            ),
            str(weightless_plan_ref["digest"]): provisioned.ExactDocument(
                provisioned.ArtifactRef(
                    str(weightless_plan_ref["digest"]), int(weightless_plan_ref["length"])
                ),
                weightless_plan_path,
                weightless_plan_bytes,
            ),
        }
        selected = provisioned.read_provisioned_release(
            plan_documents,
            read_document=provisioned.staged_document_reader(binding_documents),
            package_release="package@release",
            store_root=root / "store",
            object_set=object_set,
            dtype_widths={"bfloat16": 2, "float32": 4},
        )
        resolved_binding = selected.bindings[str(plan_ref["digest"])]
        assert resolved_binding.application == "package:app"
        assert resolved_binding.components == ("transformer",)
        # The TensorFS payload is FP8-sized, while the allocator destinations are BF16
        # plus one derived f32 buffer. Stored/custody bytes never become the fill budget.
        assert int(fp8_weights_ref["length"]) < 16 * 16 * 2
        assert resolved_binding.model_weight_bytes == 16 * 16 * 2 + 8 * 4
        header_envelope = fill.price_envelope(
            [("transformer", 16 * 16 * 2, int(fp8_weights_ref["length"]), "decoded_float")],
            decode_scratch_bytes=64,
        )
        complete_envelope = fill.complete_device_envelope(
            header_envelope, resolved_binding.model_weight_bytes
        )
        assert complete_envelope["derived_destination_bytes"] == 8 * 4
        assert complete_envelope["required_bytes"] == 16 * 16 * 2 + 8 * 4 + 256
        assert resolved_binding.expected_logical_model_digest == contract.logical_model_digest(
            bound_contract
        )
        weightless_binding = selected.bindings[str(weightless_plan_ref["digest"])]
        assert weightless_binding.weightless and weightless_binding.model_weight_bytes == 0
        assert selected.config_assets == {
            str(config_asset_ref["digest"]): int(config_asset_ref["length"])
        }
        assert str(config_asset_ref["digest"]) not in selected.documents
        extra_root_ref: Ref = {"digest": "sha256:" + "c" * 64, "length": 31}
        extra_layout_ref: Ref = {"digest": "sha256:" + "d" * 64, "length": 37}
        extra_object_set = provisioned.read_artifact_object_set(
            canonical.write(
                {
                    "kind": "tensorhub.resolved_object_set/1",
                    "roots": [
                        {
                            "objects": sorted(
                                [config_asset_ref, fp8_weights_ref, layout_ref],
                                key=lambda row: str(row["digest"]),
                            ),
                            "root": model_root_ref,
                        },
                        {"objects": [extra_layout_ref], "root": extra_root_ref},
                    ],
                }
            )
        )
        try:
            provisioned.read_provisioned_release(
                plan_documents,
                read_document=provisioned.staged_document_reader(binding_documents),
                package_release="package@release",
                store_root=root / "store",
                object_set=extra_object_set,
                dtype_widths={"bfloat16": 2, "float32": 4},
            )
        except PlanRefusal as exc:
            assert exc.code == "artifact_object_set_mismatch"
        else:
            raise AssertionError("an unconsumed model root was accepted")
        try:
            provisioned.read_provisioned_release(
                {str(plan_ref["digest"]): plan_documents[str(plan_ref["digest"])]},
                read_document=provisioned.staged_document_reader(binding_documents),
                package_release="package@release",
                store_root=root / "store",
                object_set=object_set,
                dtype_widths={"bfloat16": 2, "float32": 4},
            )
        except PlanRefusal as exc:
            assert exc.code == "binding_plan_set_invalid"
        else:
            raise AssertionError("incomplete visible-entrypoint plan set was accepted")

        boot_raw = bytes(range(32))
        boot_id = base64.urlsafe_b64encode(boot_raw).rstrip(b"=").decode()
        boot_path = root / "pod-boot-id"
        boot_path.write_text(boot_id + "\n")
        assert readiness.pod_boot_id(boot_path) == boot_id
        fixed_certificate = mint_certificate(root, "fixed-worker", "DNS:cozy-worker")
        assert readiness.certificate(fixed_certificate)
        ip_certificate = mint_certificate(root, "provider-ip", "IP:127.0.0.1")
        try:
            readiness.certificate(ip_certificate)
        except LaunchRefusal as exc:
            assert exc.code == "tls_certificate_invalid"
        else:
            raise AssertionError("provider IP SAN was accepted as worker identity")
        cpu_payload = readiness.receipt_payload(
            control_public_key_ed25519_b64url=config.control_public_key_ed25519_b64url,
            media_token_sha256=config.media_token_sha256,
            pod_boot_id_value=boot_id,
            certificate_der=b"certificate-der",
            worker_internal_port=config.worker_internal_port,
            media_internal_port=config.media_internal_port,
            gpus=[],
        )
        cpu_receipt = canonical.parse_canonical(cpu_payload)
        assert isinstance(cpu_receipt, dict) and cpu_receipt["runtime_gpus"] == []
        payload = readiness.receipt_payload(
            control_public_key_ed25519_b64url=config.control_public_key_ed25519_b64url,
            media_token_sha256=config.media_token_sha256,
            pod_boot_id_value=boot_id,
            certificate_der=b"certificate-der",
            worker_internal_port=config.worker_internal_port,
            media_internal_port=config.media_internal_port,
            gpus=[
                {
                    "device_index": 0,
                    "device_name": "NVIDIA H200",
                    "device_uuid": "GPU-live",
                    "driver_version": "999.0",
                    "memory_bytes": 141 << 30,
                    "pci_bus_id": "00000000:01:00.0",
                }
            ],
        )
        receipt = canonical.parse_canonical(payload)
        assert isinstance(receipt, dict)
        assert receipt["worker_listener_bound"] is True
        assert receipt["media_foreign_credential_refused"] is True
        assert receipt["pod_boot_id"] == boot_id
        assert receipt["observed_record_owner_auth"] == {
            "control_public_key_ed25519_b64url": control_key,
            "media_token_sha256": [token],
        }
        assert "provider_pod_id" not in receipt
        # An EMPTY boot has staged nothing, so the receipt asserts nothing about a closure;
        # per-placement observed state is the rev-2 wire's job, continuously. And the pod
        # measures nothing about its own image: the hub pinned that image by OCI digest,
        # so a control-Runtime digest echoed back is a value the reader already chose.
        # The identity echo went with them, for a third reason: the hub verifies this
        # payload under a per-attempt key it recovered from that attempt's own create
        # bytes, so the signature binds the attempt before any id is read.
        for retired in (
            "acquisition_attempt_id",
            "acquisition_attempt_ordinal",
            "artifact_sources",
            "measured_control_runtime_wheel_digest",
            "measured_control_runtime_wheel_length",
            "measured_installed_environment_receipt_digest",
            "network_cache_degradations",
            "provision_bundle_digest",
            "provision_spec_digest",
            "rental_id",
            "staged_placement_set_digest",
        ):
            assert retired not in receipt
        receipt_path = root / "readiness-payload"
        readiness.publish(receipt_path, payload)
        assert receipt_path.read_bytes() == payload
        readiness.publish(receipt_path, payload)
        assert receipt_path.read_bytes() == payload
        try:
            readiness.publish(receipt_path, payload + b"\n")
        except LaunchRefusal as exc:
            assert exc.code == "readiness_receipt_conflict"
        else:
            raise AssertionError("different readiness receipt overwrote the published bytes")
    print("runtime-worker contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
