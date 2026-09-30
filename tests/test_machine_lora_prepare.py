"""Captured child model overrides use the native CPU writer and real SDK census child."""

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal import (
    derive_child,
    fill,
    lora_contract,
    package_interface,
    storage_admission,
)
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.worker import machine_model_defaults, machine_model_overrides
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.machine_capture import Installation, Preparation
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config
from test_lora_composition import Value, fixture
from test_machine_execution import offer
from upload_standins import HuggingFace, Supervisor

PROJECT = """import msgspec
from cozy_runtime.author import App, AdapterCompatibility, Context, Model, uses_components, invocable
app = App()
class Request(msgspec.Struct): pass
class Result(msgspec.Struct):
    value: int
class Pipeline:
    def __init__(self, config):
        import torch
        component = torch.nn.Module()
        component.proj = torch.nn.Linear(3, 2)
        component.untouched = torch.nn.Linear(1, 1, bias=False)
        self.components = {"transformer": component}
class TinyModel(Model[Pipeline]):
    __adapter_compatibility__ = (AdapterCompatibility("lora", "", ("transformer",)),)
    def load(self, loader):
        self.pipe = loader.construct(Pipeline, factory=Pipeline)
    @uses_components("transformer")
    def sample(self):
        return 1
@invocable
async def generate(ctx: Context, *, payload: Request, model: TinyModel) -> Result:
    return Result(model.sample())
app.entrypoint(generate)
@app.job
def long_form(ctx: Context, payload: Request) -> Result:
    return Result(1)
"""


def test_captured_child_stack_builds_real_cpu_native_view_and_exact_census(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    store, base, first, second = fixture(tmp_path)
    for name, artifact in (("base", base), ("first", first), ("second", second)):
        operation = store.begin_operation("named-" + name, "proof", name)
        operation.hold_manifest(artifact.manifest.digest, artifact.manifest.length)
        operation.commit_release(
            None, "1.0.0", "fp32", artifact.manifest.digest, artifact.manifest.length
        )
    project = tmp_path / "project"
    project.mkdir()
    (project / "lora_prepare_package.py").write_text(PROJECT)
    (project / "package.toml").write_text('[application]\nobject="lora_prepare_package:app"\n')
    interface = package_interface.canonical_bytes(package_interface.build(discover(project)))
    installation = "local-" + "11" * 16
    placement = {
        "installation_id": installation,
        "development": {"package": "local/lora-prepare", "release": "1.0.0"},
        "package_interface": base64.b64encode(interface).decode(),
    }
    capture = pb.MachineExecutionCapture(
        root_installation_id=installation,
        installed_packages=[
            pb.InstalledPackage(installation_id=installation, package="local/lora-prepare")
        ],
        bindings=[
            pb.MachineCallableBinding(
                caller_installation_id=installation,
                callee_installation_id=installation,
                entrypoint="generate",
            )
        ],
    )
    choice = capture.model_choices.add(parameter="generate.models.model", repository="proof/base")
    choice.manifest.digest = documents.raw(base.manifest.digest)
    choice.manifest.length = base.manifest.length
    for name, artifact, scale in (("first", first, "0.5"), ("second", second, "-0.25")):
        choice.adapters.add(
            model="proof/" + name,
            manifest=artifact.manifest.digest,
            component="transformer",
            scale=scale,
        )
    raw, digest = documents.identity(capture)
    worker = Worker(
        _config(tmp_path / "home"),
        WorkerOptions(
            root=tmp_path / "worker",
            tensorfs_root=Path(store.root),
            devices="",
            accelerator_backend="none",
            publication_authority=PublicationAuthority("http://127.0.0.1:9", "unused", "unused"),
        ),
        InMemoryControlHost(),
    )
    try:
        assert worker.executions is not None
        worker.executions.submit(
            "owner",
            "root",
            digest,
            offer(),
            capture_document=raw,
            preparation=Preparation(
                installations={installation: Installation(placement, installation)},
                entrypoint="long_form",
                account="proof",
            ).encode(),
            expected_execution_workspace_id=worker.executions.workspace_id,
        )
        declaration = package_interface.read_bytes(interface, "test")["entrypoints"][0]
        target = Target(
            installation,
            "generate",
            declaration,
            {"placement": placement, "installation_id": installation},
            None,
            "",
        )
        call = Call("root", 0, 1, 1, b"i" * 32, b"{}", "child", b"", False, b"", "", "")
        choices = machine_model_overrides.captured(worker, "owner", call, target)
        assert set(choices) == {"model"}
        (selected,) = machine_model_defaults.select(
            worker, "owner", call, target, {}, model_choices=choices
        )
        prepared = machine_model_overrides.apply(
            worker,
            selected,
            choices["model"],
            package="local/lora-prepare",
            hub="",
            owner="proof",
            credentials={},
            note=lambda *_: None,
            check=lambda: None,
        )
        same = machine_model_overrides.apply(
            worker,
            selected,
            choices["model"],
            package="local/lora-prepare",
            hub="",
            owner="proof",
            credentials={},
            note=lambda *_: None,
            check=lambda: None,
        )
        assert same == prepared
        composed = machine_model_defaults.decode([prepared])[0].composed
        assert composed is not None
        checkpoint = fill.Checkpoint(store.root, composed.manifest)
        graph = lora_contract.read(checkpoint.header["configs"][lora_contract.GRAPH_CONFIG])
        assert [(a.scale, a.ref) for a in graph.adapters] == [
            (0.5, first.manifest.digest),
            (-0.25, second.manifest.digest),
        ]
        private_request = pb.PreparePrivatePlacementRequest(model_choices=[choice], owner="proof")
        private_request.model_choices[0].parameter = "generate.models.model"
        rows, native = machine_model_overrides.private(
            worker,
            private_request,
            pb.Placement(
                development=pb.DevelopmentPackage(package="local/lora-prepare"),
                package_interface=interface,
            ),
            check=lambda: None,
        )
        assert not native and len(rows) == 1
        assert rows[0]["slot"] == "generate.models.model"
        assert rows[0]["composed"]["manifest"] == composed.manifest
        request = derive_child.DeriveRequest(
            (
                derive_child.SlotRequest(
                    "generate.models.model",
                    b"{}",
                    adapters=checkpoint.header["configs"][lora_contract.GRAPH_CONFIG],
                ),
            ),
            project=project,
        )
        (census,) = derive_child.derive_in(Path(sys.executable), request)
        assert isinstance(census, derive_child.TensorRequirements)
        assert {row[1] for row in census.rows} == {
            row.name for row in checkpoint.rows("transformer")
        }
        assert tensorfs.fit(
            [
                tensorfs.TensorRequirement(
                    component=component, key=key, logical_dtype=dtype, shape=list(shape)
                )
                for component, key, dtype, shape in census.rows
            ],
            checkpoint.header_bytes,
            custody="canonical",
            encoded_leaves=False,
        )["ok"]
        assert worker.gpu.view() == {"leases": {}, "grants": {}, "waiting": {}}
        assert not __import__("torch").cuda.is_initialized()
    finally:
        worker.shutdown()


def test_provider_adapter_acquisition_keeps_recognizable_token_transient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    store, base, _first, _second = fixture(tmp_path)
    operation = store.begin_operation("base-published", "proof", "base")
    operation.hold_manifest(base.manifest.digest, base.manifest.length)
    operation.commit_release(None, "1.0.0", "fp32", base.manifest.digest, base.manifest.length)
    factors = {
        "proj.lora_A.weight": Value((1, 3), (1, 0, 2)),
        "proj.lora_B.weight": Value((2, 1), (3, 4)),
    }
    payload = bytearray()
    header = {}
    for key, value in factors.items():
        raw = value.bytes()
        header[key] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    metadata = json.dumps(header).encode()
    body = struct.pack("<Q", len(metadata)) + metadata + payload
    origin = HuggingFace({"lora.safetensors": bytes(body)})
    secret = "hf_cozy_lora_override_transient_token_292"
    supervisor = Supervisor(Path(store.root), 1 << 30, 0)
    worker = Worker(
        _config(tmp_path / "home"),
        WorkerOptions(
            root=tmp_path / "worker",
            tensorfs_root=Path(store.root),
            devices="",
            accelerator_backend="none",
        ),
        InMemoryControlHost(),
    )
    try:
        with origin.running() as hf, supervisor.running() as channel:
            monkeypatch.setattr(storage_admission, "_channel", channel)
            assert worker.source_calls is not None
            worker.source_calls.endpoints = {"huggingface": hf}
            choice = pb.ModelChoice(parameter="model")
            choice.adapters.add(
                source="hf://example/character@" + "c" * 40 + "/lora.safetensors",
                profiles=["as-is/1"],
                component="transformer",
                source_component="model",
                scale="0.5",
            )
            prepared = machine_model_overrides.apply(
                worker,
                {
                    "parameter": "model",
                    "repository": "proof/base",
                    "manifest": {"digest": base.manifest.digest, "length": base.manifest.length},
                },
                choice,
                package="local/lora-prepare",
                hub="",
                owner="proof",
                credentials={"huggingface": "bearer " + secret},
                note=lambda *_: None,
                check=lambda: None,
            )
            (selected,) = machine_model_defaults.decode([prepared])
            assert selected.composed is not None and len(selected.adapters) == 1
            assert any(secret in value for value in origin.authorization)
            graph = lora_contract.read(
                fill.Checkpoint(store.root, selected.composed.manifest).header["configs"][
                    lora_contract.GRAPH_CONFIG
                ]
            )
            assert len(graph.layers) == 1 and graph.layers[0].strength == 0.5
            assert worker.gpu.view() == {"leases": {}, "grants": {}, "waiting": {}}
    finally:
        worker.shutdown()
    for path in tmp_path.rglob("*"):
        if path.is_file() and not path.is_symlink():
            assert secret.encode() not in path.read_bytes(), (
                f"LoRA credential persisted in {path.relative_to(tmp_path)}"
            )
    out, err = capfd.readouterr()
    assert secret not in out + err
