"""A generic Hugging Face Diffusers upload carries each component's construction config.

A reviewed profile converts tensors only. The pipeline's own index names its components;
each model's ``config.json`` and each scheduler's ``scheduler_config.json`` at the same
pinned revision ride as the checkpoint's inline configs, so a package can construct it.
"""

from __future__ import annotations

import hashlib
import json
import queue
from pathlib import Path
from weakref import WeakValueDictionary

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author._loader import Config
from cozy_runtime.internal import source_interfaces, storage_admission
from cozy_runtime.internal.worker import workspace_sources
from cozy_runtime.internal.worker.package_prepare import _selected_models
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from upload_standins import Hub, HuggingFace, Supervisor

FIXTURE = Path(__file__).parent / "testdata/native_source"
PROFILE = "fixture/diffusers/1"
SLOT = "generate.models.model"
CONFIGS = {
    "encoder": {"_class_name": "FixtureEncoder", "hidden_size": 1},
    "decoder": {"_class_name": "AutoencoderKL", "latent_channels": 4},
    "scheduler": {"_class_name": "EulerDiscreteScheduler", "num_train_timesteps": 1000},
}


def _registry() -> bytes:
    """The tiny native fixture's reviewed entries, as one Diffusers pipeline's profile."""
    registry = json.loads((FIXTURE / "registry.json").read_bytes())
    (variant,) = registry["source_profiles"][0]["components"][0]["variants"]
    registry["source_profiles"] = [
        {
            "name": PROFILE,
            "components": [
                {
                    "component": name,
                    "source_member": f"{name}/diffusion_pytorch_model.safetensors",
                    "target_encoding": "plain/1",
                    "variants": [variant],
                }
                for name in ("encoder", "decoder")
            ],
        }
    ]
    return json.dumps(registry).encode()


def _repository(modular: bool) -> dict[str, bytes]:
    weights = (FIXTURE / "first.safetensors").read_bytes()
    size = 8 + int.from_bytes(weights[:8], "little")
    other = weights[:size] + bytes(255 - byte for byte in weights[size:])  # same tensor, new values
    classes: dict[str, list[object]] = {
        "encoder": ["diffusers", "FixtureEncoder"],
        "decoder": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "EulerDiscreteScheduler"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
    }
    if modular:  # a modular pipeline names each component's subfolder and class hint
        index_name = "modular_model_index.json"
        classes = {
            name: [library, kind, {"subfolder": name, "type_hint": [library, kind]}]
            for name, (library, kind) in classes.items()
        }
    else:
        index_name = "model_index.json"
    return {
        index_name: json.dumps({"_class_name": "FixturePipeline", **classes}).encode(),
        "README.md": b"# fixture pipeline\n",
        "encoder/config.json": json.dumps(CONFIGS["encoder"]).encode(),
        "encoder/diffusion_pytorch_model.safetensors": weights,
        "decoder/config.json": json.dumps(CONFIGS["decoder"]).encode(),
        "decoder/diffusion_pytorch_model.safetensors": other,
        "scheduler/scheduler_config.json": json.dumps(CONFIGS["scheduler"]).encode(),
        "tokenizer/tokenizer_config.json": b'{"model_max_length": 77}',
        "tokenizer/vocab.json": b'{"a": 0}',
    }


def _command(workspace: Workspace, request: dict[str, object]) -> pb.NativeSourceCommand:
    raw = documents.canonical_bytes(
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                installation_id="upload-fixture", job_descriptor_id="sha256:" + "44" * 32
            )
        )
    )
    spec = hashlib.sha256(raw).digest()
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id="ingest",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=raw,
        ),
    )
    workspace.mark_running(
        "owner",
        pb.AttemptAccepted(request_id="ingest", attempt_ordinal=1, invocation_spec_digest=spec),
    )
    intent = canonical_json.encode(
        {"module": source_interfaces.MODULE, "export": "upload_huggingface", "request": request}
    )
    command = pb.NativeSourceCommand(
        service_id=workspace_sources.identity("owner", "ingest", 0),
        operation=pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE,
        parent_call=pb.ChildCallRequest(
            parent_request_id="ingest",
            parent_attempt_ordinal=1,
            parent_invocation_spec_digest=spec,
            call_index=0,
            module=source_interfaces.MODULE,
            export="upload_huggingface",
            intent_digest=hashlib.sha256(intent).digest(),
            request_canonical_bytes=canonical_json.encode(request),
        ),
    )
    workspace_sources.accepted(workspace, "owner", command)
    return command


@pytest.fixture(autouse=True)
def isolated_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage_admission, "_reclaimers", WeakValueDictionary())


NAMED = {
    "carriers": [f"{name}/diffusion_pytorch_model.safetensors" for name in ("decoder", "encoder")],
    "profiles": [PROFILE],
}


@pytest.mark.parametrize(
    ("modular", "named"),
    [(False, NAMED), (True, NAMED), (False, {})],
    ids=["model_index", "modular_model_index", "profile_selected_on_the_machine"],
)
def test_a_diffusers_upload_carries_each_components_config_and_prepares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modular: bool, named: dict[str, object]
) -> None:
    files = _repository(modular)
    request = {
        "repository": "example/diffusers",
        "revision": "c" * 40,
        "destination": "example/diffusers-model",
        # With nothing named, the machine's TensorFS selects the profile from the headers.
        **named,
    }
    origin, hub, root = HuggingFace(files), Hub(), tmp_path / "machine"
    with origin.running() as hf, hub.running(), Supervisor(root, 1 << 30, 0).running() as channel:
        monkeypatch.setattr(storage_admission, "_channel", channel)
        workspace = Workspace(root)
        messages: queue.Queue[pb.WorkerFrame] = queue.Queue()
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={"huggingface": hf},
            native_registry=_registry(),
            publication=lambda *_: hub.client(),
        )
        command = _command(workspace, request)
        command.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
        calls.handle(command)
        resolved = messages.get(timeout=120).native_source_status
        assert resolved.state == pb.NATIVE_SOURCE_STATE_RESOLVED, resolved.safe_detail
        selected = {row.member for row in resolved.selection.members}
        assert {"encoder/config.json", "scheduler/scheduler_config.json"} <= selected
        assert not any(member.startswith("tokenizer/") for member in selected)
        assert {m for m in selected if m.endswith(".safetensors")} == set(NAMED["carriers"])
        command.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
        command.selection.CopyFrom(resolved.selection)
        calls.handle(command)
        finished = messages.get(timeout=600).native_source_status
        assert finished.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, finished.safe_detail
        manifest = canonical_json.decode(finished.result_canonical_bytes)["manifest"]

    # The package's preparation reads the checkpoint as a rented machine's would.
    models = _selected_models(
        "example/pipeline",
        [
            {
                "package": "example/pipeline",
                "slot": SLOT,
                "model": "example/diffusers-model",
                "manifest": manifest["digest"],
            }
        ],
        root,
        lambda *_: None,
        construction_slots={SLOT},
    )
    constructor = json.loads(models[SLOT]["config_bytes"])
    assert constructor == CONFIGS
    Config(constructor)  # carries no source carrier a factory could read
