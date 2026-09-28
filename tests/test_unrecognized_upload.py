"""A source no reviewed profile recognizes uploads as-is, end to end on stand-ins.

TensorFS's fingerprint registry is a hint, not a gate: an unrecognized Civitai single file
and an unrecognized Hugging Face repository both publish, recorded `normalization.state =
raw`, and the upload logs the as-is note. A package that cannot construct from such a
checkpoint refuses `checkpoint_unnormalized` naming the fix. A malformed carrier still refuses.
"""

from __future__ import annotations

import hashlib
import json
import queue
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from weakref import WeakValueDictionary

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author import Artifact, CheckpointUnnormalized, Config, Loader, ModelFitRefused
from cozy_runtime.author._loader import Backend, Census, Fit
from cozy_runtime.internal import fill, model_config, source_interfaces, storage_admission
from cozy_runtime.internal.worker import upload_plan, workspace_sources
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    Selected,
    _selected_models,
    selections,
)
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from upload_standins import Hub, HuggingFace, Supervisor

SLOT = "generate.models.model"
RAW = {"converter": "identity/1", "dialect": "safetensors", "state": "raw"}
CONFIGS = {
    "encoder": {"_class_name": "FixtureEncoder", "hidden_size": 1},
    "decoder": {"_class_name": "AutoencoderKL", "latent_channels": 4},
}


def _safetensors(tensors: dict[str, tuple[str, int]], *, truncate: int = 0) -> bytes:
    """`key -> (dtype, elements)`; payload bytes derive from the key."""
    header: dict[str, Any] = {}
    data = b""
    for key, (dtype, elements) in tensors.items():
        width = {"F16": 2, "F32": 4, "U8": 1}[dtype]
        body = bytes((len(key) + i) % 251 for i in range(elements * width))
        header[key] = {
            "dtype": dtype,
            "shape": [elements],
            "data_offsets": [len(data), len(data) + len(body)],
        }
        data += body
    raw = json.dumps(header).encode()
    return (struct.pack("<Q", len(raw)) + raw + data)[: -truncate or None]


def _command(
    workspace: Workspace, export: str, request: dict[str, object]
) -> pb.NativeSourceCommand:
    raw = documents.canonical_bytes(
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                installation_id="upload", job_descriptor_id="sha256:" + "44" * 32
            )
        )
    )
    spec = hashlib.sha256(raw).digest()
    offer = pb.AttemptOffer(
        request_id="ingest",
        attempt_ordinal=1,
        invocation_spec_digest=spec,
        invocation_spec_canonical_bytes=raw,
    )
    workspace.accept("owner", offer)
    workspace.mark_running(
        "owner",
        pb.AttemptAccepted(request_id="ingest", attempt_ordinal=1, invocation_spec_digest=spec),
    )
    intent = canonical_json.encode(
        {"module": source_interfaces.MODULE, "export": export, "request": request}
    )
    command = pb.NativeSourceCommand(
        service_id=workspace_sources.identity("owner", "ingest", 0),
        operation=pb.NATIVE_SOURCE_OPERATION_CIVITAI
        if export == "upload_civitai"
        else pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE,
        parent_call=pb.ChildCallRequest(
            parent_request_id="ingest",
            parent_attempt_ordinal=1,
            parent_invocation_spec_digest=spec,
            call_index=0,
            module=source_interfaces.MODULE,
            export=export,
            intent_digest=hashlib.sha256(intent).digest(),
            request_canonical_bytes=canonical_json.encode(request),
        ),
    )
    workspace_sources.accepted(workspace, "owner", command)
    return command


def _upload(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    export: str,
    files: dict[str, bytes],
    request: dict[str, Any],
) -> tuple[set[str], pb.NativeSourceStatus, list[dict[str, Any]]]:
    """Resolve then execute on stand-ins with the built-in registry only."""
    origin, hub, frames = HuggingFace(files), Hub(), []
    provider = "civitai" if export == "upload_civitai" else "huggingface"
    with origin.running() as url, hub.running(), Supervisor(root, 1 << 30, 0).running() as channel:
        monkeypatch.setattr(storage_admission, "_channel", channel)
        workspace = Workspace(root)
        messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={provider: url},
            progress=lambda *call: frames.append(call[-1]),
            publication=lambda *_: hub.client(),
        )
        command = _command(workspace, export, request)
        command.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
        calls.handle(command)
        resolved = messages.get(timeout=120)
        assert resolved.state == pb.NATIVE_SOURCE_STATE_RESOLVED, resolved.safe_detail
        command.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
        command.selection.CopyFrom(resolved.selection)
        calls.handle(command)
        finished = messages.get(timeout=600)
    return {row.member for row in resolved.selection.members}, finished, frames


def _checkpoint(root: Path, finished: pb.NativeSourceStatus) -> tuple[str, dict[str, Any]]:
    assert finished.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, finished.safe_detail
    digest = canonical_json.decode(finished.result_canonical_bytes)["manifest"]["digest"]
    header = fill.tensorfs_module().parse_header(
        fill.store(Workspace(root).store_root).manifest(digest)["header"]
    )
    return digest, dict(header)


def _prepare(root: Path, model: str, digest: str) -> dict[str, Selected]:
    rows = [{"package": "example/pipeline", "slot": SLOT, "model": model, "manifest": digest}]
    return _selected_models(
        "example/pipeline", selections(rows), root, lambda *_: None, construction_slots={SLOT}
    )


# Until the lock names a TensorFS whose registry is a hint, these stay an explicit skip.
needs_as_is = pytest.mark.skipif(
    not hasattr(tensorfs, "AS_IS_PROFILE"), reason="the locked TensorFS refuses unknown sources"
)


@pytest.fixture(autouse=True)
def isolated_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage_admission, "_reclaimers", WeakValueDictionary())


@needs_as_is
def test_an_unrecognized_civitai_file_uploads_as_is_and_a_package_refuses_it_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _safetensors(
        {
            "model.diffusion_model.input_blocks.0.0.weight": ("F16", 8),
            "first_stage_model.decoder.conv_in.weight": ("F16", 4),
            "unknown.scale_weight": ("F32", 1),
            "unknown.comfy_quant": ("U8", 3),
        }
    )
    root = tmp_path / "machine"
    request = {"version": 2831949, "destination": "example/as-is", "profiles": [], "file": ""}
    selected, finished, frames = _upload(
        root, monkeypatch, "upload_civitai", {"model.safetensors": body}, request
    )
    assert selected == {"civitai/files/1"}
    digest, header = _checkpoint(root, finished)
    assert json.loads(header["configs"]["normalization"]) == RAW
    assert {name: sorted(keys) for name, keys in header["components"].items()} == {
        "model": sorted(json.loads(body[8 : 8 + struct.unpack("<Q", body[:8])[0]]))
    }
    notes = [frame for frame in frames if frame.get("name") == upload_plan.AS_IS_NOTE]
    assert notes and notes[0]["fields"] == {"members": ["civitai/files/1"]}

    # A package needs construction configs this as-is checkpoint lacks: a typed refusal
    # that names the fix, not a KeyError in the package's factory.
    with pytest.raises(PreparationRefusal) as refused:
        _prepare(root, "example/as-is", digest)
    assert refused.value.code == "checkpoint_unnormalized"
    assert "normalize job" in str(refused.value)


@needs_as_is
def test_an_unrecognized_repository_uploads_one_default_file_per_folder_as_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = {"conv.weight": ("F32", 64), "conv.bias": ("F32", 8)}
    half = {"conv.weight": ("F16", 64), "conv.bias": ("F16", 8)}
    files = {
        "model_index.json": json.dumps(
            {
                "_class_name": "UnknownPipeline",
                "encoder": ["diffusers", "FixtureEncoder"],
                "decoder": ["diffusers", "AutoencoderKL"],
            }
        ).encode(),
        "legacy-single-file.safetensors": _safetensors({"everything.weight": ("F32", 4)}),
        "encoder/config.json": json.dumps(CONFIGS["encoder"]).encode(),
        "encoder/diffusion_pytorch_model.safetensors": _safetensors(weights),
        "encoder/diffusion_pytorch_model.fp16.safetensors": _safetensors(half),
        "decoder/config.json": json.dumps(CONFIGS["decoder"]).encode(),
        "decoder/diffusion_pytorch_model.fp32.safetensors": _safetensors(weights),
        "decoder/diffusion_pytorch_model.fp16.safetensors": _safetensors(half),
    }
    request = {
        "repository": "example/unknown",
        "revision": "d" * 40,
        "destination": "example/unknown-model",
        "profiles": [],
        "carriers": [],
    }
    root = tmp_path / "machine"
    selected, finished, _ = _upload(root, monkeypatch, "upload_huggingface", files, request)
    # Folders win over a loose root file; per folder the non-variant, else lowest precision.
    assert {m for m in selected if m.endswith(".safetensors")} == {
        "encoder/diffusion_pytorch_model.safetensors",
        "decoder/diffusion_pytorch_model.fp16.safetensors",
    }
    digest, header = _checkpoint(root, finished)
    assert sorted(header["components"]) == ["decoder", "encoder"]
    assert json.loads(header["configs"]["normalization"]) == RAW
    # The record is TensorFS metadata, never construction input.
    built = _prepare(root, "example/unknown-model", digest)[SLOT].construction
    assert built is not None
    constructor = json.loads(built.config_bytes)
    assert constructor == {**CONFIGS, "model_index": json.loads(files["model_index.json"])}


def test_a_malformed_carrier_still_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    truncated = _safetensors({"weight": ("F32", 16)}, truncate=4)
    request = {
        "repository": "example/broken",
        "revision": "e" * 40,
        "destination": "example/broken-model",
        "profiles": [],
        "carriers": [],
    }
    _, finished, _ = _upload(
        tmp_path / "machine",
        monkeypatch,
        "upload_huggingface",
        {"model.safetensors": truncated},
        request,
    )
    assert finished.state == pb.NATIVE_SOURCE_STATE_FAILED
    assert finished.safe_code == "CARRIER_TRUNCATED", finished.safe_detail


def test_a_construction_failing_on_an_as_is_checkpoint_names_the_fix() -> None:
    remedy = model_config.unnormalized({"normalization": json.dumps(RAW).encode()})
    artifact = Artifact("as-is", {}, Config({}), unnormalized=remedy)

    class Sdxl:  # reads the component configs an as-is single file does not carry
        def __init__(self, config: Config) -> None:
            self.components = {"unet": config.mapping()["unet"]}

    loader = Loader(artifact, owner=SimpleNamespace())
    with pytest.raises(CheckpointUnnormalized, match="normalize job") as failed:
        loader.construct(Sdxl, factory=Sdxl)
    assert isinstance(failed.value.__cause__, KeyError)

    class Verdict:  # the backend's fit over keys the construction does not find
        def fit(self, walked: Census, *, encoded_leaves: str) -> Fit:
            return {"ok": False, "code": "missing_tensor", "detail": "unet.conv_in.weight"}

    class Empty:
        def __init__(self, config: Config) -> None:
            self.components: dict[str, object] = {}

    loader = Loader(artifact, owner=SimpleNamespace(), backend=cast(Backend, Verdict()))
    with pytest.raises(ModelFitRefused, match="normalize job") as refused:
        loader.construct(Empty, factory=Empty)
    assert refused.value.code == "checkpoint_unnormalized"
    assert refused.value.fit["code"] == "missing_tensor"
