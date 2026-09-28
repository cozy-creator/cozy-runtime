"""A Civitai SDXL single file uploads normalized and self-contained.

The carrier is the real Civitai 128078 key set at tiny shapes. The machine's TensorFS
selects the reviewed SDXL profile, whose routed converter writes Diffusers keys in the
constructor's order, and names a pinned reference pipeline. The upload pins that
reference's index, configs and tokenizers beside the carrier and embeds them: configs
inline, tokenizers as assets. Real SourceCalls, upload child and TensorFS; stand-in
Civitai, HuggingFace (the reference) and Hub.
"""

from __future__ import annotations

import hashlib
import json
import queue
import struct
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from weakref import WeakValueDictionary

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal import source_interfaces, storage_admission
from cozy_runtime.internal.worker import workspace_sources
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR
from test_machine_partial_work import Machine, machine
from test_owner_memo import observed, recorded, succeeded
from upload_standins import Civitai, Hub, HuggingFace, Supervisor

FIXTURE = Path(__file__).parent / "testdata/sdxl-single-file.json"
VERSION, FILE_ID = 5550001, 5550002
CONVERTER = "sdxl.single_file/1-diffusers0.40.0-transformers5.16.1"
REFERENCE_URI = (
    "hf://stabilityai/stable-diffusion-xl-base-1.0@462165984030d82259a11f4367a4eed129e94a7b"
)
INDEX = {
    "_class_name": "StableDiffusionXLPipeline",
    "scheduler": ["diffusers", "EulerDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "tokenizer_2": ["transformers", "CLIPTokenizer"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}
CONFIGS = {
    name: {"_class_name": entry[1], "fixture": name}
    for name, entry in INDEX.items()
    if not name.startswith(("_", "tokenizer"))
}
TOKENIZERS = {
    f"{folder}/{name}": f"{folder} {name}".encode()
    for folder in ("tokenizer", "tokenizer_2")
    for name in ("merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json")
}
REFERENCE = {
    "model_index.json": json.dumps(INDEX).encode(),
    **{
        # A saved config's private `_name_or_path` names a source; it never reaches a factory.
        f"{name}/{'scheduler_config' if name == 'scheduler' else 'config'}.json": json.dumps(
            {**body, "_name_or_path": "../sdxl-vae/"}
        ).encode()
        for name, body in CONFIGS.items()
    },
    **TOKENIZERS,
    "unet/diffusion_pytorch_model.safetensors": b"never selected",
}


def carrier() -> tuple[bytes, dict[str, bytes]]:
    """The fixture key set as one F16 safetensors file with distinct finite values."""
    shapes: dict[str, list[int]] = json.loads(FIXTURE.read_bytes())["tensors"]
    header, payload, values = {}, bytearray(), {}
    for index, (key, shape) in enumerate(shapes.items()):
        count = 1
        for extent in shape:
            count *= extent
        body = b"".join(
            struct.pack("<H", (index * 131 + element * 7) % 0x7BFF) for element in range(count)
        )
        header[key] = {
            "dtype": "F16",
            "shape": shape,
            "data_offsets": [len(payload), len(payload) + len(body)],
        }
        payload += body
        values[key] = body
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + bytes(payload), values


def command(
    workspace: Workspace,
    request: dict[str, object],
    parent: str,
    export: str = "upload_civitai",
    index: int = 0,
    operation: pb.NativeSourceOperation = pb.NATIVE_SOURCE_OPERATION_CIVITAI,
) -> pb.NativeSourceCommand:
    raw = documents.canonical_bytes(
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                installation_id="upload-fixture", job_descriptor_id="sha256:" + "44" * 32
            )
        )
    )
    spec = hashlib.sha256(raw).digest()
    if index == 0:
        workspace.accept(
            "owner",
            pb.AttemptOffer(
                request_id=parent,
                attempt_ordinal=1,
                invocation_spec_digest=spec,
                invocation_spec_canonical_bytes=raw,
            ),
        )
        workspace.mark_running(
            "owner",
            pb.AttemptAccepted(request_id=parent, attempt_ordinal=1, invocation_spec_digest=spec),
        )
    intent = canonical_json.encode(
        {"module": source_interfaces.MODULE, "export": export, "request": request}
    )
    built = pb.NativeSourceCommand(
        service_id=workspace_sources.identity("owner", parent, index),
        operation=operation,
        parent_call=pb.ChildCallRequest(
            parent_request_id=parent,
            parent_attempt_ordinal=1,
            parent_invocation_spec_digest=spec,
            call_index=index,
            module=source_interfaces.MODULE,
            export=export,
            intent_digest=hashlib.sha256(intent).digest(),
            request_canonical_bytes=canonical_json.encode(request),
        ),
    )
    workspace_sources.accepted(workspace, "owner", built)
    return built


@pytest.fixture(autouse=True)
def isolated_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage_admission, "_reclaimers", WeakValueDictionary())


REQUEST = {"version": VERSION, "destination": "example/sdxl", "profiles": [], "file": ""}


def upload(
    calls: SourceCalls,
    workspace: Workspace,
    messages: queue.Queue[pb.NativeSourceStatus],
    parent: str,
) -> tuple[pb.NativeSourceStatus, pb.NativeSourceStatus | None]:
    built = command(workspace, REQUEST, parent)
    built.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
    calls.handle(built)
    resolved = messages.get(timeout=120)
    if resolved.state != pb.NATIVE_SOURCE_STATE_RESOLVED:
        return resolved, None
    built.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
    built.selection.CopyFrom(resolved.selection)
    calls.handle(built)
    return resolved, messages.get(timeout=600)


def test_a_civitai_sdxl_upload_is_normalized_with_its_reference_configs_and_tokenizers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body, values = carrier()
    origin, reference, hub = Civitai(VERSION, FILE_ID, body), HuggingFace(REFERENCE), Hub()
    root = tmp_path / "machine"
    with (
        origin.running() as civitai,
        reference.running() as hf,
        hub.running(),
        Supervisor(root, 1 << 30, 0).running() as channel,
    ):
        monkeypatch.setattr(storage_admission, "_channel", channel)
        workspace = Workspace(root)
        messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={"civitai": civitai, "huggingface": hf},
            publication=lambda *_: hub.client(),
        )
        resolved, finished = upload(calls, workspace, messages, "ingest")
        assert resolved.state == pb.NATIVE_SOURCE_STATE_RESOLVED, resolved.safe_detail
        selected = {row.member for row in resolved.selection.members}
        assert selected == {f"civitai/files/{FILE_ID}", *REFERENCE} - {
            "unet/diffusion_pytorch_model.safetensors"
        }
        assert finished is not None
        assert finished.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, finished.safe_detail
        manifest = canonical_json.decode(finished.result_canonical_bytes)["manifest"]

        _, again = upload(calls, workspace, messages, "ingest-again")
        assert again is not None and again.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED
        assert again.computation_digest == finished.computation_digest
        assert canonical_json.decode(again.result_canonical_bytes)["manifest"] == manifest

    header = checkpoint_header(workspace, manifest["digest"])
    components = header["components"]
    assert list(components) == ["text_encoder", "text_encoder_2", "unet", "vae"]
    assert [len(components[name]) for name in components] == [196, 517, 1680, 248]
    assert "conv_in.weight" in components["unet"]
    assert all(
        row["logical"]["logical_dtype"] == "f16"
        for table in components.values()
        for row in table.values()
    )
    # q|k|v are the fused in_proj's row thirds, bit for bit; its shape is [6, 2] here.
    fused = values["conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight"]
    layer = components["text_encoder_2"]
    for index, member in enumerate(("q_proj", "k_proj", "v_proj")):
        row = layer[f"text_model.encoder.layers.0.self_attn.{member}.weight"]
        assert row["logical"]["shape"] == [2, 2]
        inline = fused[index * 8 : (index + 1) * 8]
        assert row["parts"]["value"] == {"dtype": "f16", "shape": [2, 2], "inline": inline}
    assert "position_ids" not in json.dumps(list(components["text_encoder"]))

    self_contained(header)


def checkpoint_header(workspace: Workspace, digest: str) -> tensorfs.Header:
    header = tensorfs.Store.open(str(workspace.store_root)).manifest(digest)["header"]
    assert header is not None
    return tensorfs.parse_header(header)


def self_contained(header: tensorfs.Header) -> None:
    """Configs, index and marker inline; tokenizers as assets; nothing fetched to run."""
    configs = {name: json.loads(value) for name, value in header["configs"].items()}
    assert configs == {
        **CONFIGS,
        "model_index": INDEX,
        "normalization": {
            "converter": CONVERTER,
            "dialect": "diffusers",
            "reference": REFERENCE_URI,
            "state": "normalized",
        },
    }
    assets = header["assets"]
    assert sorted(assets) == sorted(TOKENIZERS)
    for name, content in TOKENIZERS.items():
        assert assets[name]["logical_sha256"] == hashlib.sha256(content).hexdigest(), name


def test_convert_cozytensors_of_a_civitai_sdxl_file_is_the_same_self_contained_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body, _ = carrier()
    origin, reference = Civitai(VERSION, FILE_ID, body), HuggingFace(REFERENCE)
    root = tmp_path / "machine"
    with (
        origin.running() as civitai,
        reference.running() as hf,
        Supervisor(root, 1 << 30, 0).running() as channel,
    ):
        monkeypatch.setattr(storage_admission, "_channel", channel)
        workspace = Workspace(root)
        messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={"civitai": civitai, "huggingface": hf},
        )
        download = command(
            workspace, {"version": VERSION, "file": ""}, "script", export="download_civitai"
        )
        download.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
        calls.handle(download)
        resolved = messages.get(timeout=120)
        assert resolved.state == pb.NATIVE_SOURCE_STATE_RESOLVED, resolved.safe_detail
        download.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
        download.selection.CopyFrom(resolved.selection)
        calls.handle(download)
        source = messages.get(timeout=120)
        assert source.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, source.safe_detail
        request = {
            "source": canonical_json.decode(source.result_canonical_bytes),
            "profiles": ["civitai/sdxl/single-file/1"],
        }
        convert = command(
            workspace,
            request,
            "script",
            export="convert_cozytensors",
            index=1,
            operation=pb.NATIVE_SOURCE_OPERATION_CONVERT,
        )
        convert.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
        calls.handle(convert)
        converted = messages.get(timeout=600)
        assert converted.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, calls.failure_detail(
            convert.service_id
        )
        manifest = canonical_json.decode(converted.result_canonical_bytes)["manifest"]
    header = checkpoint_header(workspace, manifest["digest"])
    assert [len(table) for table in header["components"].values()] == [196, 517, 1680, 248]
    self_contained(header)


def test_an_unreachable_reference_refuses_the_upload_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body, _ = carrier()
    origin, hub = Civitai(VERSION, FILE_ID, body), Hub()
    reference = HuggingFace({"README.md": b"no pipeline index here"})
    root = tmp_path / "machine"
    with (
        origin.running() as civitai,
        reference.running() as hf,
        hub.running(),
        Supervisor(root, 1 << 30, 0).running() as channel,
    ):
        monkeypatch.setattr(storage_admission, "_channel", channel)
        workspace = Workspace(root)
        messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={"civitai": civitai, "huggingface": hf},
            publication=lambda *_: hub.client(),
        )
        resolved, _ = upload(calls, workspace, messages, "ingest")
    assert resolved.state == pb.NATIVE_SOURCE_STATE_FAILED
    assert resolved.safe_code == "model_reference_unavailable"
    assert "stabilityai/stable-diffusion-xl-base-1.0@462165984030" in resolved.safe_detail


SCRIPT = """import asyncio
from pathlib import Path
import msgspec
from cozy_runtime.author import App, Context
from cozy_runtime.author.sources import upload_civitai
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    checkpoint: str

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    while not Path(GATE).exists():
        await asyncio.sleep(0.02)
    ref = await upload_civitai(VERSION, destination='example/sdxl')
    return Result(ref.checkpoint)
""".replace("VERSION", str(VERSION))


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_a_repeated_civitai_upload_is_the_owners_memo_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rented pods, the owner's memo: the second pod asks with the same computation (which
    names the versioned converter) and takes the recorded checkpoint without a source read."""
    body, _ = carrier()
    gate, hub = tmp_path / "gate", Hub()
    origin, reference = Civitai(VERSION, FILE_ID, body), HuggingFace(REFERENCE)
    with origin.running() as civitai, reference.running() as hf, hub.running():

        def pod() -> tuple[AbstractContextManager[Machine], Machine]:
            rented = machine(monkeypatch, SCRIPT.replace("GATE", repr(str(gate))))
            machine_ = rented.__enter__()
            calls = machine_.worker.source_calls
            assert calls is not None
            calls.endpoints = {"civitai": civitai, "huggingface": hf}
            calls.publication = lambda *_: hub.client()
            machine_.start(())
            return rented, machine_

        context, first = pod()
        try:
            owner = observed(first, gate, "computed", lambda *_: None, owner_memo=False)
            checkpoint = succeeded(first, "computed")
            assert owner.kinds() == ["memo.record"]
            (record,) = [body for kind, body in owner.events if kind == "memo.record"]
        finally:
            context.__exit__(None, None, None)
        context, fresh = pod()
        asked_at: list[int] = []
        answer = recorded(canonical_json.encode(record["result"]))

        def remember(body: dict[str, Any], sequence: int) -> pb.MachineMemoAnswer:
            asked_at.append(origin.requests + reference.requests)
            return answer(body, sequence)

        try:
            owner = observed(fresh, gate, "reused", remember)
            assert succeeded(fresh, "reused") == checkpoint
            assert owner.kinds() == ["memo.lookup"] and not owner.refusals
            (lookup,) = [body for kind, body in owner.events if kind == "memo.lookup"]
            assert lookup["computation_digest"] == record["computation_digest"]
        finally:
            context.__exit__(None, None, None)
        # Pinning precedes the lookup; after the answer no provider is asked again.
        assert asked_at == [origin.requests + reference.requests]
