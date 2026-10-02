"""A small AdaLN-pruned H3 DiT with a real prepared LoRA view, served the executor's way.

Real: a TensorFS store holding a bf16/fp32 H3 checkpoint and an fp32 LoRA checkpoint, the
production adapter-view child composing them, `lora_composition.bind`, the serving substrate,
the weight plane's `PlaneBackend` and `WeightResidency` (a CPU executor computes on its pinned
tier), and the group's real follower `Executor`. Run as a script, this is one follower rank.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import socket
from pathlib import Path
from typing import Any

import tensorfs
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author import (
    AdapterCompatibility,
    Artifact,
    Config,
    Loader,
    Model,
    ModelRegistry,
    ObjectRef,
    uses_components,
)
from cozy_runtime.internal import adapter_view_child, attention, fill, lora_composition
from cozy_runtime.internal.derive import Observations, serving_substrate
from cozy_runtime.internal.encoding import SPEC_PLAIN, DeviceFacts
from cozy_runtime.internal.executor import Executor, _capture_seal
from cozy_runtime.internal.executor_commands import decode
from cozy_runtime.internal.fill import Checkpoint, tensor_schema_of
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.weights import PlaneBackend, WeightResidency, Weights
from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.internal.worker.source_steps import (
    AdapterViewInput,
    AdapterViewReady,
    Launch,
    PrepareAdapterView,
)
from cozy_runtime.models.minimax_h3.adaln_pruned import AdaLNPrunedMiniMaxH3Transformer
from cozy_runtime.models.minimax_h3.official import _apply_transformer_dtype
from cozy_runtime.models.minimax_h3.turbo import LORA_FAMILIES

COMPONENT = "fl2va_dit"
TIMES = (0.0, 0.25, 0.75)
KEYS = [(row, tag) for row in range(3) for tag in range(3)]
#: H3's 56 heads, so every declared degree divides them; widths kept small for CPU.
CONFIG = {
    "num_attention_heads": 56,
    "attention_head_dim": 16,
    "hidden_size": 64,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 96,
    "in_channels": 2,
    "audio_in_channels": 2,
    "patch_size": (1, 1, 1),
    "text_dim": 4,
    "freq_dim": 8,
    "time_embed_hidden_dim": 16,
    "time_embed_dim": 8,
    "rope_freq_dim": 2,
}
RANK, ALPHA, STRENGTH = 4, 8.0, 0.8
SCALING = STRENGTH * ALPHA / RANK
_DTYPES = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16"}


def dit() -> Any:
    """The served structure and H3's mixed bf16/fp32 dtype policy (`_build_dit`)."""
    return _apply_transformer_dtype(
        AdaLNPrunedMiniMaxH3Transformer(table_timesteps=TIMES, table_block_keys=KEYS, **CONFIG)
    )


class Pipe:
    def __init__(self, config: Config) -> None:
        self.components = {COMPONENT: dit()}


class LoRAH3(Model[Pipe], encoded_leaves="accept"):
    __adapter_compatibility__ = (
        AdapterCompatibility("lora", "", (COMPONENT,), -math.inf, math.inf),
    )
    pipe: Pipe

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipe, factory=Pipe)

    @uses_components(COMPONENT)
    def denoise(self, inputs: dict[str, Any]) -> tuple[Any, Any]:
        with torch.inference_mode():
            return self.pipe.components[COMPONENT](**inputs)  # type: ignore[no-any-return]


def _write(store: Any, name: str, component: str, tensors: dict[str, Any]) -> ObjectRef:
    model = component != "adapter"
    grant = sum(value.numel() * value.element_size() for value in tensors.values()) + (1 << 20)
    writer = store.begin_derived(
        weights_transaction_id(name, "create", "sha256:" + "1" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={
                component: Target(
                    add={
                        key: Tensor(
                            _DTYPES[value.dtype],
                            tuple(value.shape),
                            SPEC_PLAIN,
                            {"value": Part(_DTYPES[value.dtype], tuple(value.shape))},
                        )
                        for key, value in tensors.items()
                    }
                )
            },
            configs={"model": NativeConfig("add")} if model else {},
            order=tuple((component, key) for key in tensors),
        ).native_arguments(grant),
        work_fingerprint="sha256:" + "2" * 64,
    )
    for key, value in tensors.items():
        data = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        writer.add_part(component, key, "value", io.BytesIO(data))
    if model:
        writer.add_config("model", io.BytesIO(b"{}"))
    receipt = writer.commit()["manifest"]
    return ObjectRef("sha256:" + receipt["sha256"], receipt["length"])


def checkpoints(
    root: Path, dtype: Any = torch.float32
) -> tuple[ObjectRef, dict[str, Any], dict[str, Any]]:
    """Base and `dtype` LoRA checkpoints in a real store, composed by the production child."""
    store = tensorfs.Store.ensure(str(root))
    torch.manual_seed(0)
    base = dit()
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.normal_(0, 0.3)
    values = dict(base.state_dict())
    generator = torch.Generator().manual_seed(1)
    factors: dict[str, Any] = {}
    for path in targets(base):
        layer = base.get_submodule(path)
        shapes = {"A": (RANK, layer.in_features), "B": (layer.out_features, RANK)}
        for role, shape in shapes.items():
            factors[f"{path}.lora_{role}.weight"] = torch.randn(shape, generator=generator)
        factors[path + ".alpha"] = torch.tensor(ALPHA)
    factors = {
        key: (value * 0.3).to(dtype) if key.endswith(".weight") else value
        for key, value in factors.items()
    }
    step = PrepareAdapterView(
        identity="sha256:" + "3" * 64,
        base=_write(store, "base", COMPONENT, values),
        adapters=(
            AdapterViewInput(
                _write(store, "adapter", "adapter", factors), COMPONENT, "adapter", str(STRENGTH)
            ),
        ),
    )
    answer = adapter_view_child.execute(Launch(store=str(root), parent_pid=os.getpid(), step=step))
    assert isinstance(answer, AdapterViewReady), answer
    return answer.manifest, values, factors


def targets(root: Any) -> list[str]:
    return [
        path
        for path, _ in root.named_modules()
        if path.startswith("transformer_blocks.") and path.endswith(LORA_FAMILIES)
    ]


def construct(root: Path, view: str) -> tuple[LoRAH3, PlaneBackend]:
    """Exactly the executor's prepare, on a CPU executor: bind the prepared view, construct
    weightless under the serving substrate, register every weight with the plane."""
    checkpoint = Checkpoint(root, view)
    rows = checkpoint.rows(COMPONENT)
    graph = fill.tensorfs_module().parse_header(checkpoint.header_bytes)["configs"]
    artifact = lora_composition.bind(
        Artifact(view, tensor_schema_of(rows), Config({})),
        LoRAH3,
        graph[lora_composition.GRAPH_CONFIG],
        prepared_snapshot=view,
    )
    weights = Weights(torch, torch.device("cpu"), "cpu")
    backend = PlaneBackend.for_script(
        {COMPONENT: checkpoint},
        rows,
        weights=weights,
        construction="lora-runtime",
        release="lora-runtime/1",
    )
    backend.expect({COMPONENT: [row.key for row in rows]})
    seen = Observations()
    registry = ModelRegistry(
        release="lora-runtime/1",
        backend=backend,
        substrate=lambda: serving_substrate("cpu", seen),
    )
    model = registry.acquire("run.models.model", LoRAH3, artifact)
    residency = WeightResidency(
        weights, backend.components, {"denoise": (COMPONENT,)}, refine=backend.refine
    )
    object.__setattr__(model, "_cozy_residency", residency)
    roots = {COMPONENT: model.pipe.components[COMPONENT]}
    attention.install(attention.available(DeviceFacts("cpu", "CPU", 0, "", "", 0)), roots)
    return model, backend


def adapters(model: LoRAH3) -> dict[str, Any]:
    """Every PEFT layer's factors and scaling, read inside the component's stage."""
    root = model.pipe.components[COMPONENT]
    residency = model._cozy_residency
    assert residency is not None
    residency.admit("denoise", (COMPONENT,))
    try:
        held: dict[str, Any] = {}
        for path in targets(root):
            layer = root.get_submodule(path)
            held[path] = (
                layer.lora_A["adapter_0"].weight.detach().clone(),
                layer.lora_B["adapter_0"].weight.detach().clone(),
                layer.scaling["adapter_0"],
                tuple(layer.active_adapters),
            )
        return held
    finally:
        residency.release("denoise", (COMPONENT,))


def digest(held: dict[str, Any]) -> str:
    sha = hashlib.sha256()
    for path, (a, b, scaling, active) in sorted(held.items()):
        sha.update(path.encode() + repr((scaling, active)).encode())
        sha.update(a.contiguous().view(torch.uint8).numpy().tobytes())
        sha.update(b.contiguous().view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def inputs() -> dict[str, Any]:
    # Gloo gathers equal shards only: 32 packed rows, which every tested degree divides.
    position, tags, video, audio, text, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=torch.ones(8, dtype=torch.long),
        num_latent_frames=3,
        latent_height=2,
        latent_width=2,
        num_audio_latents=6,
        patch_size=(1, 1, 1),
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
        keyframe_anchors=(),
    )
    generator = torch.Generator().manual_seed(2)
    return {
        "hidden_states": torch.randn(1, len(video), 2, generator=generator),
        "audio_hidden_states": torch.randn(1, len(audio), 2, generator=generator),
        "encoder_hidden_states": torch.randn(1, len(text), 4, generator=generator),
        "timestep": torch.tensor([0.75, 0.25]),
        "timestep_indices": torch.where(tags == 2, 1, 0),
        "token_tags": tags,
        "position_ids": position,
        "video_indices": video,
        "audio_indices": audio,
        "text_indices": text,
        "return_dict": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for flag in ("--root", "--leader", "--view"):
        parser.add_argument(flag)
    for flag in ("--rank", "--world", "--rank-fd"):
        parser.add_argument(flag, type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    _capture_seal()
    root = Path(args.root)
    model, backend = construct(root / "store", args.view)
    (root / f"adapters-{args.rank}.txt").write_text(digest(adapters(model)))
    channel = Channel(socket.socket(fileno=args.rank_fd))
    executor: Any = Executor(channel, root, rank=args.rank, world=args.world)
    executor.device_kind = "cpu"
    executor.residency = model._cozy_residency
    executor._group_model_key = "fixture"
    executor.backend = backend
    executor.torch = torch
    executor.ready = True
    while (command := channel.recv()) is not None:
        name = command["cmd"]
        if name == "shutdown":
            return
        if name == "join":
            reply = executor.join(decode(command))
            if reply.get("ok"):
                refused = executor._install_group(torch, model)
                assert refused is None, refused
        elif name == "run":
            reply = executor.run(command)
        else:
            raise AssertionError(name)
        channel.send({**reply, "reply": name})


if __name__ == "__main__":
    main()
