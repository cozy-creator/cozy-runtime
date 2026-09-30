"""Small real models for rank-call integration; checkpoint preparation is a separate gate."""

from __future__ import annotations

import argparse
import io
import os
import socket
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from cozy_runtime.author import Model, placeable, uses_components
from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout, attention_scope
from cozy_runtime.internal import attention
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.executor import Executor, _capture_seal
from cozy_runtime.internal.executor_commands import decode
from cozy_runtime.internal.parallel.cp import in_gated_call
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.models.minimax_h3.vae_tiles import TileBatchedVideoVAE


class Collective(torch.nn.Module):  # type: ignore[misc]  # torch is optional in the base checking env
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((1,), 2.0))

    def forward(
        self,
        value: Any,
        *,
        check: bool = False,
        fail: bool = False,
        expected_stages: int = 0,
        step: int | None = None,
    ) -> Any:
        assert in_gated_call()
        expected_layout = None if step is None else AttentionLayout(29, 9, step, 1, ("blocks.0",))
        assert _ACTIVE_LAYOUT.get() == expected_layout
        assert not torch.is_grad_enabled()
        torch.testing.assert_close(self.weight.cpu(), torch.tensor([2.0]))
        if check:
            torch.testing.assert_close(value.cpu(), torch.tensor([3.0, 5.0, 9.0]))
            return value
        if fail and torch.distributed.get_rank() > 0:
            raise ValueError("follower failed before joining the collective")
        result = value.detach().cpu().clone() + torch.distributed.get_rank()
        torch.distributed.all_reduce(result)
        return result


class ModelCalls(Model[object]):
    block: Any
    overlay: Any
    spare: Any

    @uses_components("block", "overlay")
    def sample(self, state: Any, *, on_step: Any, cancel: Any) -> Any:
        # Rich mutable state and callbacks are deliberately not rank-wire values.
        with torch.no_grad():
            self.block(state.value, check=True, expected_stages=state.expected_stages)
            for step in range(2):
                assert not cancel()
                with attention_scope(AttentionLayout(29, 9, step, 1, ("blocks.0",))):
                    result = self.block(
                        state.value, expected_stages=state.expected_stages, step=step
                    )
                assert _ACTIVE_LAYOUT.get() is None
                degree = torch.distributed.get_world_size()
                torch.testing.assert_close(
                    result, (state.value + 1) * degree + degree * (degree - 1) // 2
                )
                state.result = result
                on_step(step)
        return state.result

    @uses_components("block", "overlay")
    def fail_before_collective(self) -> None:
        with torch.no_grad():
            self.block(torch.ones(1), fail=True)


def collective_model(root: Path | None = None) -> ModelCalls:
    overlay, spare = torch.nn.Module(), torch.nn.Module()
    overlay.weight = torch.nn.Parameter(torch.ones(1))
    spare.weight = torch.nn.Parameter(torch.full((1,), 5.0))
    model = ModelCalls.for_test(block=Collective(), overlay=overlay, spare=spare)
    if root is not None:
        fill_residency(root, model)

    def check_scope(module: Any, args: Any, kwargs: Any) -> Any:
        assert model._cozy_active is not None
        assert model._cozy_active[1] == ("block", "overlay")
        residency: Any = model._cozy_residency
        if residency is not None:
            assert residency.placement == "component_staged"
            assert residency.headroom == 37
            assert residency.scope_headrooms == {"sample": 11, "fail_before_collective": 13}
            assert set(residency.backend.components) == {"block", "overlay"}
            # Multiple DiT forwards retain their weights; a per-call open_attempt would
            # erase the first call's two stages and this accumulated observation.
            assert residency.attempt_stages == kwargs.get("expected_stages", 0)
        return (args[0].to(model.overlay.weight.device) + model.overlay.weight, *args[1:]), kwargs

    model.block.register_forward_pre_hook(check_scope, with_kwargs=True)
    return model


def fill_residency(root: Path, model: ModelCalls) -> None:
    """Materialize three real native TensorFS components, then park the two used by DiT."""
    import tensorfs

    from cozy_runtime.author._loader import census
    from cozy_runtime.internal.encoding import SPEC_PLAIN
    from cozy_runtime.internal.fill import Checkpoint, StreamingFillBackend, _Holder
    from cozy_runtime.internal.residency import ComponentResidency

    root.mkdir(parents=True, exist_ok=True)
    store = tensorfs.Store.ensure(str(root / "store"))
    values = {"block": 2.0, "overlay": 1.0, "spare": 5.0}
    target = Tensor("f32", (1,), SPEC_PLAIN, {"value": Part("f32", (1,))})
    writer = store.begin_derived(
        weights_transaction_id("parallel-test", "create", "sha256:" + "1" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={name: Target(add={"weight": target}) for name in values},
            configs={"model": NativeConfig("add")},
            order=tuple((name, "weight") for name in values),
        ).native_arguments(4096),
        work_fingerprint="sha256:" + "2" * 64,
    )
    for name, value in values.items():
        writer.add_part(name, "weight", "value", io.BytesIO(struct.pack("<f", value)))
    writer.add_config("model", io.BytesIO(b"{}"))
    receipt = writer.commit()
    manifest = "sha256:" + receipt["manifest"]["sha256"]
    checkpoint = Checkpoint(root / "store", manifest)
    backend = StreamingFillBackend.for_script(
        checkpoint,
        [row for name in values for row in checkpoint.rows(name)],
        release="parallel-test/1",
        store="fixture",
        snapshot=manifest,
        device="cuda",
        encoded_leaves="refuse",
        window_bytes=1 << 20,
        slots=2,
        readers=1,
        inflight=1,
    )
    holder = _Holder({name: getattr(model, name) for name in values})
    walked = census(holder)
    backend.expect(
        {name: [d.key for d in walked.destinations if d.component == name] for name in values}
    )
    backend.resident_budget = 12
    backend.hold_lease = True
    live = backend.materialize(holder, walked)
    for destination in walked.destinations:
        backend.fill(destination.key, destination.spec, live[destination.key])
    backend.commit()
    backend.evict("block")
    backend.evict("overlay")
    residency = ComponentResidency(backend, torch)
    object.__setattr__(model, "_cozy_residency", residency)


class H3Calls(Model[object]):
    dit: Any
    text_encoder: Any

    def warm(self, ctx: Any) -> None:
        ctx.raise_if_cancelled()
        self.warm_dit()

    @uses_components("dit")
    def warm_dit(self) -> None:
        with torch.no_grad():
            self.dit(**h3_batch())

    @uses_components("dit")
    def compute(self, batch: dict[str, Any], expected: tuple[Any, Any]) -> Any:
        with torch.no_grad():
            result = self.dit(**batch)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
        return result


class H3LoRACalls(H3Calls):
    turbo_overlay: torch.nn.Module

    @uses_components("dit", "turbo_overlay")
    def warm_dit(self) -> None:
        with torch.no_grad():
            self.dit(**h3_turbo_inputs(h3_batch(), self.turbo_overlay))

    @uses_components("dit", "turbo_overlay")
    def compute(
        self, batch: dict[str, object], expected: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            result = self.dit(**h3_turbo_inputs(batch, self.turbo_overlay))
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
        assert self.dit._arming is None
        assert self.dit.proj_out.base_layer._armed is None
        assert self.dit.audio_proj_out.base_layer._armed is None
        return cast("tuple[torch.Tensor, torch.Tensor]", result)


def h3_turbo_inputs(batch: dict[str, object], overlay: torch.nn.Module) -> dict[str, object]:
    from cozy_runtime.models.minimax_h3.turbo import ATTENTION_KWARG, OVERLAY_KWARG, TURBO_BANK

    tags = cast("torch.Tensor", batch["token_tags"])
    # Text/video share the video sigma; audio uses its paired sigma. No terminal
    # origin is evaluated here, matching the ordinary H3 Turbo forward contract.
    return {
        **batch,
        "timestep": torch.tensor([0.75, 0.25]),
        "timestep_indices": torch.where(tags == 2, 1, 0),
        "attention_kwargs": {ATTENTION_KWARG: TURBO_BANK, OVERLAY_KWARG: overlay},
    }


def h3_lora_model(dit: torch.nn.Module) -> H3LoRACalls:
    from cozy_runtime.author import AdapterRef
    from cozy_runtime.internal import lora_composition, lora_contract
    from cozy_runtime.models.minimax_h3.turbo import TurboOverlay, TurboSchedule

    dit.install_lora_consumers()
    overlay = TurboOverlay(
        hidden_size=16,
        inner_dim=56 * 16,
        ffn_dim=32,
        num_layers=2,
        num_refiner_layers=1,
        video_out=2,
        audio_out=2,
        rank=1,
        alpha=1,
        schedule=TurboSchedule((0.75, 0.0), (0.25, 0.0)),
        table_timesteps=(0.0, 0.25, 0.75),
        table_block_keys=[(row, tag) for row in range(3) for tag in range(3)],
        block_table_dtype=torch.float32,
        final_table_dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        for parameter in overlay.parameters():
            parameter.normal_(0, 0.1)
    paths = (
        "proj_in",
        "audio_proj_in",
        "context_embedder",
        "token_refiner.refiner_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.1.attn.to_q",
        "transformer_blocks.0.ff.net.2",
        "proj_out",
        "audio_proj_out",
    )
    rows = tuple(
        lora_contract.Linear("dit", path, f"adapter_{index}", rank, alpha, strength, dtype)
        for path in paths
        for index, (rank, alpha, strength, dtype) in enumerate(
            ((1, 2.0, 0.5, "f32"), (2, 1.0, -0.25, "f16"))
        )
    )
    graph = lora_contract.Graph(
        lora_contract.FORMAT,
        rows,
        tuple(
            AdapterRef("sha256:" + digest * 32, strength, "lora", "dit", "adapter")
            for digest, strength in (("12", 0.5), ("34", -0.25))
        ),
    )
    lora_composition.composer(graph)(SimpleNamespace(components={"dit": dit}))
    with torch.no_grad():
        for row in rows:
            layer = dit.get_submodule(row.target)
            layer.lora_A[row.adapter].weight.normal_(0, 0.1)
            layer.lora_B[row.adapter].weight.normal_(0, 0.1)
    return H3LoRACalls.for_test(dit=dit, text_encoder=torch.nn.Linear(4, 4), turbo_overlay=overlay)


def h3_model(*, adapters: bool = False) -> H3Calls:
    from cozy_runtime.models.minimax_h3.adaln_pruned import AdaLNPrunedMiniMaxH3Transformer

    torch.manual_seed(231)
    # The served AdaLN-pruned DiT with H3's 56 heads, so every declared degree divides them.
    dit = AdaLNPrunedMiniMaxH3Transformer(
        table_timesteps=(0.0, 0.25, 0.75),
        table_block_keys=[(row, tag) for row in range(3) for tag in range(3)],
        num_attention_heads=56,
        attention_head_dim=16,
        hidden_size=16,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=4,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=2,
    ).eval()
    with torch.no_grad():
        for table in (dit.norm_out.table, *(b.adaln_proj.table for b in dit.transformer_blocks)):
            table.normal_()
    # Runtime's own selection: its SDPA runs through the Ulysses wrapper on sharded sites.
    attention.install(attention.available(DeviceFacts("cpu", "CPU", 0, "", "", 0)), {"dit": dit})

    def no_grad(module: Any, args: Any) -> None:
        assert not torch.is_grad_enabled()

    dit.register_forward_pre_hook(no_grad)
    if adapters:
        return h3_lora_model(dit)
    return H3Calls.for_test(dit=dit, text_encoder=torch.nn.Linear(4, 4))


#: settled component bytes for the hosting plan: rank 0's capacity holds the DiT and one of the
#: other two, a follower holds the conditioner beside its DiT shard
HOSTED_SIZES = {"dit": 100, "text_encoder": 60, "vae": 30}
HOSTED_CAPACITY = 170


class Conditioner(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, x: Any) -> Any:
        # The process that computed it rides along, so the caller can tell where it ran.
        return BaseModelOutputWithPast(
            last_hidden_state=self.proj(x), hidden_states=(torch.tensor([os.getpid()]),)
        )


@placeable("text_encoder")
class HostedCalls(H3Calls):
    vae: Any

    @uses_components("text_encoder")
    def encode(self, x: Any) -> Any:
        with torch.no_grad():
            return self.text_encoder(x)


def hosted_model() -> HostedCalls:
    base = h3_model()
    torch.manual_seed(7)
    return HostedCalls.for_test(dit=base.dit, text_encoder=Conditioner(), vae=torch.nn.Linear(2, 2))


class SpreadCalls(H3Calls):
    vae: Any

    @uses_components("vae")
    def decode(self, z: Any) -> Any:
        with torch.no_grad():
            return torch.cat(list(self.vae.decode_chunks(z)), dim=2)

    @uses_components("vae")
    def encode(self, x: Any) -> Any:
        with torch.no_grad():
            return self.vae.encode(x, return_dict=False)[0].parameters


def tiny_vae() -> Any:
    """H3's video VAE at a few thousand parameters: the same temporal clips and cross-fade."""
    torch.manual_seed(3)
    return TileBatchedVideoVAE(
        latent_channels=4,
        block_out_channels=(8,) * 6,
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=1,
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
    ).eval()


def spread_model() -> SpreadCalls:
    base = h3_model()
    return SpreadCalls.for_test(dit=base.dit, text_encoder=base.text_encoder, vae=tiny_vae())


def h3_batch(*, mixed_lengths: bool = False) -> dict[str, Any]:
    # Interleaved modalities exercise packed indices across each shard. Gloo cannot
    # gather uneven tensor sizes, so both packings are 56 rows, which every declared degree
    # divides; uneven lengths are exercised by the attention exchange tests and NCCL.
    generator = torch.Generator().manual_seed(91)
    if mixed_lengths:
        # Different modality lengths and a nonperiodic packing catch an accidental
        # per-modality split.
        tags = torch.tensor([0] * 9 + [1] * 28 + [2] * 19)
        tags = tags[torch.randperm(56, generator=generator)]
        return {
            "hidden_states": torch.randn(1, 9, 2, generator=generator),
            "audio_hidden_states": torch.randn(1, 19, 2, generator=generator),
            "encoder_hidden_states": torch.randn(1, 28, 4, generator=generator),
            "timestep": torch.tensor([0.0, 0.25, 0.75]),
            "timestep_indices": torch.arange(56) % 3,
            "token_tags": tags,
            "position_ids": torch.arange(168).reshape(56, 3) % 5,
            "video_indices": torch.where(tags == 0)[0],
            "text_indices": torch.where(tags == 1)[0],
            "audio_indices": torch.where(tags == 2)[0],
            "return_dict": False,
        }
    return {
        "hidden_states": torch.randn(1, 19, 2, generator=generator),
        "audio_hidden_states": torch.randn(1, 18, 2, generator=generator),
        "encoder_hidden_states": torch.randn(1, 19, 4, generator=generator),
        "timestep": torch.tensor([0.0, 0.25, 0.75]),
        "timestep_indices": torch.arange(56) % 3,
        "token_tags": torch.arange(56) % 3,
        "position_ids": torch.arange(168).reshape(56, 3) % 5,
        "video_indices": torch.arange(0, 56, 3),
        "text_indices": torch.arange(1, 56, 3),
        "audio_indices": torch.arange(2, 56, 3),
        "return_dict": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world", type=int)
    parser.add_argument("--rank-fd", type=int)
    parser.add_argument("--leader")
    parser.add_argument("--h3", action="store_true")
    parser.add_argument("--h3-lora", action="store_true")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--hosted", action="store_true")
    parser.add_argument("--spread", action="store_true")
    args = parser.parse_args()
    args.h3 = args.h3 or args.h3_lora or args.hosted or args.spread
    torch.set_num_threads(1)
    _capture_seal()
    channel = Channel(socket.socket(fileno=args.rank_fd))
    executor: Any = Executor(channel, args.root, rank=args.rank, world=args.world)
    executor.device_kind = "cpu"
    model: Any = (
        hosted_model()
        if args.hosted
        else spread_model()
        if args.spread
        else h3_model(adapters=args.h3_lora)
        if args.h3
        else collective_model(args.root / f"rank-{args.rank}" if args.staged else None)
    )
    executor.residency = model._cozy_residency
    executor._group_model_key = "fixture"
    roots = (
        {"dit": model.dit, "text_encoder": model.text_encoder, "vae": model.vae}
        if args.hosted or args.spread
        else {"dit": model.dit, "text_encoder": model.text_encoder}
        if args.h3
        else {"block": model.block, "overlay": model.overlay, "spare": model.spare}
    )
    if args.h3_lora:
        roots["turbo_overlay"] = model.turbo_overlay
    executor.backend = SimpleNamespace(
        components=roots, parked={}, component_bytes=HOSTED_SIZES if args.hosted else {}
    )
    executor.torch = torch
    executor.ready = True
    if not args.h3:
        executor.group_components = {("fixture", key): (model, root) for key, root in roots.items()}
        executor.group_sharded.add(("fixture", "block"))
    while (command := channel.recv()) is not None:
        name = command["cmd"]
        if name == "shutdown":
            return
        if name == "join":
            reply = executor.join(decode(command))
            if args.h3 and reply.get("ok"):
                refused = executor._install_group(torch, model, HOSTED_CAPACITY)
                assert refused is None, refused
        elif name == "run":
            reply = executor.run(command)
        else:
            raise AssertionError(name)
        channel.send({**reply, "reply": name})


if __name__ == "__main__":
    main()
