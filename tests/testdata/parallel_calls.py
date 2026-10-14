"""Small real models for rank-call integration; checkpoint preparation is a separate gate."""

from __future__ import annotations

import argparse
import faulthandler
import io
import os
import socket
import struct
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import tensorfs
import torch
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from cozy_runtime.author import AdapterRef, Model, placeable, spread, uses_components
from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout, attention_scope
from cozy_runtime.author._loader import census
from cozy_runtime.internal import attention, lora_composition, lora_contract, plane
from cozy_runtime.internal.encoding import SPEC_PLAIN, DeviceFacts
from cozy_runtime.internal.executor import Executor, _capture_seal, answer
from cozy_runtime.internal.executor_commands import decode
from cozy_runtime.internal.fill import Checkpoint
from cozy_runtime.internal.parallel.cp import in_gated_call
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.weight_policy import Layout
from cozy_runtime.internal.weights import PlaneBackend, WeightResidency, Weights
from cozy_runtime.internal.weights_sink import weights_transaction_id


class Collective(torch.nn.Module):  # type: ignore[misc]  # torch is optional in the base checking env
    #: this process's calls that asked to run out of memory, after and before the collective
    ooms = 0
    ooms_before = 0

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((1,), 2.0))
        #: calls that counted themselves in place (`counted`): what a rerun must not double
        self.register_buffer("seen", torch.zeros(1), persistent=False)

    def forward(
        self,
        value: Any,
        *,
        check: bool = False,
        fail: bool = False,
        fail_leader: bool = False,
        step: int | None = None,
        oom: int = 0,
        counted: bool = False,
        oom_before: int = 0,
        stall: float = 0.0,
        leader_skips: bool = False,
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
        if fail_leader and torch.distributed.get_rank() == 0:
            raise RuntimeError("GPU 0 ran out of memory")
        rank = torch.distributed.get_rank()
        nccl = torch.distributed.get_backend() == "nccl"
        result = value.detach().clone() if nccl else value.detach().cpu().clone()
        result = result + rank
        if counted:
            self.seen.add_(1)  # in place, in a buffer the call did not make
            result = result + self.seen.to(result.device)
        if oom_before and rank > 0:
            # A follower's first `oom_before` calls run out of memory before the collective:
            # GPU 0 waits in it until the group's collectives are aborted.
            Collective.ooms_before += 1
            if Collective.ooms_before <= oom_before:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 64.00 MiB.")
        if stall and rank > 0:
            time.sleep(stall)  # GPU 0 is inside the collective meanwhile
        if leader_skips and rank == 0:
            return result  # GPU 0 never joins: GPU 1 waits in the collective for good
        torch.distributed.all_reduce(result)
        if leader_skips:
            result = result.cpu()  # GPU 1's host waits on it too
        if oom and torch.distributed.get_rank() > 0:
            # A follower's first `oom` calls run out of memory after its collectives.
            Collective.ooms += 1
            if Collective.ooms <= oom:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 64.00 MiB.")
        return result


class ModelCalls(Model[object]):
    block: Any
    overlay: Any
    spare: Any

    @uses_components("block", "overlay")
    def sample(self, state: Any, *, on_step: Any, cancel: Any) -> Any:
        # Rich mutable state and callbacks are deliberately not rank-wire values.
        with torch.no_grad():
            self.block(state.value, check=True)
            for step in range(2):
                assert not cancel()
                with attention_scope(AttentionLayout(29, 9, step, 1, ("blocks.0",))):
                    result = self.block(state.value, step=step)
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

    @uses_components("block", "overlay")
    def out_of_memory(self, times: int) -> Any:
        with torch.no_grad():
            return self.block(torch.ones(1), oom=times)

    @uses_components("block", "overlay")
    def counted(self, after: int = 0, before: int = 0) -> Any:
        with torch.no_grad():
            value = torch.ones(1, device=self.block.weight.device)
            return self.block(value, oom=after, oom_before=before, counted=True)

    @uses_components("block", "overlay")
    def stalled(self, seconds: float) -> Any:
        with torch.no_grad():
            value = torch.ones(1, device=self.block.weight.device)
            return self.block(value, stall=seconds)

    @uses_components("block", "overlay")
    def stuck(self) -> Any:
        with torch.no_grad():
            value = torch.ones(1, device=self.block.weight.device)
            return self.block(value, leader_skips=True)

    @uses_components("block", "overlay")
    def fail_on_leader(self) -> None:
        with torch.no_grad():
            self.block(torch.ones(1), fail_leader=True)


def collective_model() -> ModelCalls:
    overlay, spare = torch.nn.Module(), torch.nn.Module()
    overlay.weight = torch.nn.Parameter(torch.ones(1))
    spare.weight = torch.nn.Parameter(torch.full((1,), 5.0))
    model = ModelCalls.for_test(block=Collective(), overlay=overlay, spare=spare)

    def check_scope(module: Any, args: Any, kwargs: Any) -> Any:
        assert model._cozy_active is not None
        assert model._cozy_active[1] == ("block", "overlay")
        residency: Any = model._cozy_residency
        if residency is not None:
            # This rank's own stage is admitted and its weights are bound (on the card, or in
            # the host tier without one).
            assert residency.active is not None and residency.active[1] == ("block", "overlay")
            assert (
                model.block.weight.is_cuda
                == model.overlay.weight.is_cuda
                == torch.cuda.is_available()
            )
            assert float(model.block.weight) == 2.0 and float(model.overlay.weight) == 1.0
        return (args[0].to(model.overlay.weight.device) + model.overlay.weight, *args[1:]), kwargs

    model.block.register_forward_pre_hook(check_scope, with_kwargs=True)
    return model


def plane_residency(root: Path, model: ModelCalls) -> PlaneBackend:
    """Register the model's three components, from a real TensorFS store, with this process's
    weight plane: construction moves to meta, and each stage binds its weights on the card."""
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
    checkpoint = Checkpoint.from_store(store, "sha256:" + writer.commit()["manifest"]["sha256"])
    # Every rank's plane on the one card; without one, each rank computes in its host tier.
    on_card = torch.cuda.is_available()
    device = torch.device("cuda", 0) if on_card else torch.device("cpu")
    weights = Weights(torch, device, device.type)
    backend = PlaneBackend.for_script(
        dict.fromkeys(values, checkpoint),
        [row for name in values for row in checkpoint.rows(name)],
        weights=weights,
        construction="parallel-test",
        release="parallel-test/1",
        encoded_leaves="refuse",
    )
    holder = SimpleNamespace(components={name: getattr(model, name).to("meta") for name in values})
    walked = census(holder)
    backend.expect(
        {name: [d.key for d in walked.destinations if d.component == name] for name in values}
    )
    live = backend.materialize(holder, walked)
    for destination in walked.destinations:
        backend.fill(destination.key, destination.spec, live[destination.key])
    backend.commit()
    scopes = {"sample": ("block", "overlay"), "fail_before_collective": ("block", "overlay")}
    object.__setattr__(
        model, "_cozy_residency", WeightResidency(weights, backend.components, scopes)
    )
    return backend


def prepared(roots: dict[str, Any], sizes: Mapping[str, int] | None = None) -> SimpleNamespace:
    """A prepared construction as `Executor._install_group` reads it: its roots, and each
    weight set's layout (only totals reach the hosting plan)."""
    return SimpleNamespace(
        roots=roots,
        components={
            name: SimpleNamespace(layout=Layout(common=size))
            for name, size in (sizes or {}).items()
        },
    )


class H3Calls(Model[object]):
    dit: Any
    text_encoder: Any

    @uses_components("dit")
    def compute(self, batch: dict[str, Any], expected: tuple[Any, Any]) -> Any:
        with torch.no_grad():
            result = self.dit(**batch)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
        return result


class H3Infer(H3Calls):
    @uses_components("dit")
    def infer(self, batch: dict[str, Any]) -> Any:
        with torch.no_grad():
            return self.dit(**batch)


class H3LoRACalls(H3Calls):
    @uses_components("dit")
    def compute(
        self, batch: dict[str, object], expected: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            result = lora_forward(self, batch)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
        return result


def lora_forward(model: H3Calls, batch: dict[str, object]) -> tuple[Any, Any]:
    """The DiT under its request LoRA. Text/video share the video sigma; audio uses its paired
    sigma."""
    tags = cast("torch.Tensor", batch["token_tags"])
    inputs = {
        **batch,
        "timestep": torch.tensor([0.75, 0.25]),
        "timestep_indices": torch.where(tags == 2, 1, 0),
    }
    return cast("tuple[Any, Any]", model.dit(**inputs))


def h3_lora_model(dit: torch.nn.Module) -> H3LoRACalls:
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
    return H3LoRACalls.for_test(dit=dit, text_encoder=torch.nn.Linear(4, 4))


def h3_model(*, adapters: bool = False) -> H3Calls:
    from diffusers import MiniMaxH3Transformer3DModel

    torch.manual_seed(231)
    # Diffusers' MiniMax-H3 DiT with H3's 56 heads, so every declared degree divides them.
    dit = MiniMaxH3Transformer3DModel(
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
    # Runtime's own selection: its SDPA runs through the Ulysses wrapper on sharded sites.
    attention.install(attention.available(DeviceFacts("cpu", "CPU", 0, "", "", 0)), {"dit": dit})

    def no_grad(module: Any, args: Any) -> None:
        assert not torch.is_grad_enabled()

    dit.register_forward_pre_hook(no_grad)
    if adapters:
        return h3_lora_model(dit)
    return H3Infer.for_test(dit=dit, text_encoder=torch.nn.Linear(4, 4))


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


class Clips(torch.nn.Module):  # type: ignore[misc]
    """A video codec's temporal clips: each a pure call `author.spread` offers the group."""

    def __init__(self) -> None:
        super().__init__()
        self.up = torch.nn.Conv3d(4, 3, 3, padding=1)
        self.down = torch.nn.Conv3d(3, 4, 3, padding=1)

    def decode_clip(self, z: Any) -> Any:
        return self.up(z)

    def encode_clip(self, x: Any) -> Any:
        return self.down(x)

    def decode(self, z: Any) -> Any:
        clips = ((clip,) for clip in z.split(5, dim=2))
        return torch.cat(list(spread(self, "decode_clip", clips)), dim=2)

    def encode(self, x: Any) -> Any:
        clips = ((clip,) for clip in x.split(17, dim=2))
        return torch.cat(list(spread(self, "encode_clip", clips)), dim=2)


class SpreadCalls(H3Calls):
    vae: Any

    @uses_components("vae")
    def decode(self, z: Any) -> Any:
        with torch.no_grad():
            return self.vae.decode(z)

    @uses_components("vae")
    def encode(self, x: Any) -> Any:
        with torch.no_grad():
            return self.vae.encode(x)


def spread_model() -> SpreadCalls:
    base = h3_model()
    torch.manual_seed(3)
    return SpreadCalls.for_test(dit=base.dit, text_encoder=base.text_encoder, vae=Clips().eval())


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


def out_of_memory_at(block: torch.nn.Module, call: int) -> None:
    """`block` runs out of memory on its `call`-th forward, before it runs: the GPUs before it in
    the group are already in its first collective."""
    calls = [0]

    def hook(module: Any, args: Any) -> None:
        calls[0] += 1
        if calls[0] == call:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB.")

    block.register_forward_pre_hook(hook)


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
    parser.add_argument("--before-plane", action="store_true")
    #: an NCCL group: this rank's card, once it has joined
    parser.add_argument("--cuda", action="store_true")
    #: the tiny H3 DiT's second transformer block runs out of memory on this call (1-based)
    parser.add_argument("--oom-at", type=int, default=0)
    args = parser.parse_args()
    if args.cuda:  # a stuck NCCL test says where each process is
        faulthandler.dump_traceback_later(60, repeat=True)
    args.h3 = args.h3 or args.h3_lora or args.hosted or args.spread
    torch.set_num_threads(1)
    _capture_seal()
    channel = Channel(socket.socket(fileno=args.rank_fd))
    executor: Any = Executor(channel, args.root, rank=args.rank, world=args.world)
    executor.device_kind = "cuda" if args.cuda else "cpu"
    model: Any = (
        hosted_model()
        if args.hosted
        else spread_model()
        if args.spread
        else h3_model(adapters=args.h3_lora)
        if args.h3
        else collective_model()
    )
    if args.staged:
        plane_residency(args.root / f"rank-{args.rank}", model)
        executor.weights = model._cozy_residency.weights
    executor.residency = model._cozy_residency
    executor._group_model_key = "fixture"
    roots = (
        {"dit": model.dit, "text_encoder": model.text_encoder, "vae": model.vae}
        if args.hosted or args.spread
        else {"dit": model.dit, "text_encoder": model.text_encoder}
        if args.h3
        else {"block": model.block, "overlay": model.overlay, "spare": model.spare}
    )
    executor.backend = prepared(roots, HOSTED_SIZES if args.hosted else None)
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
            again = int(command.get("room", -1)) >= 0  # the group formed again: installed
            if args.cuda and reply.get("ok") and not again:
                for root in roots.values():
                    root.to(executor._rank_device(torch))
                if args.oom_at:
                    out_of_memory_at(model.dit.transformer_blocks[1], args.oom_at)
            if args.h3 and reply.get("ok") and not again:
                refused = executor._install_group(torch, model, HOSTED_CAPACITY)
                assert refused is None, refused
        elif name == "run":
            reply = executor.run(command)
        elif name == "attention":
            # Holding rank 0's pin (`hosts`) or restoring the prepared selection.
            reply = {"ok": True, "rank": args.rank, **({"hosts": {}} if "pin" in command else {})}
        elif name == "hello":
            reply = executor.hello()
            if args.before_plane:  # a Runtime from before the weight plane says no such thing
                reply["memory"] = [m for m in reply["memory"] if m != plane.CAPABILITY]
        elif name == "budget" and args.before_plane:
            reply = {"ok": False, "code": "follower_command_unsupported", "detail": "budget"}
        else:
            answered = answer(executor, command)
            assert answered is not None, name
            reply = answered
        channel.send({**reply, "reply": name})


if __name__ == "__main__":
    main()
