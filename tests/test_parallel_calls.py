"""Rank transport and mirrored execution, using real follower processes and gloo.

The tiny model is constructed in the test; checkpoint fill and CUDA are separate gates.
The command transport, follower Executor.run, model wrappers, and collectives are real.
"""

from __future__ import annotations

import functools
import inspect
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import (  # noqa: E402
    attention,
    attention_sol,
    attention_ulysses,
    execution_evidence,
    spawn,
)
from cozy_runtime.internal.executor import (  # noqa: E402
    Executor,
    residence_ceiling,
    settled_capacity,
)
from cozy_runtime.internal.fill import complete_device_envelope, price_envelope  # noqa: E402
from cozy_runtime.internal.parallel import cp, mirror, wire  # noqa: E402
from cozy_runtime.internal.parallel.group import RankGroup  # noqa: E402
from cozy_runtime.internal.parallel.mirror import mirror_component  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from cozy_runtime.internal.worker.lanes import LaneSet  # noqa: E402
from cozy_runtime.internal.worker.plan import (  # noqa: E402
    DeclaredBinding,
    PreparedModel,
    PreparedRequest,
)
from testdata.parallel_calls import (  # noqa: E402
    HOSTED_CAPACITY,
    collective_model,
    h3_batch,
    h3_model,
    h3_turbo_inputs,
    hosted_model,
    spread_model,
)


@pytest.mark.parametrize("shape", ["scalar", "offset", "transpose", "empty", "shared"])
def test_tensor_spool_writes_only_the_tensor_values(tmp_path: Path, shape: str) -> None:
    original = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    cases = {
        "scalar": original[2, 3],
        "offset": original[1:3],
        "transpose": original.T,
        "empty": original[:0],
        "shared": original[0],
    }
    value = cases[shape]
    spool = wire.TensorSpool(tmp_path)
    encoded = wire.marshal(value, spool)
    restored = wire.unmarshal(encoded, spool, device="cpu")
    assert isinstance(restored, torch.Tensor)
    torch.testing.assert_close(restored, value)
    assert (tmp_path / encoded["v"]).stat().st_size == value.numel() * value.element_size()
    restored.fill_(-1)
    torch.testing.assert_close(original, torch.arange(24, dtype=torch.bfloat16).reshape(4, 6))


def test_truncated_tensor_refuses_instead_of_returning_uninitialized_memory(tmp_path: Path) -> None:
    spool = wire.TensorSpool(tmp_path)
    encoded = wire.marshal(torch.ones(2), spool)
    (tmp_path / encoded["v"]).write_bytes(b"")
    with pytest.raises(wire.UncrossableArgument, match="expected 8"):
        wire.unmarshal(encoded, spool, device="cpu")


@pytest.mark.parametrize(
    ("kind", "degree"),
    [
        ("collective", 2),
        ("collective", 4),
        ("staged", 2),
        ("h3", 2),
        ("h3", 4),
        ("h3", 7),
        ("h3", 8),
        ("h3-lora", 2),
        ("h3-lora", 4),
    ],
)
def test_leader_calls_real_follower_executors_twice(tmp_path: Path, degree: int, kind: str) -> None:
    is_h3 = kind in ("h3", "h3-lora")
    if is_h3:
        pytest.importorskip("diffusers")
    if kind == "h3-lora":
        pytest.importorskip("peft")
    if kind == "staged" and not torch.cuda.is_available():
        pytest.skip("native component residency requires CUDA; both replicas use one physical GPU")
    fixture = Path(__file__).parent / "testdata" / "parallel_calls.py"

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=[
                str(fixture),
                *module_argv[2:],
                *(["--" + kind] if is_h3 or kind == "staged" else []),
            ],
            inherit_fd=fd,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )

    group = RankGroup(degree=degree, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    model: Any = (
        h3_model(adapters=kind == "h3-lora")
        if is_h3
        else collective_model(tmp_path / "rank-0" if kind == "staged" else None)
    )
    if is_h3:
        with torch.no_grad():
            batches = [h3_batch(), h3_batch(mixed_lengths=True)]
            expected = [
                (
                    batch,
                    model.dit(
                        **(
                            h3_turbo_inputs(batch, model.turbo_overlay)
                            if kind == "h3-lora"
                            else batch
                        )
                    ),
                )
                for batch in batches
            ]
    spool = tmp_path / "attempt-a"
    ours, theirs = socket.socketpair()
    leader: Any = Executor(Channel(ours), tmp_path, world=degree)
    leader.device_kind = "cpu"
    leader.group = group
    leader._group_model_key = "fixture"
    roots = (
        {"dit": model.dit, "text_encoder": model.text_encoder}
        if is_h3
        else {"block": model.block, "overlay": model.overlay, "spare": model.spare}
    )
    if kind == "h3-lora":
        roots["turbo_overlay"] = model.turbo_overlay
    leader.backend = SimpleNamespace(components=roots, parked={})
    try:
        group.spawn()
        group.form(torch, {})
        leader.pg = group.pg
        if is_h3:
            # The first processor is valid; an incompatible later site must still refuse.
            rejected = h3_model()
            rejected.dit.transformer_blocks[
                -1
            ].attn.processor._attention_backend = "_sage_qk_int8_pv_fp8_cuda_sm90"
            comms = cp.CpComms(group.pg, 0, torch.device("cpu"))
            with pytest.raises(cp.ContextParallelUnavailable, match="does not support"):
                cp.install_context_parallel(rejected, degree=degree, comms=comms)
            signature = inspect.signature(model.dit.forward)
            assert leader._install_group(torch, model) is None
            assert inspect.signature(model.dit.forward) == signature
            assert leader._attempt_spool is None
            assert leader._warm_model(model) >= 0
            assert leader._attempt_spool is None
            assert not list((tmp_path / "warm").rglob("tensor-*.raw"))
        else:
            mirror_component(
                model.block,
                name=("fixture", "block"),
                prepared={("fixture", key): (model, root) for key, root in roots.items()},
                group=group,
                spool=lambda: spool,
            )
        if kind == "staged":
            # Exercise the real group chooser, then carry its result through native
            # CUDA residency and the follower's actual mirrored scope. The tiny
            # capacity here is a conservative planner input, not a GPU limit claim.
            lanes = LaneSet.from_envelope(
                ",".join(str(rank) for rank in range(degree)), worker_pid=os.getpid()
            )
            row = lanes.group(tuple(range(degree))).row("fixture")
            backend = model._cozy_residency.backend
            row.ledger.observe_construction(
                {
                    "filled_bytes": 12,
                    "allocator_bytes": 4,
                    "reserved_bytes": 4,
                    "resident": {name: backend.vram_charge[name] for name in backend.components},
                    "parked": sorted(backend.parked),
                    "evicted": {
                        name: backend.component_memory[name].settled for name in backend.parked
                    },
                    "declared_scopes": {
                        "sample": ["block", "overlay"],
                        "fail_before_collective": ["block", "overlay"],
                    },
                    "device_free_bytes": 1,
                    "device_total_bytes": 1024,
                }
            )
            row.ledger.activations_by_cell["-"] = 37
            row.ledger.activation_scopes_by_cell["-"] = {"sample": 11, "fail_before_collective": 13}
            binding = DeclaredBinding(
                entrypoint_binding_digest="sha256:" + "0" * 64,
                entrypoint="sample",
                model_class="ModelCalls",
                model_binding_path="fixture",
                model_parameter_name="model",
                release="fixture/1",
                logical_weight_bytes=12,
            )
            plan = row.chooser.choose(
                binding,
                PreparedModel(delivery_rung="verbatim"),
                PreparedRequest.unresolved("sample", {}),
                fits=False,
            )
            assert plan.placement == "component_staged"
        for attempt in ("attempt-a", "attempt-b"):
            spool = tmp_path / attempt
            leader._attempt_spool = spool
            if kind == "staged":
                model._cozy_residency.open_attempt(
                    plan.placement, plan.headroom_bytes, dict(plan.scope_headroom_bytes)
                )
            served: dict[str, set[str]] = {}
            if is_h3:
                for batch, full_output in expected:
                    # Observe the inputs of real SDPA calls, not just approximately equal
                    # model outputs: replicated text must not become a degree-times-long
                    # attention document before the packed sequence is sharded.
                    with (
                        torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
                        ) as profile,
                        attention_ulysses.observing() as seen,
                    ):
                        model.compute(batch, full_output)
                    for backend, impls in seen.items():
                        served.setdefault(backend, set()).update(impls)
                    attention_shapes = [
                        event.input_shapes[0]
                        for event in profile.events()
                        if event.name == "aten::scaled_dot_product_attention"
                    ]
                    config = model.dit.config
                    local_text = [
                        1,
                        config.num_attention_heads,
                        batch["encoder_hidden_states"].shape[1],
                        config.attention_head_dim,
                    ]
                    distributed_packed = [
                        1,
                        config.num_attention_heads // degree,
                        batch["position_ids"].shape[0],
                        config.attention_head_dim,
                    ]
                    assert attention_shapes == (
                        [local_text] * config.num_refiner_layers
                        + [distributed_packed] * config.num_layers
                    )
            else:
                state = SimpleNamespace(
                    value=torch.tensor([2.0, 4.0, 8.0]),
                    expected_stages=2 if kind == "staged" and attempt == "attempt-a" else 0,
                )
                steps: list[int] = []
                rank_counts: dict[int, dict[str, int]] = {}
                with attention_sol.observing(ranks=rank_counts):
                    model.sample(state, on_step=steps.append, cancel=lambda: False)
                assert set(rank_counts) == set(range(1, degree))
                assert all(not any(counts.values()) for counts in rank_counts.values())
                assert attention_sol._RANK_COUNTS.get() is None
                assert steps == [0, 1]
                torch.testing.assert_close(
                    state.result, (state.value + 1) * degree + degree * (degree - 1) // 2
                )
            assert not cp.in_gated_call()
            assert not list((spool / "ranks").iterdir())
            # Every follower reports its own process, interval and attention for the attempt.
            records = group.rank_records
            assert sorted(records) == list(range(1, degree))
            assert tuple(records[rank]["pid"] for rank in sorted(records)) == group.pids()
            assert all(0 < row["start_us"] <= row["end_us"] for row in records.values())
            leader_view = execution_evidence.observed(attention.observed(roots).hosts)
            assert {row["attention"]["observed"] for row in records.values()} == {leader_view}
            # What served on each follower is recorded, as the leader observed its own: the
            # SDPA backend torch dispatched inside the Ulysses wrapper (none without attention).
            impls = {row["attention"]["impl"] for row in records.values()}
            assert impls == ({attention.implementations(served)} if is_h3 else {""})
            assert not is_h3 or impls <= {"flash", "efficient", "math"}
            execution = leader._execution("ref2va_dit=sdpa", None, 1, 2)
            assert execution["degree"] == degree
            assert [row["rank"] for row in execution["ranks"]] == list(range(degree))
            assert {row["attention"]["requested"] for row in execution["ranks"]} == {
                "ref2va_dit=sdpa"
            }
            records.clear()
        if kind in ("collective", "staged"):
            # The follower returns a failure while rank 0 is already in all_reduce.
            # Reading the failure only AFTER rank 0's body returns would wait forever.
            with pytest.raises((RuntimeError, ValueError)) as failed:
                model.fail_before_collective()
            if isinstance(failed.value, ValueError):
                # A follower can fail before rank 0 enters all_reduce. The error
                # monitor then destroys the group first, so Torch refuses entry
                # rather than interrupting a collective already in progress.
                assert "Default process group has not been initialized" in str(failed.value)
            assert group.broken
            assert not cp.in_gated_call()
    finally:
        group.close()
        if model._cozy_residency is not None:
            model._cozy_residency.backend.close()
        ours.close()
        theirs.close()
    assert not torch.distributed.is_initialized()
    if kind == "h3-lora":
        assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("degree", [2, 4])
def test_a_placeable_component_runs_on_its_follower(tmp_path: Path, degree: int) -> None:
    """Rank 0 holds no bytes of a hosted component; its module calls run on the home rank
    and the result, a transformers model output, comes back identical."""
    pytest.importorskip("diffusers")
    fixture = Path(__file__).parent / "testdata" / "parallel_calls.py"

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=[str(fixture), *module_argv[2:], "--hosted"],
            inherit_fd=fd,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )

    group = RankGroup(degree=degree, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    model: Any = hosted_model()
    x = torch.randn(3, 4)
    with torch.no_grad():
        expected = model.text_encoder.proj(x)
    ours, theirs = socket.socketpair()
    leader: Any = Executor(Channel(ours), tmp_path, world=degree)
    leader.device_kind = "cpu"
    leader.group = group
    leader._group_model_key = "fixture"
    roots = {"dit": model.dit, "text_encoder": model.text_encoder, "vae": model.vae}
    leader.backend = SimpleNamespace(
        components=roots, parked={}, component_bytes={"dit": 100, "text_encoder": 60, "vae": 30}
    )
    try:
        group.spawn()
        group.form(torch, {})
        leader.pg = group.pg
        assert leader._install_group(torch, model, HOSTED_CAPACITY) is None
        assert leader.facts["sequence_parallel"]["hosted"] == {"text_encoder": 1}
        for attempt in ("attempt-a", "attempt-b"):
            leader._attempt_spool = tmp_path / attempt
            output = model.encode(x)
            assert type(output).__name__ == "BaseModelOutputWithPast"
            assert torch.equal(output.last_hidden_state, expected)
            assert output.hidden_states[0].tolist() == [group.pids()[0]]
            assert not list((tmp_path / attempt).rglob("tensor-*.raw"))
        # A call outside any component scope is refused on rank 0, before a follower sees it.
        with pytest.raises(cp.UngatedShardedForward):
            model.text_encoder(x)
        # The sharded DiT still mirrors to every rank beside the hosted component.
        batch = h3_batch()
        with torch.no_grad():
            model.compute(batch, h3_model().dit(**batch))
    finally:
        group.close()
        ours.close()
        theirs.close()
    assert not torch.distributed.is_initialized()


@pytest.mark.parametrize("degree", [2, 4])
def test_h3_vae_clips_spread_over_the_group_equal_one_rank(tmp_path: Path, degree: int) -> None:
    """Each temporal clip decodes on some rank from its own latent window; rank 0 cross-fades
    them in order. Encode clips spread the same way. Both must equal a single-rank run."""
    pytest.importorskip("diffusers")
    fixture = Path(__file__).parent / "testdata" / "parallel_calls.py"

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=[str(fixture), *module_argv[2:], "--spread"],
            inherit_fd=fd,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )

    group = RankGroup(degree=degree, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    model: Any = spread_model()
    # 33 latent frames are 7 clips: rounds of `degree`, the last one partial.
    z = torch.randn(1, 4, 33, 2, 3, generator=torch.Generator().manual_seed(5))
    expected = model.decode(z)
    # 56 frames pad to four 17-frame encode clips, as a 56-frame continuation window does.
    x = torch.rand(1, 3, 56, *expected.shape[-2:], generator=torch.Generator().manual_seed(6))
    moments = model.encode(x)
    ours, theirs = socket.socketpair()
    leader: Any = Executor(Channel(ours), tmp_path, world=degree)
    leader.device_kind = "cpu"
    leader.group = group
    leader._group_model_key = "fixture"
    roots = {"dit": model.dit, "text_encoder": model.text_encoder, "vae": model.vae}
    leader.backend = SimpleNamespace(components=roots, parked={})
    try:
        group.spawn()
        group.form(torch, {})
        leader.pg = group.pg
        assert leader._install_group(torch, model, HOSTED_CAPACITY) is None
        assert leader.facts["sequence_parallel"]["spread_ranks"] == list(range(degree))
        leader._attempt_spool = tmp_path / "attempt"
        assert torch.equal(model.encode(x), moments)
        assert sorted(group.rank_records) == list(range(1, degree))
        group.rank_records.clear()
        actual = model.decode(z)
        assert torch.equal(actual, expected)
        assert sorted(group.rank_records) == list(range(1, degree))
        assert not list((tmp_path / "attempt").rglob("tensor-*.raw"))
    finally:
        group.close()
        ours.close()
        theirs.close()
    assert not torch.distributed.is_initialized()


def test_hosting_plan_keeps_h3s_text_encoder_on_a_follower_only_when_rank_0_cannot() -> None:
    sizes = {
        "text_encoder": 51_506_192_496,
        "ref2va_dit": 21_105_997_260,
        "fl2va_dit": 21_105_997_260,
        "video_vae": 5_570_955_392,
        "audio_vae": 605_306_340,
    }
    scopes = {
        "condition_text": ("text_encoder",),
        "condition_ref2va_media": ("video_vae", "audio_vae"),
        "sample_ref2va": ("ref2va_dit",),
        "sample_fl2va": ("fl2va_dit",),
    }
    plan = functools.partial(
        mirror.hosting_plan,
        placeable=("text_encoder",),
        sizes=sizes,
        sharded={"ref2va_dit", "fl2va_dit"},
        scopes=scopes,
    )
    assert plan(capacity=85_017_493_504, world=4) == {"text_encoder": 1}
    assert plan(capacity=85_017_493_504, world=1) == {}
    # An H200 holds everything on rank 0; nothing moves.
    assert plan(capacity=140_000_000_000, world=2) == {}
    # A card too small for the conditioner beside a DiT shard keeps staging it on rank 0.
    assert plan(capacity=70_000_000_000, world=4) == {}


def test_an_h100_hosts_h3s_text_encoder_once_the_fp8_dit_fill_has_settled() -> None:
    """Runs 1435-1441 (4xH100, fp8-pruned): `residency.hosted` stayed empty and rank 0
    staged the text encoder on every request. Each fp8 DiT fills as 41.2 GB of float
    destinations beside its 20.1 GB encoded payload, then settles to 21.1 GB. That transient
    comes off the fill ceiling, so the ceiling cannot hold the 51.5 GB encoder beside a DiT.
    The settled card can."""
    encoded, payload = 40_106_235_904, 20_053_117_952
    rows = [
        (dit, logical, stored, route, stored if route == "encoded_gemm" else 0, 0)
        for dit in ("ref2va_dit", "fl2va_dit")
        for logical, stored, route in (
            (encoded, payload, "encoded_gemm"),
            (1_052_879_308, 1_052_879_308, "verbatim"),
        )
    ]
    sizes = {"ref2va_dit": 21_105_997_260, "fl2va_dit": 21_105_997_260}
    for name, size in (
        ("text_encoder", 51_506_192_496),
        ("video_vae", 5_570_955_392),
        ("audio_vae", 605_306_340),
    ):
        rows.append((name, size, size, "verbatim", 0, 0))
        sizes[name] = size
    envelope = complete_device_envelope(price_envelope(rows, 0), 0)
    overhead = envelope["fill_overhead_bytes"]
    assert overhead == payload
    card = 84_000_000_000
    fill_ceiling = residence_ceiling(
        destinations=envelope["base_bytes"], authorized=card - overhead, allocatable=card - overhead
    )
    plan = functools.partial(
        mirror.hosting_plan,
        placeable=("text_encoder",),
        sizes=sizes,
        sharded={"ref2va_dit", "fl2va_dit"},
        scopes={"condition_text": ("text_encoder",), "sample_ref2va": ("ref2va_dit",)},
        world=4,
    )
    assert plan(capacity=fill_ceiling) == {}
    capacity = settled_capacity(resident_budget=fill_ceiling, fill_overhead=overhead)
    assert capacity == card
    assert plan(capacity=capacity) == {"text_encoder": 1}
    # A card that holds the whole construction keeps it on rank 0.
    roomy = envelope["base_bytes"]
    assert plan(capacity=settled_capacity(resident_budget=roomy, fill_overhead=overhead)) == {}


def test_a_model_output_crosses_the_rank_seam(tmp_path: Path) -> None:
    outputs = pytest.importorskip("transformers.modeling_outputs")
    value = outputs.BaseModelOutputWithPast(
        last_hidden_state=torch.randn(1, 3, 4), hidden_states=(torch.ones(2),)
    )
    spool = wire.TensorSpool(tmp_path)
    restored = wire.unmarshal(wire.marshal(value, spool), spool, device="cpu")
    assert isinstance(restored, type(value)) and type(restored) is type(value)
    assert torch.equal(restored.last_hidden_state, value.last_hidden_state)
    assert restored.past_key_values is None
    assert torch.equal(restored.hidden_states[0], value.hidden_states[0])


def test_actual_accelerate_wrappers_refuse_only_on_sharded_components() -> None:
    pytest.importorskip("diffusers")
    hooks = pytest.importorskip("accelerate.hooks")

    model = h3_model()
    hooks.add_hook_to_module(model.text_encoder, hooks.AlignDevicesHook(execution_device="cpu"))
    cp._refuse_if_forward_wrapped(cp.sharding_candidates(model))
    hooks.add_hook_to_module(
        model.dit.transformer_blocks[1], hooks.AlignDevicesHook(execution_device="cpu")
    )
    with pytest.raises(cp.ContextParallelUnavailable, match=r"transformer_blocks\.1"):
        cp.install_context_parallel(model, degree=2, comms=None)
