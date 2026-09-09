"""Rank transport and mirrored execution, using real follower processes and gloo.

The tiny model is constructed in the test; checkpoint fill and CUDA are separate gates.
The command transport, follower Executor.run, model wrappers, and collectives are real.
"""

from __future__ import annotations

import inspect
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import spawn  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.parallel import cp, wire  # noqa: E402
from cozy_runtime.internal.parallel.group import RankGroup  # noqa: E402
from cozy_runtime.internal.parallel.mirror import mirror_component  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from testdata.parallel_calls import collective_model, h3_batch, h3_model  # noqa: E402


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
    [("collective", 2), ("collective", 4), ("staged", 2), ("h3", 2), ("h3", 4)],
)
def test_leader_calls_real_follower_executors_twice(tmp_path: Path, degree: int, kind: str) -> None:
    if kind == "h3":
        pytest.importorskip("diffusers")
    if kind == "staged" and not torch.cuda.is_available():
        pytest.skip("native component residency requires CUDA; both replicas use one physical GPU")
    fixture = Path(__file__).parent / "testdata" / "parallel_calls.py"

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=[
                str(fixture),
                *module_argv[2:],
                *(["--" + kind] if kind in ("h3", "staged") else []),
            ],
            inherit_fd=fd,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )

    group = RankGroup(degree=degree, backend="gloo", root=str(tmp_path), launch=launch)
    model: Any = (
        h3_model()
        if kind == "h3"
        else collective_model(tmp_path / "rank-0" if kind == "staged" else None)
    )
    batch = h3_batch() if kind == "h3" else {}
    if kind == "h3":
        with torch.no_grad():
            expected = model.dit(**batch)
    spool = tmp_path / "attempt-a"
    ours, theirs = socket.socketpair()
    leader: Any = Executor(Channel(ours), tmp_path, world=degree)
    leader.device_kind = "cpu"
    leader.group = group
    try:
        group.spawn()
        group.form(torch, {})
        leader.pg = group.pg
        if kind == "h3":
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
            assert not list((tmp_path / "warm" / "ranks").iterdir())
        else:
            mirror_component(
                model.block, model=model, name="block", group=group, spool=lambda: spool
            )
        for attempt in ("attempt-a", "attempt-b"):
            spool = tmp_path / attempt
            leader._attempt_spool = spool
            if kind == "staged":
                model._cozy_residency.open_attempt(
                    "component_staged", 37, {"sample": 11, "fail_before_collective": 13}
                )
            if kind == "h3":
                model.compute(batch, expected)
            else:
                state = SimpleNamespace(
                    value=torch.tensor([2.0, 4.0, 8.0]),
                    expected_stages=2 if kind == "staged" and attempt == "attempt-a" else 0,
                )
                steps: list[int] = []
                model.sample(state, on_step=steps.append, cancel=lambda: False)
                assert steps == [0, 1]
                torch.testing.assert_close(
                    state.result, (state.value + 1) * degree + degree * (degree - 1) // 2
                )
            assert not cp.in_gated_call()
            assert not list((spool / "ranks").iterdir())
        if kind in ("collective", "staged"):
            # The follower returns a failure while rank 0 is already in all_reduce.
            # Reading the failure only AFTER rank 0's body returns would wait forever.
            with pytest.raises(RuntimeError):
                model.fail_before_collective()
            assert group.broken
            assert not cp.in_gated_call()
    finally:
        group.close()
        if model._cozy_residency is not None:
            model._cozy_residency.backend.close()
        ours.close()
        theirs.close()
    assert not torch.distributed.is_initialized()


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
