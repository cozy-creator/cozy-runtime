"""CPU request lifetime and replay; CUDA arithmetic is a separate ordinary-CLI gate."""

from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention  # noqa:E402

from cozy_runtime.author import Model, uses_components  # noqa:E402
from cozy_runtime.author._attention_scope import AttentionLayout  # noqa:E402
from cozy_runtime.internal import attention  # noqa:E402
from cozy_runtime.internal import attention_producer as producer  # noqa: E402
from cozy_runtime.internal.encoding import DeviceFacts  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa:E402
from cozy_runtime.internal.executor_commands import Attention, Unload  # noqa:E402
from cozy_runtime.internal.parallel import wire  # noqa:E402
from cozy_runtime.internal.seam import Channel  # noqa:E402

NAME = "kitchen-sol-producer-fp4-fixed-means-shared-qkv"


class StateCalls(torch.nn.Module):  # type: ignore[name-defined, misc]
    """Real follower calls exercise state only; no pretend CPU Kitchen kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = MiniMaxH3Attention(hidden_size=256, heads=2, dim_head=128)
        self.attn.processor._attention_backend = attention._member(attention.BY_NAME[NAME])
        self.inputs: list[tuple[Any, Any]] = []

    def forward(self, step: int, fail: bool = False) -> Any:
        state = producer._STATE.get()
        assert state is not None
        inputs = producer.previous_statistics(
            state, ("dit", "site", id(self)), step, (129, 1, 17, 2, 1), True, replay_safe=True
        )
        self.inputs.append(inputs)
        state.statistics[("dit", "site", id(self))] = (
            step,
            (129, 1, 17, 2, 1),
            torch.tensor([step]),
            torch.tensor([step + 1]),
        )
        if fail:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 64.00 MiB.")
        return torch.tensor([step])


class StateModel(Model[object]):
    dit: Any

    @uses_components("dit")
    def step(self, step: int) -> Any:
        return self.dit(step)


def test_real_follower_commands_share_attempt_state_and_restore_clears(tmp_path: Path) -> None:
    ours, theirs = socket.socketpair()
    follower: Any = Executor(Channel(ours), tmp_path, rank=1, world=2)
    target: Any = Executor(
        follower.channel, tmp_path, rank=1, world=2, host=follower, construction="proof"
    )
    root = StateCalls()
    model = StateModel.for_test(dit=root)
    target.backend = SimpleNamespace(roots={"dit": root}, close=lambda: None)
    target.attention_defaults = attention.snapshot(target.backend.roots)
    target.device_facts = DeviceFacts("cpu", "fixture", 0, "", "", 0)
    follower.constructions["proof"] = target
    follower.group_components[("proof#m", "dit")] = (model, root)
    follower.group_sharded.add(("proof#m", "dit"))
    follower.torch, follower.device_kind, follower.ready = torch, "cpu", True
    # Empty CLI pin still holds the authored/prepared CP producer, selected by actual sites.
    reply = follower.select_attention(Attention(construction="proof", pin=""))
    assert reply["ok"] and NAME in reply["hosts"]["dit"]
    assert follower._producer_hold is not None
    state = follower._producer_hold[1]

    def call(step: int, fail: bool = False) -> dict[str, Any]:
        command = wire.RunCommand(
            component=("proof#m", "dit"),
            spool=str(tmp_path / "attempt"),
            payloads="ranks",
            scopes=(wire.Scope("proof#m", "step", ("dit",)),),
            grad_enabled=False,
            inference_mode=True,
            autocast={},
            attention_layout=AttentionLayout(129, 17, step, 4),
            args=[step],
            kwargs={"fail": fail},
        )
        return follower.run(msgspec.json.decode(msgspec.json.encode(command)))

    try:
        first = call(4)
        assert first["ok"], first
        prior = next(iter(state.statistics.values()))[2:]
        assert call(5, fail=True)["code"] == "device_out_of_memory"
        assert call(5)["ok"]
        assert root.inputs[0] == (None, None)
        assert root.inputs[1][0] is prior[0] and root.inputs[2][0] is prior[0]
        assert producer._STATE.get() is None
        # Unrelated construction unload cannot erase the active hold.
        assert follower.unload(Unload(construction="other"))["ok"]
        assert follower._producer_hold is not None and state.statistics
        assert follower.select_attention(Attention(construction="proof"))["ok"]
        assert not state.statistics and not state.replay_inputs and not state.parallel_rotary
        assert follower._producer_hold is None
        assert follower.select_attention(Attention(construction="proof", pin=""))["ok"]
        assert follower._producer_hold is not None
        fresh = follower._producer_hold[1]
        assert fresh is not state and not fresh.statistics
        assert call(6)["ok"] and root.inputs[-1] == (None, None)
        refused = follower.select_attention(Attention(construction="proof", pin="unknown-kernel"))
        assert not refused["ok"] and not fresh.statistics and follower._producer_hold is None
        assert follower.select_attention(Attention(construction="proof", pin=""))["ok"]
        assert follower._producer_hold is not None
        fresh = follower._producer_hold[1]
        assert call(6)["ok"] and root.inputs[-1] == (None, None)
        assert follower.unload(Unload(construction="proof"))["ok"]
        assert follower._producer_hold is None and not fresh.statistics
    finally:
        follower._attention_hold.close()
        ours.close()
        theirs.close()


def test_borrowed_scope_and_component_replay_inputs_are_independent() -> None:
    state = producer.RequestState()
    geometry = (129, 28, 257, 2, 0)
    a, b = ("dit", "block", object()), ("dit", "block", object())
    first, second = object(), object()
    state.statistics[a] = (4, geometry, first, second)
    with producer.request_scope(state):
        assert producer.previous_statistics(state, a, 5, geometry, True, replay_safe=True) == (
            first,
            second,
        )
        state.statistics[a] = (5, geometry, object(), object())
    with producer.request_scope(state):
        assert producer.previous_statistics(state, a, 5, geometry, True, replay_safe=True) == (
            first,
            second,
        )
        assert producer.previous_statistics(state, b, 5, geometry, True, replay_safe=True) == (
            None,
            None,
        )
        assert producer.previous_statistics(state, a, 7, geometry, True, replay_safe=True) == (
            None,
            None,
        )
    state.clear()
    assert not state.statistics and not state.replay_inputs


def test_optional_h3_absence_does_not_break_other_diffusers_cp_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "diffusers.models.transformers.transformer_minimax_h3", None)
    producer.install_parallel_rotary(torch.nn.Linear(2, 2))


@pytest.mark.parametrize("cancel", [False, True])
def test_parent_attempt_finally_restores_real_follower_hold(tmp_path: Path, cancel: bool) -> None:
    """Real command transport and follower process; no GPU or kernel stand-in."""
    import os
    import sys

    from cozy_runtime.author._errors import Cancelled
    from cozy_runtime.internal import spawn
    from cozy_runtime.internal.parallel.group import RankGroup

    fixture = Path(__file__).parent / "testdata/cp_producer_hold.py"

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            # spawn_follower correctly uses -I, so PYTHONPATH cannot select the
            # checkout under test. Select it explicitly before importing the fixture.
            module_argv=[
                "-c",
                "import runpy,sys; sys.path.insert(0, "
                + repr(str(Path(__file__).parents[1] / "src"))
                + "); runpy.run_path("
                + repr(str(fixture))
                + ", run_name='__main__')",
                *module_argv[2:],
            ],
            inherit_fd=fd,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )

    group = RankGroup(degree=2, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    ours, theirs = socket.socketpair()
    leader = Executor(Channel(ours), tmp_path, world=2, construction="proof")
    leader.group = group
    error = Cancelled("caller canceled") if cancel else RuntimeError("body failed")
    try:
        group.spawn()
        with (
            pytest.raises(type(error)),
            leader._ranks_attention("", attention.Applied({"dit": {NAME: 1}}, ())),
        ):
            assert group.call_one(1, {"cmd": "state-seed"})["ok"]
            raise error
        result = group.call_one(1, {"cmd": "state-read"})
        assert result["ok"] and not result["live"] and result["cleared"]
    finally:
        group.close()
        ours.close()
        theirs.close()
