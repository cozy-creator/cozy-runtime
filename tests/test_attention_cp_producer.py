"""CPU request lifetime of a package producer's state on a follower; no CUDA kernel."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention  # noqa:E402

from cozy_runtime.author import AttentionProducer, Model, uses_components  # noqa:E402
from cozy_runtime.author._attention import _PRODUCERS  # noqa:E402
from cozy_runtime.author._attention_scope import AttentionLayout  # noqa:E402
from cozy_runtime.internal import attention  # noqa:E402
from cozy_runtime.internal.encoding import DeviceFacts  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa:E402
from cozy_runtime.internal.executor_commands import Attention, Unload  # noqa:E402
from cozy_runtime.internal.parallel import wire  # noqa:E402
from cozy_runtime.internal.seam import Channel  # noqa:E402

NAME = "test-cp-producer"


class State:
    """One request's producer state: the steps it saw, and whether it was closed."""

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.closed = False

    def close(self) -> None:
        self.closed = True


def closed(state: State) -> bool:
    return state.closed


def _reject(call: Any, *args: Any, **kwargs: Any) -> Any:
    raise AssertionError("no attention call reaches a producer here")


PRODUCER = AttentionProducer(
    NAME,
    "kitchen-sol-fp4-fixed-means",
    _reject,
    lambda module: None,
    state=State,
    context_parallel=True,
)


@pytest.fixture(autouse=True)
def _registered(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setitem(_PRODUCERS, NAME, PRODUCER)
    yield


class StateCalls(torch.nn.Module):  # type: ignore[name-defined, misc]
    """Real follower calls exercise state only; no pretend CPU Kitchen kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = MiniMaxH3Attention(hidden_size=256, heads=2, dim_head=128)
        self.attn.processor._attention_backend = attention._member(attention.known()[NAME])
        self.states: list[State] = []

    def forward(self, step: int, fail: bool = False) -> Any:
        state = PRODUCER.request_state()
        assert isinstance(state, State)
        state.steps.append(step)
        self.states.append(state)
        if fail:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 64.00 MiB.")
        return torch.tensor([step])


class StateModel(Model[object]):
    dit: Any

    @uses_components("dit")
    def step(self, step: int) -> Any:
        return self.dit(step)


def test_real_follower_commands_share_attempt_state_and_restore_closes(tmp_path: Path) -> None:
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
    reply = follower.select_attention(Attention(construction="proof", pin=""))
    assert reply["ok"] and NAME in reply["hosts"]["dit"]
    assert follower._producer_hold is not None

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
        return cast(dict[str, Any], follower.run(msgspec.json.decode(msgspec.json.encode(command))))

    try:
        assert call(4)["ok"]
        assert call(5, fail=True)["code"] == "device_out_of_memory"
        assert call(5)["ok"]
        state = root.states[0]
        assert all(s is state for s in root.states) and state.steps == [4, 5, 5]
        assert PRODUCER.request_state() is None and not closed(state)
        # Unrelated construction unload cannot close the active hold.
        assert follower.unload(Unload(construction="other"))["ok"]
        assert follower._producer_hold is not None and not closed(state)
        assert follower.select_attention(Attention(construction="proof"))["ok"]
        assert closed(state) and follower._producer_hold is None
        assert follower.select_attention(Attention(construction="proof", pin=""))["ok"]
        assert call(6)["ok"] and root.states[-1] is not state and root.states[-1].steps == [6]
        fresh = root.states[-1]
        refused = follower.select_attention(Attention(construction="proof", pin="unknown-kernel"))
        assert not refused["ok"] and closed(fresh) and follower._producer_hold is None
        assert follower.select_attention(Attention(construction="proof", pin=""))["ok"]
        assert call(7)["ok"]
        last = root.states[-1]
        assert follower.unload(Unload(construction="proof"))["ok"]
        assert follower._producer_hold is None and closed(last)
    finally:
        follower._attention_hold.close()
        ours.close()
        theirs.close()


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
        assert result["ok"] and not result["live"] and result["closed"]
    finally:
        group.close()
        ours.close()
        theirs.close()
