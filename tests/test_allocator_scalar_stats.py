"""Fresh native allocator facts under real CPU Torch; no CUDA device opens."""

from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.internal.memory import Planner


def test_missing_nested_api_uses_existing_scalar_readers_and_cpu_never_queries() -> None:
    calls: list[str] = []

    def read(name: str, value: int) -> Any:
        def scalar(device: object) -> int:
            assert device == "assigned"
            calls.append(name)
            return value

        return scalar

    cuda = SimpleNamespace(
        **{
            name: read(name, value)
            for name, value in (
                ("memory_allocated", 11),
                ("memory_reserved", 23),
                ("max_memory_allocated", 37),
                ("max_memory_reserved", 53),
            )
        },
        reset_peak_memory_stats=lambda _: None,
    )
    owner = Planner(SimpleNamespace(cuda=cuda), SimpleNamespace(type="cpu"))
    owner.device = "assigned"
    assert owner._allocated() == owner._reserved() == 0
    owner._sample()
    assert not calls
    owner.cuda = True
    assert (owner._allocated(), owner._reserved()) == (11, 23)
    owner._sample()
    assert (owner.peak_allocated, owner.peak) == (37, 53)
    assert calls == [
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ]


@pytest.mark.parametrize(
    "metric,field,public",
    [
        ("allocated_bytes", "current", "memory_allocated"),
        ("reserved_bytes", "current", "memory_reserved"),
        ("allocated_bytes", "peak", "max_memory_allocated"),
        ("reserved_bytes", "peak", "max_memory_reserved"),
    ],
)
@pytest.mark.parametrize("value", [0, -7, 2**40])
def test_real_torch_native_mapping_matches_public_scalar(
    monkeypatch: pytest.MonkeyPatch, metric: str, field: str, public: str, value: int
) -> None:
    torch = pytest.importorskip("torch")
    assert not torch.cuda.is_initialized()
    state: dict[str, Any] = {metric: {"all": {field: value}, "large_pool": {field: 99}}}
    calls: list[int] = []

    def native(device: int) -> dict[str, Any]:
        calls.append(device)
        return state

    owner = Planner(torch, torch.device("cpu"))
    owner.cuda, owner.device = True, torch.device("cuda:0")
    # Only the read-only native stats seam is replaced. Torch's real public scalar
    # and nested reader implementations execute, including explicit device selection.
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda.memory, "is_initialized", lambda: True)
        patch.setattr(torch._C, "_cuda_memoryStats", native)
        assert owner._allocator_scalar(metric, field, public) == getattr(torch.cuda, public)(
            owner.device
        )
        state.clear()
        assert (
            owner._allocator_scalar(metric, field, public)
            == getattr(torch.cuda, public)(owner.device)
            == 0
        )
    assert calls == [0, 0, 0, 0]
    assert not torch.cuda.is_initialized()


def test_fresh_nested_reads_preserve_driver_race_and_peak_sampling_order() -> None:
    events: list[str] = []
    state = {"reserved": 400, "peak": 450, "used": 550}

    def nested(device: object) -> dict[str, Any]:
        events.append("allocator")
        return {
            "reserved_bytes": {"all": {"current": state["reserved"], "peak": 600}},
            "allocated_bytes": {"all": {"current": 300, "peak": state["peak"]}},
        }

    def process() -> int:
        events.append("driver")
        return state["used"]

    def reset(device: object) -> None:
        events.append("reset")

    cuda = SimpleNamespace(memory_stats_as_nested_dict=nested, reset_peak_memory_stats=reset)
    owner = Planner(SimpleNamespace(cuda=cuda), SimpleNamespace(type="cpu"), process_bytes=process)
    owner.cuda, owner.overhead = True, 150
    owner._sample()
    assert events == ["allocator", "driver", "allocator", "allocator", "allocator", "reset"]
    assert (owner.peak_allocated, owner.peak, owner.overhead) == (450, 600, 150)
    events.clear()

    def racing() -> int:
        events.append("driver")
        state["reserved"] = 200
        return 550

    owner.process_bytes = racing
    assert not owner.observe_process()
    assert events == ["allocator", "driver", "allocator"]
    assert owner.overhead == 150  # never derives 350 from mismatched allocator facts
    owner.process_bytes = lambda: None
    assert not owner.observe_process()
    assert owner.driver_process_bytes == -1 and owner.overhead == 150
    state["reserved"] = 123
    assert owner._reserved() == 123  # no cached allocator snapshot


def test_nested_query_failure_is_not_silently_zero_or_legacy_fallback() -> None:
    def broken(device: object) -> dict[str, Any]:
        raise RuntimeError("allocator observation failed")

    cuda = SimpleNamespace(memory_stats_as_nested_dict=broken, memory_reserved=lambda _: 42)
    owner = Planner(SimpleNamespace(cuda=cuda), SimpleNamespace(type="cpu"))
    owner.cuda = True
    with pytest.raises(RuntimeError, match="allocator observation failed"):
        owner._reserved()


def test_real_torch_uninitialized_and_missing_nested_paths_match_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    assert not torch.cuda.is_initialized()
    owner = Planner(torch, torch.device("cpu"))
    owner.cuda, owner.device = True, torch.device("cuda:0")
    assert owner._reserved() == torch.cuda.memory_reserved(owner.device) == 0
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda.memory, "is_initialized", lambda: True)
        for state in ({"reserved_bytes": 17}, {"reserved_bytes": {"all": 17}}):
            patch.setattr(torch._C, "_cuda_memoryStats", lambda _, facts=state: facts)
            assert owner._reserved() == torch.cuda.memory_reserved(owner.device) == 0
    assert not torch.cuda.is_initialized()
