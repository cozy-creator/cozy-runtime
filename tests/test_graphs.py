"""Development numerical and lifetime qualification for shipped pure CUDA graph regions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime import graphs  # noqa: E402
from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope  # noqa: E402
from cozy_runtime.author._model import WARNINGS  # noqa: E402

CUDA = torch.cuda.is_available()


def _mlp(device: str) -> Any:
    torch.manual_seed(0)
    module = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)
    ).to(device)
    graphs.install(torch, object(), {"mlp": module})
    return module


def _counting() -> Any:
    """Adds how often it ran: a call that depends on state beside its arguments."""

    base: Any = torch.nn.Module

    class Counting(base):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("calls", torch.zeros(()))

        def forward(self, x: Any) -> Any:
            self.calls += 1
            return x + self.calls

    return Counting()


def test_calls_on_a_cpu_run_as_written() -> None:
    module = _mlp("cpu")
    x = torch.randn(4, 64)
    with torch.inference_mode():
        assert torch.equal(module(x), torch.nn.Sequential.__call__(module, x))
    assert type(module).__name__ == "Sequential" and isinstance(module, torch.nn.Sequential)


def test_a_cpu_attention_scope_stays_eager() -> None:
    module = _mlp("cpu")
    x = torch.randn(4, 64)
    with torch.inference_mode(), attention_scope(AttentionLayout(4, 0, 0)):
        assert torch.equal(module(x), torch.nn.Sequential.__call__(module, x))
    assert module not in graphs._RECORDERS


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_recordings_replay_across_calls_and_signatures() -> None:
    module = _mlp("cuda")
    with torch.inference_mode():
        for _ in range(4):
            x = torch.randn(4, 64, device="cuda")
            assert torch.allclose(module(x), torch.nn.Sequential.__call__(module, x))
        recorder = graphs._RECORDERS[module]
        assert len(recorder.recordings) == 1 and recorder.replays == 3  # calls 2, 3, 4
        wider = torch.randn(8, 64, device="cuda")
        for _ in range(2):  # a new signature: as written first, then recorded
            assert torch.allclose(module(wider), torch.nn.Sequential.__call__(module, wider))
        assert len(recorder.recordings) == 2
        token = graphs.BYPASS.set(True)
        try:
            module(wider)
        finally:
            graphs.BYPASS.reset(token)
        assert recorder.replays == 4
        facts = graphs.describe(module)
        assert facts["active_graphs"] == facts["captures"] == facts["graph_pools"] == 2
        assert facts["declared_regions"] == facts["domains"] == 1
        assert facts["captured_regions"] == [""] and facts["replays"] == 4
        assert facts["graph_pool_bytes"] > 0 and facts["domain_workspace_bytes"] > 0
        assert facts["replay_copy_bytes"] > 0
        torch.cuda.synchronize()
        graphs._shed(torch, recorder)
        assert not recorder.recordings
        retired = graphs.describe(module)
        assert retired["active_graphs"] == retired["graph_pools"] == 0
        assert retired["retired_graphs"] == retired["captures"] == 2
        assert retired["domain_workspace_bytes"] == facts["domain_workspace_bytes"]


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_call_that_depends_on_hidden_state_runs_as_written() -> None:
    module = _counting().cuda()
    graphs.install(torch, object(), {"counting": module})
    WARNINGS.clear()
    with torch.inference_mode():
        x = torch.zeros(3, device="cuda")
        for _ in range(4):
            module(x)
    assert module not in graphs._RECORDERS


def _stepped() -> Any:
    """HiDiffusion's patched blocks: a step counter in a plain attribute picks the path, and a
    window shift is drawn from the CPU generator."""

    base: Any = torch.nn.Module

    class Stepped(base):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(64, 64)
            self.timestep = 0

        def forward(self, x: Any) -> Any:
            shift = float(torch.rand(1))
            self.timestep += 1
            return self.linear(x) * self.timestep + shift

    return Stepped()


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_first_request_equals_a_warm_one_for_a_module_with_state() -> None:
    """The owner: measuring and recording never change model state, so a fresh executor's
    first request equals a warm one with the same seed (run 11226 drifted two steps)."""
    torch.manual_seed(1)
    module = _stepped().cuda()
    graphs.install(torch, object(), {"stepped": module})

    def request() -> tuple[list[Any], int]:
        module.timestep = 0
        torch.manual_seed(5)
        x = torch.ones(4, 64, device="cuda")
        return [module(x) for _ in range(3)], module.timestep

    with torch.inference_mode():
        first, steps = request()
        warm, again = request()
    assert steps == again == 3
    assert all(torch.equal(a, b) for a, b in zip(first, warm, strict=True))
    assert module not in graphs._RECORDERS


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_an_unknown_python_hook_keeps_the_region_eager() -> None:
    module = _mlp("cuda")
    called: list[int] = []

    def hook(*_: Any) -> None:
        called.append(1)

    module[1].register_forward_pre_hook(hook)
    with torch.inference_mode():
        for _ in range(5):
            x = torch.randn(4, 64, device="cuda")
            assert torch.allclose(module(x), torch.nn.Sequential.__call__(module, x))
    assert module not in graphs._RECORDERS and len(called) == 10


@pytest.mark.real_gpu
@pytest.mark.skipif(not CUDA, reason="cross-stream graph buffers need a GPU")
def test_replays_on_two_streams_preserve_each_callers_outputs() -> None:
    module = _mlp("cuda")
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    first.wait_stream(torch.cuda.current_stream())
    second.wait_stream(torch.cuda.current_stream())

    def run(stream: Any, value: float, rows: int) -> tuple[Any, Any]:
        with torch.inference_mode(), torch.cuda.stream(stream):
            x = torch.full((rows, 64), value, device="cuda")
            expected = torch.nn.Sequential.__call__(module, x)
            module(x)
            returned = module(x)
            torch.cuda._sleep(1 << 20)
            return returned.square(), expected.square()

    with ThreadPoolExecutor(max_workers=2) as workers:
        a, b = workers.submit(run, first, 1.0, 4), workers.submit(run, second, 2.0, 8)
        outputs = [a.result(), b.result()]
    torch.cuda.synchronize()
    recorder = graphs._RECORDERS[module]
    assert len(recorder.recordings) == 2
    for got, expected in outputs:
        torch.testing.assert_close(got, expected)
    # Retire one executable/pool while the other still embeds the domain's BLAS pointers.
    owner = recorder.domain
    shared = owner.pool
    segments = shared.snapshot(include_traces=False)
    addresses = {s["address"] for s in segments}

    def live(pool: Any) -> int:
        return sum(
            b["size"]
            for s in pool.snapshot(include_traces=False)
            for b in s["blocks"]
            if b["state"] == "active_allocated"
        )

    active = live(shared)
    assert active > 0
    key = next(k for k, v in recorder.recordings.items() if v.inputs[0].shape[0] == 4)
    recorder.recordings.pop(key).close()
    assert live(shared) == active, "a sibling graph reset released its library workspace"
    # Force reuse pressure inside the very pool that owned the workspace, then replay B.
    with torch.cuda.use_mem_pool(shared):
        for _ in range(3):
            churn = torch.empty(active // 4, dtype=torch.float32, device="cuda").fill_(17)
            del churn
    torch.cuda.synchronize()
    assert live(shared) == active
    got, expected = run(second, 3.0, 8)
    torch.testing.assert_close(got, expected)
    physical = sum(s["total_size"] for s in shared.snapshot(include_traces=False))
    free_before = torch.cuda.mem_get_info()[0]
    graphs.release(torch, module)
    assert live(shared) == 0 and owner.closed and owner.stream is owner.control is None
    del shared
    torch.cuda.synchronize()
    # Pool destruction, without a global flush, returns physical segments to the allocator.
    assert not addresses.intersection(s["address"] for s in torch.cuda.memory_snapshot())
    assert torch.cuda.mem_get_info()[0] >= free_before + physical
    # Returned copies outlive both the executable and its private allocator pool.
    for got, expected in outputs:
        torch.testing.assert_close(got, expected)


@pytest.mark.real_gpu
@pytest.mark.skipif(not CUDA, reason="graph retirement needs real completion events")
def test_a_pool_is_not_shed_until_its_last_output_copy_finishes() -> None:
    module = _mlp("cuda")
    with torch.inference_mode():
        x = torch.ones(4, 64, device="cuda")
        module(x)
        module(x)
        torch.cuda.synchronize()
        recorder = graphs._RECORDERS[module]
        torch.cuda._sleep(1 << 29)
        result = module(x)
        assert graphs._shed(torch, recorder) == 0 and recorder.recordings
        torch.cuda.synchronize()
        assert graphs._shed(torch, recorder) > 0 and not recorder.recordings
        torch.testing.assert_close(result, torch.nn.Sequential.__call__(module, x))


def test_a_custom_call_is_not_given_a_library_purity_contract() -> None:
    calls: list[int] = []
    base: Any = torch.nn.Linear

    class CustomCall(base):  # type: ignore[misc]
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return super().__call__(*args, **kwargs)

    module = CustomCall(4, 4)
    assert graphs.install(torch, object(), {"custom": module}) == []
    with torch.inference_mode():
        module(torch.ones(1, 4))
    assert calls == [1] and module not in graphs._RECORDERS


def test_inherited_forward_does_not_grant_a_custom_call_impl_purity() -> None:
    calls: list[int] = []
    base: Any = torch.nn.Linear

    class PackageLinear(base):
        def _call_impl(self, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return super()._call_impl(*args, **kwargs)

    module = PackageLinear(4, 4)
    assert graphs.install(torch, object(), {"custom": module}) == []
    with torch.inference_mode():
        for _ in range(4):
            module(torch.ones(1, 4))
    assert calls == [1] * 4 and module not in graphs._RECORDERS


def test_only_runtime_generated_classes_unwrap_to_known_contracts() -> None:
    module = torch.nn.Linear(4, 4)
    assert graphs.install(torch, object(), {"plain": module}) == ["plain"]
    assert graphs.safe(torch, module)
    base: Any = type(module)

    class PackageSubclass(base):
        pass

    inherited = PackageSubclass(4, 4)
    assert not graphs.safe(torch, inherited)
    assert graphs.install(torch, object(), {"inherited": inherited}) == []


def test_execution_wrapper_does_not_hide_a_package_instance_forward() -> None:
    from cozy_runtime.internal import execution

    calls: list[int] = []
    module = torch.nn.Linear(4, 4)
    original = module.forward

    def package(x: Any) -> Any:
        calls.append(1)
        return original(x)

    module.forward = package
    execution.install(module)
    assert graphs.install(torch, object(), {"package": module}) == []
    with torch.inference_mode():
        module(torch.ones(1, 4))
    assert calls == [1] and module not in graphs._RECORDERS
