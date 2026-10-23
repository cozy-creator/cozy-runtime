"""Development numerical and lifetime qualification for shipped pure CUDA graph regions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime import graphs  # noqa: E402
from cozy_runtime.author import ConformanceError, Model, graph_keys  # noqa: E402
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


def _idle() -> None:
    """The device idle after a request: the queued signatures record."""
    assert graphs.capture_idle(lambda: False)


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


def test_a_root_attribute_that_changes_undeclared_is_reported() -> None:
    """Between a component root's first two calls, a plain attribute that changed and is not
    declared in `__graph_keys__` is reported (`task conform` fails on it); a declared one keys
    its graphs, and later calls are not watched."""

    class Declares:
        __graph_keys__ = MappingProxyType({"net": ("mode",)})

    net = torch.nn.Linear(4, 4)
    graphs.install(torch, Declares(), {"net": net})
    graphs.UNDECLARED.clear()
    with torch.inference_mode():
        for mode, step in ((1, 0), (2, 1), (2, 2)):
            net.mode, net.step = mode, step
            net(torch.ones(1, 4))
    assert {"Linear": {"step"}} == graphs.UNDECLARED


def test_graph_keys_declares_the_attributes_that_steer_a_component() -> None:
    class Piped(Model[object]):
        pass

    assert graph_keys(unet=("hidiffusion_active", "config.mode"))(Piped).__graph_keys__ == {
        "unet": ("hidiffusion_active", "config.mode")
    }
    with pytest.raises(ConformanceError):
        graph_keys(unet="hidiffusion_active")(Piped)  # type: ignore[arg-type]


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
        for request in range(2):  # one as written, then one that replays
            for _ in range(2):
                x = torch.randn(4, 64, device="cuda")
                assert torch.allclose(module(x), torch.nn.Sequential.__call__(module, x))
            recorder = graphs._RECORDERS[module]
            assert recorder.capture_attempts == request  # nothing records inside a request
            _idle()
        assert len(recorder.recordings) == 1 and recorder.replays == 2
        wider = torch.randn(8, 64, device="cuda")
        assert torch.allclose(module(wider), torch.nn.Sequential.__call__(module, wider))
        _idle()  # a new signature of a measured region records after its first call
        assert torch.allclose(module(wider), torch.nn.Sequential.__call__(module, wider))
        assert len(recorder.recordings) == 2
        token = graphs.BYPASS.set(True)
        try:
            module(wider)
        finally:
            graphs.BYPASS.reset(token)
        assert recorder.replays == 3
        facts = graphs.describe(module)
        assert facts["active_graphs"] == facts["captures"] == 2 and facts["graph_pools"] == 1
        assert facts["declared_regions"] == facts["domains"] == 1
        assert facts["captured_regions"] == [""] and facts["replays"] == 3
        assert facts["graph_pool_bytes"] == 0  # unplaced: no varena memory prepares its pool
        assert facts["replay_copy_bytes"] > 0
        torch.cuda.synchronize()
        # Per region, design §5's measures: captures and their time, and each way it ran (three
        # calls as written, three replays), host submission and device time, from events alone;
        # each idle capture's cost, its launch-bound measure, a replay's saving and its payback.
        region = graphs.describe(module)["regions"][""]
        assert region["captures"] == 2 and region["capture_ms"] > 0
        assert region["eager"]["calls"] == 3 and region["replay"]["calls"] == 3
        for way in ("eager", "replay"):
            assert region[way]["device_timed"] == region[way]["calls"], region
            assert region[way]["host_ms"] > 0 and region[way]["device_ms"] > 0, region
        graph = region["graph"]
        assert len(graph["capture_each_ms"]) == 2 and len(graph["measured"]) == 2, graph
        assert all(m["busy_ms"] < m["submit_ms"] for m in graph["measured"]), graph
        assert graph["saving_ms"] is not None and not graph["eager"], graph
        recorder.drop()  # varena dropped the region: its graphs go
        assert not recorder.recordings and recorder.pool is None
        retired = graphs.describe(module)
        assert retired["active_graphs"] == retired["graph_pools"] == 0
        assert retired["retired_graphs"] == retired["captures"] == 2


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_forward_that_changes_its_own_state_never_records() -> None:
    """A replay would not keep its count: the idle period finds the state the recording runs
    changed, puts it back and leaves the component as written."""
    module = _counting().cuda()
    graphs.install(torch, object(), {"counting": module})
    with torch.inference_mode():
        x = torch.zeros(3, device="cuda")
        for _ in range(2):
            module(x)
        _idle()
        for _ in range(2):
            module(x)
    recorder = graphs._RECORDERS[module]
    assert not recorder.recordings and recorder.replays == 0
    (why,) = recorder.judged.values()
    assert "calls" in why, why
    assert module.calls.item() == 4  # the four real calls' count: recording left none of its own


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
        _idle()
        warm, again = request()
    assert steps == again == 3
    assert all(torch.equal(a, b) for a, b in zip(first, warm, strict=True))
    recorder = graphs._RECORDERS[module]
    assert not recorder.recordings and "timestep" in next(iter(recorder.judged.values()))


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_package_hook_keeps_a_component_as_written() -> None:
    """A hook runs Python at every call, which a replay would skip."""
    module = _mlp("cuda")
    called: list[int] = []

    def hook(*_: Any) -> None:
        called.append(1)

    module[1].register_forward_pre_hook(hook)
    with torch.inference_mode():
        for turn in range(5):
            x = torch.randn(4, 64, device="cuda")
            assert torch.allclose(module(x), torch.nn.Sequential.__call__(module, x))
            if turn == 2:
                _idle()
    recorder = graphs._RECORDERS[module]
    assert not recorder.recordings and len(called) == 10
    assert "hook" in next(iter(recorder.judged.values()))


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
            returned = module(x)
            torch.cuda._sleep(1 << 20)
            return returned.square(), expected.square()

    for stream, rows in ((first, 4), (second, 8), (first, 4)):  # each signature as written
        run(stream, 0.0, rows)
    _idle()
    with ThreadPoolExecutor(max_workers=2) as workers:
        a, b = workers.submit(run, first, 1.0, 4), workers.submit(run, second, 2.0, 8)
        outputs = [a.result(), b.result()]
    torch.cuda.synchronize()
    recorder = graphs._RECORDERS[module]
    assert len(recorder.recordings) == 2
    for got, expected in outputs:
        torch.testing.assert_close(got, expected)
    # Retire one executable while the other, in the same pool, still embeds the library
    # stream's workspace pointers (the process's, made outside any capture: `prime_blas`).
    owner = recorder.domain
    assert owner is not None and owner.primed and recorder.pool is not None
    key = next(k for k, v in recorder.recordings.items() if v.inputs[0].shape[0] == 4)
    recorder.recordings.pop(key).close()
    got, expected = run(second, 3.0, 8)
    torch.testing.assert_close(got, expected)
    graphs.release(torch, module)
    assert owner.closed and owner.stream is owner.control is None and recorder.pool is None
    torch.cuda.synchronize()
    # Returned copies outlive both the executable and its pool.
    for got, expected in outputs:
        torch.testing.assert_close(got, expected)


def _two_segments() -> Any:
    """A package's root module of its own, the second segment reading the first."""
    torch.manual_seed(0)
    base: Any = torch.nn.Module

    class TwoSegments(base):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Sequential(
                torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)
            )
            self.second = torch.nn.Sequential(
                torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)
            )

        def forward(self, x: Any) -> Any:
            return self.second(self.first(x)) * 2

    module = TwoSegments().cuda()
    assert graphs.install(torch, object(), {"two": module}) == ["two"]
    return module


def _as_written(module: Any, x: Any) -> Any:
    token = graphs.BYPASS.set(True)
    try:
        return module(x)
    finally:
        graphs.BYPASS.reset(token)


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_component_of_package_code_records_whole() -> None:
    """No purity contract: a component's root is its region whatever code it runs, and its
    replay is served once it matches eager bit for bit."""
    module = _two_segments()
    with torch.inference_mode():
        for _ in range(2):
            x = torch.randn(4, 64, device="cuda")
            torch.testing.assert_close(module(x), _as_written(module, x))
        _idle()
        for _ in range(2):
            x = torch.randn(4, 64, device="cuda")
            torch.testing.assert_close(module(x), _as_written(module, x))
    recorder = graphs._RECORDERS[module]
    assert recorder.replays == 2 and module.first not in graphs._RECORDERS
    assert graphs.describe(module)["captured_regions"] == [""]
    graphs.release(torch, module)


def _stepping_unet(device: str) -> Any:
    """sdxl's HiDiffusion UNet with HiDiffusion off: a package subclass the package sets a step
    counter on before every call (v6 13590 recaptured the whole UNet every warm step)."""
    diffusers = pytest.importorskip("diffusers")
    base: Any = diffusers.UNet2DConditionModel

    class Stepping(base):  # type: ignore[misc]
        cozy_step = 0

    torch.manual_seed(2)
    return Stepping.from_config(
        {
            "sample_size": 8,
            "block_out_channels": (16, 32),
            "layers_per_block": 1,
            "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
            "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
            "attention_head_dim": (2, 4),
            "cross_attention_dim": 16,
            "norm_num_groups": 8,
        }
    ).to(device)


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_cold_request_runs_as_written_and_the_next_replays_every_step() -> None:
    """The owner: nothing records on a request's path. Twenty cold steps run as written, the
    idle period records the root once, and twenty warm steps replay it."""
    unet = _stepping_unet("cuda")
    graphs.install(torch, object(), {"unet": unet})
    gen = torch.Generator(device="cuda").manual_seed(3)
    sample = torch.randn(2, 4, 8, 8, device="cuda", generator=gen)
    context = torch.randn(2, 77, 16, device="cuda", generator=gen)

    def request() -> Any:
        for step in range(20):
            unet.cozy_step = step
            timestep = torch.tensor(999 - step * 50, device="cuda")  # a scheduler's, a tensor
            got = unet(sample, timestep, encoder_hidden_states=context).sample
        return got, timestep

    with torch.inference_mode():
        request()
        cold = graphs.describe(unet)
        assert cold["captures"] == cold["capture_attempts"] == cold["replays"] == 0
        _idle()
        got, timestep = request()
        token = graphs.BYPASS.set(True)
        want = unet(sample, timestep, encoder_hidden_states=context).sample
        graphs.BYPASS.reset(token)
    recorder = graphs._RECORDERS[unet]
    assert (recorder.capture_attempts, recorder.replays) == (1, 20)
    torch.testing.assert_close(got, want)
    graphs.release(torch, unet)


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_device_bound_region_never_records() -> None:
    """Its eager calls submit faster than the device works: a graph would save nothing, so the
    idle period measures it and leaves it as written."""
    torch.manual_seed(0)
    module = torch.nn.Sequential(torch.nn.Linear(4096, 4096), torch.nn.Linear(4096, 4096)).cuda()
    graphs.install(torch, object(), {"wide": module})
    with torch.inference_mode():
        x = torch.randn(4096, 4096, device="cuda")
        for _ in range(3):
            module(x)
        _idle()
        module(x)
    recorder = graphs._RECORDERS[module]
    assert not recorder.recordings and recorder.replays == 0
    (why,) = recorder.judged.values()
    assert why.startswith("device-bound"), why
    recorder.drop()
    assert recorder.judged  # a measured verdict outlives the graphs: it never queues again
    ((submit, busy),) = recorder.bound.values()
    assert busy > submit
    graphs.release(torch, module)


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_python_number_that_changes_every_call_is_named() -> None:
    """A timestep passed as a Python number keys every call apart, so the root records again
    every call: the receipts count the change by argument and the run warns once, by name."""
    unet = _stepping_unet("cuda")
    graphs.install(torch, object(), {"unet": unet})
    gen = torch.Generator(device="cuda").manual_seed(4)
    sample = torch.randn(2, 4, 8, 8, device="cuda", generator=gen)
    context = torch.randn(2, 77, 16, device="cuda", generator=gen)
    WARNINGS.clear()
    with torch.inference_mode():
        for step in range(3):
            unet(sample, 999 - step * 50, encoder_hidden_states=context)
    changes = graphs.describe(unet)["regions"][""]["key_changes"]
    assert changes == {"value [0][1]": 2}, changes
    assert any(
        what == "a graph region records again every call" and fields["argument"] == "[0][1]"
        for what, fields in WARNINGS
    ), WARNINGS
    graphs.release(torch, unet)


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_component_root_copies_its_results_out() -> None:
    """Its results leave the component (a package loop keeps them past its next call): a root
    region copies them out, so a held result neither blocks the next replay nor changes."""
    module = _mlp("cuda")
    with torch.inference_mode():
        x = torch.randn(4, 64, device="cuda")
        module(x)
        module(x)
        _idle()
        held = module(x)  # replayed
        kept = held.clone()
        recorder = graphs._RECORDERS[module]
        other = torch.randn(4, 64, device="cuda")
        torch.testing.assert_close(module(other), _as_written(module, other))
        assert recorder.replays == 2 and recorder.held == 0 and torch.equal(held, kept)
        entry = next(iter(recorder.recordings.values()))
        assert held.data_ptr() not in {t.data_ptr() for t in entry.statics}
        assert recorder.copy_bytes == 2 * (x.nbytes + held.nbytes)
    graphs.release(torch, module)
