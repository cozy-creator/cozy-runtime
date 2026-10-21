"""Immutable LoRA fill selections, exact bytes, and the root's synchronous boundary."""

from __future__ import annotations

import gc
import inspect
import threading
import weakref
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
import tensorfs  # noqa: E402
from tensorfs.varena.torch import Declared, Memory, Unbound  # noqa: E402
from test_graph_cache_gpu import residency  # noqa: E402

from cozy_runtime import graphs  # noqa: E402
from cozy_runtime.internal.fill import Checkpoint  # noqa: E402
from cozy_runtime.internal.weights import _Bakes, _forward_arguments  # noqa: E402

BASE = [[1.0, 2.0], [3.0, 4.0]]
EXPECTED = {None: BASE, "A": [[3.0, 2.0], [6.0, 4.0]], "B": [[1.0, 7.0], [3.0, 11.0]]}


def test_actual_h3_named_and_positional_selector_ignore_wrapped_instance_forward() -> None:
    diffusers = pytest.importorskip("diffusers")
    root = diffusers.MiniMaxH3Transformer3DModel(
        num_attention_heads=1,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=8,
        freq_dim=4,
        time_embed_hidden_dim=8,
        time_embed_dim=4,
        rope_freq_dim=1,
    )
    signature = inspect.signature(type(root).forward)
    declared = root.forward

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return declared(*args, **kwargs)

    root.forward = wrapped
    graphs.install(torch, object(), {"net": root})
    selector = {"source": "immutable-source"}
    named = {
        p.name: object()
        for p in tuple(signature.parameters.values())[1:]
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    named["attention_kwargs"] = selector
    assert _forward_arguments(root, (), named)["attention_kwargs"] is selector
    positional: list[object] = []
    for parameter in tuple(signature.parameters.values())[1:]:
        if parameter.name == "attention_kwargs":
            positional.append(selector)
            break
        assert parameter.kind != inspect.Parameter.KEYWORD_ONLY
        positional.append(named.get(parameter.name, parameter.default))
    assert _forward_arguments(root, tuple(positional), {})["attention_kwargs"] is selector


module_type: Any = torch.nn.Module


class Factors(module_type):  # type: ignore[misc]
    def __init__(self, identity: str) -> None:
        super().__init__()
        self.down = torch.nn.Parameter(torch.empty(1, 2))
        self.up = torch.nn.Parameter(torch.empty(2, 1))
        self.identity = identity


class Selection:
    _cozy_baked = False

    def __init__(self, factors: dict[str, Factors]) -> None:
        self.factors = factors
        self.armed: str | None = None

    def lora_for_call(self, arguments: Any) -> Any:
        key = arguments.get("adapter")
        if key is None:
            return None
        value = self.factors[key]
        return value.identity, value.down, value.up, 1.0

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        if self.armed is None or self._cozy_baked:
            return output
        factor = self.factors[self.armed]
        return output + torch.nn.functional.linear(
            torch.nn.functional.linear(args[0], factor.down), factor.up
        )


class Block(module_type):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x: Any) -> Any:
        return self.linear(x)


class Root(module_type):  # type: ignore[misc]
    _no_split_modules = ["Block"]

    def __init__(self, selection: Selection) -> None:
        super().__init__()
        self.block = Block()
        self.block.linear.register_forward_hook(selection)

        def arm(_: Any, args: Any, kwargs: Any) -> None:
            selection.armed = kwargs.get("adapter")

        def disarm(*_: Any) -> None:
            selection.armed = None

        self.register_forward_pre_hook(arm, with_kwargs=True)
        self.register_forward_hook(disarm, always_call=True)

    def forward(self, x: Any, *, adapter: str | None = None, fail: bool = False) -> Any:
        if fail:
            raise ValueError("deliberate selected-forward failure")
        return self.block(x)


def fixture(folder: Path, device: str = "cpu") -> tuple[Any, ...]:
    writer = tensorfs.Writer(cache_dir=folder)
    writer.component("net", cls="", config={}).put(
        "block.linear.weight", torch.tensor(BASE).numpy(), dtype="F32", shape=[2, 2]
    )
    for name, down, up in (
        ("A", [[1.0, 0.0]], [[2.0], [3.0]]),
        ("B", [[0.0, 1.0]], [[5.0], [7.0]]),
    ):
        component = writer.component(name, cls="", config={})
        component.put("down", torch.tensor(down).numpy(), dtype="F32", shape=[1, 2])
        component.put("up", torch.tensor(up).numpy(), dtype="F32", shape=[2, 1])
    checkpoint = writer.commit()
    memory = Memory.open(torch.device(device))
    factors, factor_models = {}, []
    for name in ("A", "B"):
        value = Factors(f"{checkpoint.id}/{name}/linear")
        factor_models.append(memory.load(value, ck=checkpoint, component=name, level="gpu"))
        factors[name] = value
    hook = Selection(factors)
    root = Root(hook)
    rows = {row.name: row for row in Checkpoint(checkpoint).rows("net")}
    bakes = _Bakes(torch, root, rows, {"block"})
    root._varena = Declared(fill=bakes.pending())
    model = memory.load(root, ck=checkpoint, component="net", level="gpu")
    return root, hook, bakes, model, factor_models, memory, checkpoint


def fill_bytes(snapshot: Any, checkpoint: Any) -> Any:
    # Real TensorFS read and the real Fill callback, with tiny exact F32 operands.
    array = checkpoint.components["net"].tensors.read("block.linear.weight")
    value = torch.from_dlpack(array).clone()
    snapshot(None, {"block.linear.weight": {"value": value}})
    return value


def test_selected_bytes_and_fingerprint_base_a_same_a_b_base(tmp_path: Path) -> None:
    root, hook, bakes, model, _, memory, checkpoint = fixture(tmp_path)
    versions = []
    try:
        assert repr(hook.factors["A"]) == repr(hook.factors["B"]), (
            "module repr is not source provenance"
        )
        for adapter in (None, "A", "A", "B", None):
            snapshot = bakes.snapshot({"adapter": adapter})
            versions.append(snapshot.prepare())
            hook.armed = "B" if adapter != "B" else "A"
            assert torch.equal(fill_bytes(snapshot, checkpoint), torch.tensor(EXPECTED[adapter]))
        assert versions[0] == versions[4]
        assert versions[1] == versions[2]
        assert len(set(versions)) == 3
        # A late fill uses A even after all live selection state changes to B.
        frozen = bakes.snapshot({"adapter": "A"})
        bakes.strategy = "per_step"
        hook.armed = "B"
        assert torch.equal(fill_bytes(frozen, checkpoint), torch.tensor(EXPECTED["A"]))
    finally:
        memory.close()
        checkpoint.close()


def test_real_call_selection_waits_for_each_attempt_and_same_a_keeps_content_identity(
    tmp_path: Path,
) -> None:
    root, _, bakes, model, fixed, memory, checkpoint = fixture(tmp_path)
    placed = residency(memory, model)
    placed.bakes["net"] = bakes
    bakes.install_call(model)
    seen: list[object] = []

    def selected_before_prehooks(_: Any, args: Any, kwargs: Any) -> None:
        selected = model.fill
        assert selected is not None and selected.prefetch_ready
        # The real immutable Fill reads the real checkpoint and selects before
        # the mutable hook arming that follows this prepended observer.
        value = fill_bytes(selected, checkpoint)
        with torch.no_grad():
            root.block.linear.weight.copy_(value)
        seen.append(selected.prepare())

    handle = root.register_forward_pre_hook(
        selected_before_prehooks, with_kwargs=True, prepend=True
    )
    try:
        for attempt, adapters in enumerate((("A", "B", None, "A"), ("A",)), 1):
            placed.open_attempt()
            assert model.fill is not None and not model.fill.prefetch_ready
            assert all(current.fill is None for current in fixed)
            raw = (fixed[0], (model, fixed[1]), (model, fixed[1]))
            with memory.request(f"attempt-{attempt}", plan=raw) as request:
                assert request.declared == (
                    (fixed[0].name,),
                    tuple(sorted((model.name, fixed[1].name))),
                    tuple(sorted((model.name, fixed[1].name))),
                )
                assert model in request._pending
                for adapter in adapters:
                    result = root(torch.eye(2), adapter=adapter)
                    assert torch.equal(result, torch.tensor(EXPECTED[adapter]).t())
                    assert not request._pending
        assert seen[0] == seen[-2] == seen[-1], "same A changed immutable content identity"
        assert len(set(seen)) == 3, "A/B/inactive selections were conflated"
        with memory.request("waiting", plan=raw) as waiting:
            root(torch.eye(2), adapter="A")
            expected_a = waiting._selected[model]
            with memory.request("selected", plan=raw) as active:
                root(torch.eye(2), adapter="B")
                assert not active._pending
                waiting.plan(raw)
                assert waiting._selected[model] == expected_a
                assert active._selected[model] != expected_a
                assert model in waiting._pending, "A borrowed the globally selected B content"
            result = root(torch.eye(2), adapter="A")
            assert torch.equal(result, torch.tensor(EXPECTED["A"]).t())
            assert waiting._selected[model] == expected_a and not waiting._pending
    finally:
        handle.remove()
        memory.close()
        checkpoint.close()


def test_fixed_stored_lora_transform_is_ready_without_call_arguments(tmp_path: Path) -> None:
    root = torch.nn.Module()
    root.block = Block()
    layer = torch.nn.Module()
    layer._cozy_baked = False
    layer.base_layer = root.block.linear
    layer.lora_A = torch.nn.ModuleDict({"fixed": torch.nn.Linear(2, 1, bias=False)})
    layer.lora_B = torch.nn.ModuleDict({"fixed": torch.nn.Linear(1, 2, bias=False)})
    layer.active_adapters = ["fixed"]
    layer.scaling = {"fixed": 1.0}
    root.block.linear = layer
    values = {
        "block.linear.base_layer.weight": BASE,
        "block.linear.lora_A.fixed.weight": [[1.0, 0.0]],
        "block.linear.lora_B.fixed.weight": [[2.0], [3.0]],
    }
    writer = tensorfs.Writer(cache_dir=tmp_path)
    component = writer.component("net", cls="", config={})
    for key, value in values.items():
        tensor = torch.tensor(value)
        component.put(key, tensor.numpy(), dtype="F32", shape=list(tensor.shape))
    checkpoint = writer.commit()
    try:
        rows = {row.name: row for row in Checkpoint(checkpoint).rows("net")}
        bakes = _Bakes(torch, root, rows, {"block"})
        assert bakes.bakes and not bakes.call_selected
        pending = bakes.pending()
        assert pending.prefetch_ready
        assert pending.prepare() == bakes.snapshot({}).prepare()
        parts = {
            key: {"value": torch.from_dlpack(checkpoint.components["net"].tensors.read(key)).clone()}
            for key in values
        }
        pending(None, parts)
        assert torch.equal(parts["block.linear.base_layer.weight"]["value"], torch.tensor(EXPECTED["A"]))
    finally:
        checkpoint.close()


def test_failure_restores_selection_and_reentrant_same_selection_is_allowed(tmp_path: Path) -> None:
    root, hook, bakes, model, _, memory, checkpoint = fixture(tmp_path)
    try:
        graphs.install(torch, object(), {"net": root})
        bakes.install_call(model)
        root(torch.eye(2), adapter="A")
        selected = model.fill
        assert selected is not None
        with pytest.raises(ValueError, match="deliberate selected-forward"):
            root(torch.eye(2), adapter="B", fail=True)
        assert model.fill is selected and hook._cozy_baked
        with bakes.call(model, {"adapter": "A"}):
            held = model.fill
            with bakes.call(model, {"adapter": "A"}):
                assert model.fill is held
            with pytest.raises(Unbound, match="nested call"):
                with bakes.call(model, {"adapter": "B"}):
                    pytest.fail("a different nested selection entered")
            assert model.fill is held
        value = torch.ones(2, 2)
        ref = weakref.ref(value)
        root(value, adapter="A")
        del value
        gc.collect()
        assert ref() is None, "snapshot retained an invocation activation"
    finally:
        memory.close()
        checkpoint.close()


def test_different_threads_cannot_replace_a_live_selection(tmp_path: Path) -> None:
    _, _, bakes, model, _, memory, checkpoint = fixture(tmp_path)
    entered, release, other = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def first() -> None:
        try:
            with bakes.call(model, {"adapter": "A"}):
                entered.set()
                assert release.wait(5)
                assert model.fill is not None
                assert torch.equal(fill_bytes(model.fill, checkpoint), torch.tensor(EXPECTED["A"]))
        except BaseException as error:
            errors.append(error)

    def second() -> None:
        try:
            assert entered.wait(5)
            with bakes.call(model, {"adapter": "B"}):
                other.set()
                assert model.fill is not None
                assert torch.equal(fill_bytes(model.fill, checkpoint), torch.tensor(EXPECTED["B"]))
        except BaseException as error:
            errors.append(error)

    a, b = threading.Thread(target=first), threading.Thread(target=second)
    try:
        a.start()
        b.start()
        assert entered.wait(5)
        assert not other.wait(0.05), "another thread replaced A while its forward was active"
        release.set()
        a.join(5)
        b.join(5)
        assert not a.is_alive() and not b.is_alive()
        assert other.is_set() and not errors, errors
    finally:
        release.set()
        a.join(5)
        b.join(5)
        memory.close()
        checkpoint.close()


@pytest.mark.real_gpu
def test_real_page_in_switches_before_prehooks_and_reuses_only_same_fingerprint(
    tmp_path: Path,
) -> None:
    assert torch.cuda.device_count() == 1
    root, hook, bakes, model, factors, memory, checkpoint = fixture(tmp_path, "cuda:0")
    registrations, versions, copied = [], [], []
    placed = residency(memory, model)
    placed.bakes["net"] = bakes
    try:
        graphs.install(torch, object(), {"net": root})
        bakes.install_call(model)
        with torch.inference_mode(), memory.request("switch", plan=[(model, *factors)]):
            for adapter in (None, "A", "A", "A", "B", None):
                result = root(torch.eye(2, device="cuda"), adapter=adapter)
                assert torch.equal(result.cpu(), torch.tensor(EXPECTED[adapter]).t())
                registrations.append(model.ws.id)
                assert model.fill is not None
                versions.append(model.fill.prepare())
                copied.append(len(memory.native.stats(memory._current().native_id)["copies"]))
            assert registrations[1] == registrations[2] == registrations[3]
            assert registrations[0] != registrations[1] != registrations[4]
            assert copied[1] == copied[2] == copied[3], (
                "same A unexpectedly refilled resident pages"
            )
            assert versions[0] == versions[-1]
            assert versions[1] == versions[2] == versions[3]
            assert len(set(versions)) == 3
        warm_id = None
        for attempt in range(2):
            placed.open_attempt()
            assert model.fill is not None and not model.fill.prefetch_ready
            with memory.request(f"warm-A-{attempt}", plan=[(model, *factors)]) as request:
                assert model in request._pending
                result = root(torch.eye(2, device="cuda"), adapter="A")
                assert torch.equal(result.cpu(), torch.tensor(EXPECTED["A"]).t())
                assert not request._pending
                if warm_id is not None:
                    assert model.ws.id == warm_id
                    assert not memory.native.stats(request.native_id)["copies"], (
                        "readiness-only activation refilled unchanged warm A"
                    )
                warm_id = model.ws.id
    finally:
        for token in placed.pins.values():
            token.cache.close()
        placed.pins.clear()
        memory.close()
        checkpoint.close()
