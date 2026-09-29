"""Published Sol routes are request state, including across real CPU rank control."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from torch import Tensor

from cozy_runtime.author._errors import UnsupportedInput
from cozy_runtime.internal import attention_sol as sol
from cozy_runtime.internal import kernel_cache, kernel_compile


def job_for(tokens: int, capability: tuple[int, int]) -> kernel_compile.Job:
    shape = sol.variant(tokens, capability)
    return kernel_compile.Job(
        "sol", kernel_cache.Key.of("sol-attn", {"arch": capability, "variant": shape}), {}, {}
    )


def setup_store(
    monkeypatch: pytest.MonkeyPatch, path: Path, capability: tuple[int, int] = (9, 0)
) -> kernel_cache.Store:
    store = kernel_cache.Store(path)
    monkeypatch.setattr(sol, "_capability", lambda device: capability)
    monkeypatch.setattr(sol, "_job_for", job_for)
    monkeypatch.setattr(sol, "_scope", lambda cap: "test-" + str(cap))
    monkeypatch.setattr(kernel_cache, "configured", lambda: store)
    monkeypatch.setattr(sol, "_KERNELS", {})
    monkeypatch.setattr(sol, "load", lambda data: sol.Loaded(data))
    return store


def publish(
    store: kernel_cache.Store, tokens: int = 63, capability: tuple[int, int] = (9, 0)
) -> kernel_compile.Job:
    found = job_for(tokens, capability)
    kernel_compile.learn(store, sol._scope(capability), found)
    store.put(found.key, b"compiled-test-object", {})
    return found


def test_mid_request_publication_does_not_change_replay_or_next_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = setup_store(monkeypatch, tmp_path)
    selected = sol.capture(None)
    with sol.selected(selected):
        assert sol._ready(63, None, 5) is None
        publish(store)
        # Same logical forward replay and later denoising steps keep the first route.
        assert sol._ready(63, None, 5) is None
        assert sol._ready(63, None, 6) is None
    with sol.selected(sol.capture(None)):
        loaded = sol._ready(63, None, 5)
        assert loaded is not None and loaded.function == b"compiled-test-object"


def test_new_length_cannot_borrow_ready_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = setup_store(monkeypatch, tmp_path)
    publish(store, 63)
    with sol.selected(sol.capture(None)):
        assert sol._ready(63, None, 5) is not None
        assert sol._ready(64, None, 5) is None
        publish(store, 64)
        assert sol._ready(64, None, 6) is None


@pytest.mark.parametrize("damage", ["missing", "corrupt", "load"])
def test_selected_artifact_never_falls_back_after_damage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, damage: str
) -> None:
    store = setup_store(monkeypatch, tmp_path)
    found = publish(store)
    selection = sol.capture(None)
    entry = store.entry(found.key)
    assert entry is not None
    path = entry / "object"
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"changed")
    else:

        def broken(data: bytes) -> sol.Loaded:
            raise ValueError("not a native object")

        monkeypatch.setattr(sol, "load", broken)
    with sol.selected(selection), pytest.raises(UnsupportedInput, match="Selected Sol artifact"):
        sol._ready(63, None, 5)


def test_nested_observations_and_exception_do_not_reset_request_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = setup_store(monkeypatch, tmp_path)
    outer = sol.capture(None)
    with sol.selected(outer):
        publish(store)
        with sol.observing(), sol.observing():
            assert sol._ready(63, None, 5) is None
        with pytest.raises(RuntimeError), sol.selected(sol.capture(None)):
            assert sol._ready(63, None, 5) is not None
            raise RuntimeError("nested request failed")
        assert sol._SELECTION.get() is outer
        assert sol._ready(63, None, 6) is None
    assert sol._SELECTION.get() is None
    with pytest.raises(UnsupportedInput, match="request selection scope"):
        sol._ready(63, None, 5)


def test_semantic_intersection_handles_generic_and_shape_specific_architectures() -> None:
    assert sol.intersect([None, ((2, False), (3, True)), ((2, False),)]) == ((2, False),)
    assert sol.intersect([None, None]) is None
    assert sol.intersect([None, ()]) == ()
    selection = sol.Selection(None, (9, 0), {}, ((2, False),))
    with pytest.raises(UnsupportedInput, match="exceeds captured"):
        sol.narrow(selection, None)


def current_variants() -> sol.Variants:
    selection = sol._SELECTION.get()
    assert selection is not None
    return selection.variants


def _rank_control(rank: int, rendezvous: str, output: str) -> None:
    """Real Gloo messages carry the actual Executor Attention command handshake."""
    import torch
    import torch.distributed as dist

    from cozy_runtime.internal import attention, executor_commands
    from cozy_runtime.internal.executor import Executor

    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    patch = pytest.MonkeyPatch()
    try:
        capability = (8, 9) if rank == 0 else (9, 0)
        store = setup_store(patch, Path(output) / f"cache-{rank}", capability)
        found = publish(store, capability=capability)
        applied = attention.Applied({"dit": {"sol-attn": 1}}, ())
        patch.setattr(attention, "observed", lambda *args: applied)
        executor = object.__new__(Executor)
        executor.rank, executor.world, executor.device_kind, executor.torch = rank, 2, "cuda", torch
        executor.construction = "owned-construction"
        executor._attention_hold = contextlib.ExitStack()
        executor._attention_sol = None
        executor.constructions = {executor.construction: executor}
        patch.setattr(executor, "_attention_snapshot", lambda: attention.Snapshot(()))
        patch.setattr(executor, "_attention_roots", lambda: {})
        observed = []
        prior = sol.Selection(None, None, {}, ())
        prior_token = sol._SELECTION.set(prior)
        if rank == 0:

            class Group:
                def broadcast(self, frame: dict[str, object]) -> None:
                    dist.broadcast_object_list([frame], src=0)

                def collect(self, kind: str) -> list[object]:
                    replies: list[object] = [None, None]
                    dist.gather_object(None, replies, dst=0)
                    return [replies[1]]

            patch.setattr(executor, "group", Group(), raising=False)
            for request in range(2):
                try:
                    with executor._ranks_attention("", applied):
                        selected = sol._SELECTION.get()
                        assert selected is not None
                        observed.append(selected.variants)
                        with sol.observing():
                            observed.append(current_variants())
                        if request == 0:
                            raise RuntimeError("authored request failed")
                except RuntimeError as exc:
                    assert request == 0 and str(exc) == "authored request failed"
                assert sol._SELECTION.get() is prior
        else:
            executor.group = None
            for index in range(6):
                frames = [None]
                dist.broadcast_object_list(frames, src=0)
                command = executor_commands.decode(frames[0])
                assert isinstance(command, executor_commands.Attention)
                reply = executor.select_attention(command)
                if index in (1, 4):
                    if index == 1:
                        publish(store, 64, capability)
                    # Follower observation is per mirrored call, not per request.
                    for _ in range(2):
                        with sol.observing():
                            observed.append(current_variants())
                dist.gather_object(reply, dst=0)
                if index in (2, 5):
                    assert sol._SELECTION.get() is prior
        sol._SELECTION.reset(prior_token)
        (Path(output) / f"rank-{rank}.json").write_text(
            json.dumps({"observed": observed, "key": found.key.digest})
        )
    finally:
        patch.undo()
        dist.destroy_process_group()


def test_real_gloo_request_selection_intersects_semantics_and_restores(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    torch.multiprocessing.spawn(
        _rank_control, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True
    )
    rows = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert rows[0]["key"] != rows[1]["key"]  # Architecture keys are deliberately different.
    expected = [[[1, False]], [[1, False]], [[1, False], [1, True]], [[1, False], [1, True]]]
    assert all(row["observed"] == expected for row in rows)


def test_capture_reads_only_entry_metadata_and_loads_only_selected_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = setup_store(monkeypatch, tmp_path)
    for tokens in (63, 64, 127, 128):
        publish(store, tokens)
    original = store.get
    reads = []

    def read(key: kernel_cache.Key) -> bytes | None:
        reads.append(key)
        return original(key)

    monkeypatch.setattr(store, "get", read)
    selection = sol.capture(None)
    assert len(selection.jobs) == 4 and not selection.loaded and reads == []
    with sol.selected(selection):
        sol._ready(63, None, 4)
        sol._ready(63, None, 5)
    assert len(reads) == len(selection.loaded) == 1


def test_wrong_advisory_length_does_not_select_the_actual_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = setup_store(monkeypatch, tmp_path)
    publish(store, 63)
    monkeypatch.setattr(sol, "_SUBMITTED", set())
    monkeypatch.setattr(
        kernel_compile, "submit", lambda *args: kernel_compile.State("sol", "compiling")
    )
    sol.expect(999, None)
    with sol.selected(sol.capture(None)):
        assert sol._ready(63, None, 5) is not None
        assert sol._ready(999, None, 5) is None


def test_real_cpu_dense_route_stays_fixed_while_new_request_adopts_sparse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.author._attention_scope import AttentionLayout

    store = setup_store(monkeypatch, tmp_path, (8, 9))
    monkeypatch.setattr(sol, "specialize", lambda *args: None)
    monkeypatch.setattr(sol, "_native", lambda q, k, v, *args: q.clone() * 0.25)

    def dense(*, query: Tensor, key: Tensor, value: Tensor, scale: float | None) -> Tensor:
        return cast(
            "Tensor",
            torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scale
            ).transpose(1, 2),
        )

    monkeypatch.setattr(sol, "_DENSE", dense)
    q = torch.randn(1, 4, 2, 128, generator=torch.Generator().manual_seed(3))
    site = sol.Site("dit", "attention")
    with sol.selected(sol.capture(q.device)), sol.observing() as counts:
        first = sol._execute(q, q, q, None, AttentionLayout(4, 0, 5), site)
        publish(store, capability=(8, 9))
        replay = sol._execute(q, q, q, None, AttentionLayout(4, 0, 5), site)
        following_step = sol._execute(q, q, q, None, AttentionLayout(4, 0, 6), site)
        assert torch.equal(first, replay) and torch.equal(first, following_step)
        assert counts["dense_step"] == 3 and counts["sparse"] == 0
    with sol.selected(sol.capture(q.device)), sol.observing() as counts:
        next_request = sol._execute(q, q, q, None, AttentionLayout(4, 0, 5), site)
        assert torch.equal(next_request, q * 0.25)
        assert counts["sparse"] == 1


def test_unrelated_corrupt_historical_metadata_does_not_block_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = setup_store(monkeypatch, tmp_path)
    publish(store, 63)
    malformed = kernel_compile.Job(
        "sol", kernel_cache.Key.of("sol-attn", {"variant": {"blocks": "broken"}}), {}, {}
    )
    kernel_compile.learn(store, sol._scope((9, 0)), malformed)
    malformed_inputs = kernel_compile.Job("sol", kernel_cache.Key("sol-attn", "[]"), {}, {})
    kernel_compile.learn(store, sol._scope((9, 0)), malformed_inputs)
    other = publish(store, 127)
    entry = store.entry(other.key)
    assert entry is not None
    (entry / "entry.json").write_text("corrupt")
    # Corruption of another length must not invalidate the published 63-token route.
    selection = sol.capture(None)
    with sol.selected(selection):
        assert sol._ready(63, None, 5) is not None


@pytest.mark.parametrize("missing", ["capture", "commit"])
def test_leader_refuses_missing_semantic_acknowledgment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    import torch
    from msgspec import UNSET

    from cozy_runtime.author._errors import InvalidRequest
    from cozy_runtime.internal import attention
    from cozy_runtime.internal.executor import Executor

    setup_store(monkeypatch, tmp_path, (8, 9))
    applied = attention.Applied({"dit": {"sol-attn": 1}}, ())
    executor = object.__new__(Executor)
    executor.torch, executor.device_kind, executor.rank = torch, "cuda", 0
    monkeypatch.setattr(executor, "group", object(), raising=False)
    restored = []

    def followers(pin: str | object = UNSET, **fields: object) -> list[dict[str, object]]:
        if pin is UNSET and "sol_variants" not in fields:
            restored.append(True)
            return [{"ok": True, "rank": 1}]
        reply: dict[str, object] = {"ok": True, "rank": 1, "hosts": applied.hosts}
        if (pin is not UNSET and missing != "capture") or (pin is UNSET and missing != "commit"):
            reply["sol_variants"] = ()
        return [reply]

    monkeypatch.setattr(executor, "_followers_attention", followers)
    with (
        pytest.raises(InvalidRequest, match="Sol variants"),
        executor._ranks_attention("", applied),
    ):
        pytest.fail("authored execution must not start")
    assert restored == [True] and sol._SELECTION.get() is None


def test_follower_refuses_sol_without_request_selection_capability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cozy_runtime.internal import attention, executor_commands
    from cozy_runtime.internal.executor import Executor

    applied = attention.Applied({"dit": {"sol-attn": 1}}, ())
    monkeypatch.setattr(attention, "observed", lambda *args: applied)
    executor = object.__new__(Executor)
    executor.rank = 1
    executor._attention_hold = contextlib.ExitStack()
    executor._attention_sol = None
    executor.constructions = {"fixture": executor}
    monkeypatch.setattr(executor, "_attention_snapshot", lambda: attention.Snapshot(()))
    monkeypatch.setattr(executor, "_attention_roots", lambda: {})
    reply = executor.select_attention(executor_commands.Attention(construction="fixture", pin=""))
    assert not reply["ok"] and reply["code"] == "attention_selection_missing"
    assert executor._attention_sol is None and sol._SELECTION.get() is None
