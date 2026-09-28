"""H3 across a group: text and reference conditioning at once, reference latents memoized,
and the rank that hosts the text encoder decoding clips only beside everything it holds.

Real follower processes over gloo run the runtime's own H3Model, conditioner, VAEs and
blocks at a few million parameters (`testdata/h3_conditioning.py`). Every result is compared
bitwise with one process running the same calls in sequence.
"""

from __future__ import annotations

import functools
import os
import socket
import sys
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from cozy_runtime.author import CapabilityError, DerivedCache, concurrently  # noqa: E402
from cozy_runtime.author.fakes import fake_telemetry  # noqa: E402
from cozy_runtime.internal import spawn  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.parallel.group import RankGroup  # noqa: E402
from cozy_runtime.internal.residency import ResidencyRefusal  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from cozy_runtime.models.minimax_h3.model import condition_references  # noqa: E402
from cozy_runtime.models.minimax_h3.official import NumericalChecks  # noqa: E402
from test_staged_demand_pull import (  # noqa: E402
    CAPACITY,
    ENVELOPE,
    SCOPES,
    TURBO,
    Device,
)
from testdata import h3_conditioning as h3  # noqa: E402

FIXTURE = Path(__file__).parent / "testdata" / "h3_conditioning.py"


def _checks(model: Any) -> tuple[NumericalChecks, Any]:
    telemetry = fake_telemetry()
    return NumericalChecks(telemetry, model.pipe.resident), telemetry


def _logged(telemetry: Any, name: str) -> list[dict[str, Any]]:
    return [dict(event.fields) for event in telemetry.events if event.name == name]


def _outputs(state: Any) -> dict[str, Any]:
    return {
        "prompt_embeds": state.prompt_embeds,
        "text_token_tags": state.text_token_tags,
        "condition_latents": list(state.condition_latents),
        "audio_condition_latents": list(state.audio_condition_latents),
    }


def _assert_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for name, value in expected.items():
        if isinstance(value, list):
            assert len(actual[name]) == len(value), name
            assert all(torch.equal(a, b) for a, b in zip(actual[name], value, strict=True)), name
        else:
            assert torch.equal(actual[name], value), name


@pytest.fixture(scope="module")
def sequential() -> dict[str, Any]:
    """One process, the calls in order: what every rank layout must reproduce."""
    torch.set_num_threads(1)
    model = h3.model()
    checks, _ = _checks(model)
    state = h3.start(model)
    model.condition_text(h3.TASK, state, checks=checks)
    model.condition_ref2va_media(h3.TASK, state, checks=checks)
    return _outputs(state)


@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
def test_reference_encode_is_the_official_block_and_its_memo_is_exact(
    sequential: dict[str, Any], device_kind: str
) -> None:
    """The per-reference encode is Diffusers' reference encoder step, value for value, and a
    memo hit hands back exactly what was encoded. Audio is never memoized."""
    if device_kind == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    model = h3.model()
    for name in ("video_vae", "audio_vae"):
        model.pipe.components[name].to(device_kind)
    checks, telemetry = _checks(model)
    official = h3.start(model)
    model.pipe._run(h3.TASK, "vae_encoder", official, component="video_vae")
    runs = []
    for _ in range(2):
        state = h3.start(model)
        model.condition_ref2va_media(h3.TASK, state, checks=checks)
        runs.append(state)
    for state in runs:
        for name in ("condition_latents", "audio_condition_latents"):
            ours, theirs = getattr(state, name), getattr(official, name)
            assert all(torch.equal(a, b) for a, b in zip(ours, theirs, strict=True)), name
    if device_kind == "cpu":
        assert all(
            torch.equal(a, b)
            for a, b in zip(runs[0].condition_latents, sequential["condition_latents"], strict=True)
        )
    # Two pictures and a clip are memoized; the voice is encoded every time.
    assert _logged(telemetry, "h3 reference latents") == [
        {"reused": 0, "encoded": 3},
        {"reused": 3, "encoded": 0},
    ]
    # The memo's own tensors never reach a request.
    runs[1].condition_latents[0].zero_()
    again = h3.start(model)
    model.condition_ref2va_media(h3.TASK, again, checks=checks)
    assert torch.equal(again.condition_latents[0], runs[0].condition_latents[0])


def test_the_derived_cache_drops_its_least_recently_used_entry_at_its_bound() -> None:
    model = h3.model()
    cache = DerivedCache("probe", max_entries=2, max_bytes=1 << 20, owner=model)
    with model._cozy_scope("condition_ref2va_media", ("video_vae", "audio_vae")):
        for key in ("a", "b", "a", "c"):
            cache.get(key, build=functools.partial(torch.full, (4,), ord(key)))
        assert "a" in cache and "c" in cache and "b" not in cache
        with pytest.raises(CapabilityError, match="exceeds its declared bound"):
            cache.get("big", build=lambda: torch.zeros(1 << 19))
    with pytest.raises(CapabilityError, match="outside an active component-use scope"):
        cache.get("a", build=lambda: torch.zeros(1))


def test_calls_that_hold_nothing_elsewhere_run_in_order_here() -> None:
    order: list[str] = []
    main = threading.get_ident()

    def call(name: str) -> str:
        assert threading.get_ident() == main
        order.append(name)
        return name

    assert concurrently(lambda: call("a"), lambda: call("b")) == ("a", "b")
    assert order == ["a", "b"]


class Group:
    """Rank 0 in this process over real followers running `testdata/h3_conditioning.py`."""

    def __init__(self, tmp_path: Path, degree: int, *follower_args: str) -> None:
        def launch(module_argv: list[str], fd: int) -> spawn.Child:
            return spawn.spawn_follower(
                python=sys.executable,
                module_argv=[str(FIXTURE), *module_argv[2:], *follower_args],
                inherit_fd=fd,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )

        self.group = RankGroup(degree=degree, backend="cpu:gloo", root=str(tmp_path), launch=launch)
        self.model = h3.model()
        self.ours, self.theirs = socket.socketpair()
        leader: Any = Executor(Channel(self.ours), tmp_path, world=degree)
        leader.device_kind = "cpu"
        leader.group = self.group
        leader._group_model_key = "fixture"
        leader.backend = SimpleNamespace(
            components=h3.roots(self.model), parked={}, component_bytes=h3.SIZES
        )
        self.leader = leader
        self.tmp_path = tmp_path

    def __enter__(self) -> Group:
        self.group.spawn()
        self.group.form(torch, {})
        self.leader.pg = self.group.pg
        assert self.leader._install_group(torch, self.model, h3.HOSTED_CAPACITY) is None
        self.leader._attempt_spool = self.tmp_path / "attempt"
        return self

    def __exit__(self, *exc: object) -> None:
        self.group.close()
        self.ours.close()
        self.theirs.close()


@pytest.mark.parametrize("degree", [2, 4])
def test_text_and_reference_conditioning_overlap_across_the_group(
    tmp_path: Path, degree: int, sequential: dict[str, Any]
) -> None:
    """Rank 1 runs the hosted conditioner while rank 0 encodes the references, and the
    state is the sequential one bitwise. The pictures go to idle ranks, never to rank 1
    while its conditioner call may arrive; the second request reuses every visual latent."""
    with Group(tmp_path, degree) as ranks:
        facts = ranks.leader.facts["sequence_parallel"]
        assert facts["hosted"] == {"text_encoder": 1}
        assert facts["sparing_ranks"] == [1]
        group, model = ranks.group, ranks.model
        # Proof of overlap without timing: the conditioner's call waits until rank 0 is
        # inside a reference encode, and that encode waits until the call is in flight.
        # Run in sequence, either wait would time out.
        encoding, calling = threading.Event(), threading.Event()
        call_one = group.call_one

        def hosted(rank: int, command: Mapping[str, object]) -> dict[str, Any]:
            assert encoding.wait(60), "the conditioner call ran before any reference encode"
            calling.set()
            return call_one(rank, command)

        def encode(_module: Any, _args: Any) -> None:
            encoding.set()
            assert calling.wait(60), "a reference encode ran without the conditioner call"

        group.call_one = hosted  # type: ignore[method-assign]
        vae = model.pipe.components["video_vae"]
        handle = vae.encoder.register_forward_pre_hook(encode)
        checks, telemetry = _checks(model)
        state = h3.start(model)
        try:
            condition_references(model, h3.TASK, state, checks=checks)
        finally:
            handle.remove()
            group.call_one = call_one  # type: ignore[method-assign]
        _assert_equal(_outputs(state), sequential)
        log = group.spread_log["video_vae.encode_condition"]
        assert log["spare"] is True
        assert "1" not in log["calls"]
        assert sum(log["calls"].values()) == 2
        if degree == 4:
            assert log["calls"] == {"0": 1, "2": 1}
        assert sorted(group.rank_records) == list(range(1, min(degree, 3)))
        again = h3.start(model)
        group.spread_log.clear()
        condition_references(model, h3.TASK, again, checks=checks)
        _assert_equal(_outputs(again), sequential)
        assert "video_vae.encode_condition" not in group.spread_log
        assert _logged(telemetry, "h3 reference latents")[-1] == {"reused": 3, "encoded": 0}
        assert not list((tmp_path / "attempt").rglob("tensor-*.raw"))
    assert not torch.distributed.is_initialized()


def _decode(model: Any, z: Any) -> Any:
    state = h3.start(model)
    state.set("latents", z)
    chunks: list[Any] = []
    with torch.no_grad():
        model.decode_video(h3.TASK, state, on_chunk=chunks.append)
    return torch.cat(chunks)


def _latents(model: Any) -> Any:
    channels = model.pipe.components["video_vae"].config.latent_channels
    # 33 latent frames are 7 temporal clips: rounds over the ranks, the last one partial.
    return torch.randn(1, channels, 33, 2, 3, generator=torch.Generator().manual_seed(5))


@pytest.fixture
def card(monkeypatch: pytest.MonkeyPatch) -> Device:
    """Rank 0's simulated H100 in this process (run 1268's capacity, the turbo overlay held)."""
    simulated = Device(CAPACITY, TURBO)
    h3.simulate(simulated, monkeypatch.setattr)
    return simulated


def _staged(card: Device, measured: tuple[str, ...]) -> Any:
    """Rank 0's own plane over the simulated card, open on run 1268's measured scopes."""
    plane = h3.simulated(card, ("ref2va_dit", "video_vae", "audio_vae"))
    plane.open_attempt("component_staged", ENVELOPE, dict(SCOPES), measured)
    return plane


@pytest.mark.parametrize(
    ("degree", "home_capacity", "measured", "home_decodes"),
    [
        (2, 0, (), True),  # no device plane to spare: the CPU rank takes clips
        (4, CAPACITY, tuple(SCOPES), False),  # an H100: the conditioner leaves no room
        (4, 96_000_000_000, tuple(SCOPES), True),  # a 96 GB card has room beside it
        (4, 96_000_000_000, (), False),  # room, but the decode's peak is not measured
    ],
)
def test_the_home_rank_decodes_clips_only_beside_what_it_holds(
    tmp_path: Path,
    card: Device,
    degree: int,
    home_capacity: int,
    measured: tuple[str, ...],
    home_decodes: bool,
) -> None:
    """Rank 1 is offered clips like every rank. With its hosted conditioner and DiT shard
    resident it takes one only on measured room for the VAE and the attempt's largest
    scratch; otherwise it declines, the clip runs elsewhere, and the frames are the same."""
    expected = _decode(h3.model(), _latents(h3.model()))
    args = ("--home-capacity", str(home_capacity)) if home_capacity else ()
    with Group(tmp_path, degree, *args) as ranks:
        model = ranks.model
        if home_capacity:
            object.__setattr__(model, "_cozy_residency", _staged(card, measured))
        actual = _decode(model, _latents(model))
        assert torch.equal(actual, expected)
        log = ranks.group.spread_log["video_vae._decode_clip"]
        assert log["spare"] is False
        assert sum(log["calls"].values()) == 7
        assert ("1" in log["calls"]) is home_decodes
        assert ("1" in log["declined"]) is not home_decodes
        if not home_decodes:
            assert "declined" in log["declined"]["1"]
            assert set(log["calls"]) == {"0", "2", "3"}
    assert not torch.distributed.is_initialized()


def test_an_h100_home_rank_spares_its_conditioner_and_a_roomier_one_keeps_the_vae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run 1268's numbers. Rank 1 holds the 51.5 GB conditioner and its 21.1 GB DiT shard.
    On an H100 a decode clip would need the 5.6 GB VAE and 5.8 GB of scratch beside them:
    declined, nothing evicted. On a 96 GB card it fits, and the VAE it stages survives the
    next request's conditioner and denoise scopes."""
    for capacity, fits in ((CAPACITY, False), (96_000_000_000, True)):
        card = Device(capacity, TURBO)
        h3.simulate(card, monkeypatch.setattr)
        plane = h3.simulated(card, ("text_encoder", "ref2va_dit"))
        plane.open_attempt("component_staged", ENVELOPE, dict(SCOPES), tuple(SCOPES))
        held = sorted(plane.backend.components)
        with plane.sparing():
            if not fits:
                with pytest.raises(ResidencyRefusal) as refused:
                    plane.admit("decode_video", ("video_vae",))
                assert refused.value.code == "device_spared"
                assert sorted(plane.backend.components) == held
                assert plane.attempt_evictions == 0
                continue
            plane.admit("decode_video", ("video_vae",))
        plane.release("decode_video", ("video_vae",))
        for _ in range(2):
            plane.open_attempt("component_staged", ENVELOPE, dict(SCOPES), tuple(SCOPES))
            for method, components in (
                ("condition_text", ("text_encoder",)),
                ("sample_ref2va_turbo", ("ref2va_dit",)),
            ):
                plane.admit(method, components)
                plane.release(method, components)
            assert plane.attempt_evictions == 0
        assert sorted(plane.backend.components) == ["ref2va_dit", "text_encoder", "video_vae"]
        assert card.allocated <= capacity


@pytest.fixture(autouse=True)
def _threads() -> Iterator[None]:
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)
