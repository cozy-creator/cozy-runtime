"""Staged admission keeps what fits beside a MEASURED scope (H3 run 1268 on an H100).

The residency plane, its eviction policy and the ledger/plan hand-off are the real code. Only
the device is simulated: a byte counter with the H100's measured allocatable capacity, so the
arithmetic is the pod's own. No claim about kernel scratch follows; the next H100 run is the
proof that the kept set does not OOM (see h3-warm-cache-opportunities.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import pytest

from cozy_runtime.internal import accel
from cozy_runtime.internal.executor_replies import Metrics
from cozy_runtime.internal.fill import StreamingFillBackend
from cozy_runtime.internal.residency import ComponentResidency
from cozy_runtime.internal.worker.ledger import Ledger

#: rank 0 of 4xH100 in run 1268: allocatable + allocated at close, and the turbo LoRA's
#: resident overlay, which a sibling model holds and this plane cannot evict.
CAPACITY = 82_120_343_552
TURBO = 798_073_844
SIZES = {
    "text_encoder": 51_506_192_496,
    "ref2va_dit": 21_105_997_260,
    "fl2va_dit": 21_105_997_260,
    "video_vae": 5_570_955_392,
    "audio_vae": 605_306_340,
}
#: the measured per-scope scratch peaks and whole-attempt envelope from the same run's plan
SCOPES = {
    "condition_text": 2_724_447_232,
    "condition_ref2va_media": 395_359_744,
    "sample_ref2va_turbo": 5_390_094_848,
    "decode_audio": 494_003_200,
    "decode_video": 5_080_392_704,
}
ENVELOPE = 5_814_289_920
REQUEST = (
    ("condition_text", ("text_encoder",)),
    ("condition_ref2va_media", ("video_vae", "audio_vae")),
    ("sample_ref2va_turbo", ("ref2va_dit",)),
    ("decode_audio", ("audio_vae",)),
    ("decode_video", ("video_vae",)),
)


@dataclass
class Device:
    """Allocated bytes against a fixed capacity: what `accel` reports on a real card."""

    capacity: int
    allocated: int = 0

    def allocation(self, *_: Any, **__: Any) -> dict[str, int]:
        return {
            "allocated_bytes": self.allocated,
            "reserved_bytes": self.allocated,
            "driver_free_bytes": self.capacity - self.allocated,
            "driver_total_bytes": self.capacity,
        }


class _Event:
    def query(self) -> bool:
        return True


@dataclass
class Backend:
    """The fill backend's residency surface over the simulated device."""

    device_bytes: Device
    components: dict[str, object] = field(default_factory=dict)
    parked: dict[str, object] = field(default_factory=dict)
    vram_charge: dict[str, int] = field(default_factory=dict)
    component_bytes: dict[str, int] = field(default_factory=lambda: dict(SIZES))
    paging: dict[str, Any] = field(default_factory=dict)
    paging_hooks: list[Any] = field(default_factory=list)
    stage_log: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    resident_budget: int = 0
    staged: list[str] = field(default_factory=list)
    device: Any = field(default_factory=lambda: SimpleNamespace(type="cuda"))

    def stage_bytes(self, name: str, headroom: int = 0) -> int:
        return SIZES[name] + headroom

    def stage(self, name: str) -> dict[str, Any]:
        self.components[name] = self.parked.pop(name)
        self.vram_charge[name] = SIZES[name]
        self.device_bytes.allocated += SIZES[name]
        self.staged.append(name)
        return {"component": name}

    def evict(self, name: str) -> int:
        self.parked[name] = self.components.pop(name)
        charged = self.vram_charge.pop(name)
        self.device_bytes.allocated -= charged
        return charged


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> Device:
    simulated = Device(CAPACITY, TURBO)
    monkeypatch.setattr(accel, "allocation", simulated.allocation)
    monkeypatch.setattr(accel, "allocated", lambda *_: simulated.allocated)
    monkeypatch.setattr(accel, "peak_allocated", lambda *_: simulated.allocated)
    monkeypatch.setattr(accel, "reset_peak", lambda *_: None)
    monkeypatch.setattr(accel, "release_cached", lambda *_: None)
    monkeypatch.setattr(accel, "synchronize", lambda *_: None)
    monkeypatch.setattr(accel, "completion_event", lambda *_: _Event())
    return simulated


def plane(device: Device, resident: tuple[str, ...]) -> tuple[ComponentResidency, Backend]:
    backend = Backend(device)
    for name in SIZES:
        backend.parked[name] = object()
    for name in resident:
        backend.stage(name)
    backend.staged.clear()
    residency = ComponentResidency(
        backend=cast(StreamingFillBackend, backend), torch=None, placement="component_staged"
    )
    return residency, backend


def serve(residency: ComponentResidency, backend: Backend, measured: bool) -> dict[str, Any]:
    residency.open_attempt(
        "component_staged", ENVELOPE, dict(SCOPES), tuple(SCOPES) if measured else ()
    )
    backend.staged.clear()
    for method, components in REQUEST:
        residency.admit(method, components)
        residency.release(method, components)
    return {
        "staged": list(backend.staged),
        "evictions": residency.attempt_evictions,
        "resident": sorted(backend.components),
    }


def test_h3_keeps_its_dit_and_restages_only_the_text_encoder(device: Device) -> None:
    residency, backend = plane(device, ("video_vae",))
    # Unmeasured: every scope frees the whole card, exactly run 1268's confession.
    first = serve(residency, backend, measured=False)
    assert first["staged"] == [
        "text_encoder",
        "video_vae",
        "audio_vae",
        "ref2va_dit",
        "audio_vae",
        "video_vae",
    ]
    assert first["evictions"] == 6
    # Measured: the 21 GB DiT stays; the text encoder cannot (85.0 GB > 82.1 GB).
    serve(residency, backend, measured=True)
    for _ in range(3):
        steady = serve(residency, backend, measured=True)
        assert steady["staged"] == ["text_encoder", "video_vae"]
        assert steady["evictions"] == 2
        assert steady["resident"] == ["audio_vae", "ref2va_dit", "video_vae"]
    assert device.allocated <= CAPACITY


def test_a_stale_component_is_the_first_victim(device: Device) -> None:
    # Both DiTs warmed; a ref2va request must give up fl2va_dit, not its own set.
    residency, backend = plane(device, ("fl2va_dit", "ref2va_dit", "video_vae", "audio_vae"))
    residency.open_attempt("component_staged", 0)
    residency.admit("warm_ref2va", ("ref2va_dit",))
    residency.release("warm_ref2va", ("ref2va_dit",))
    steady = [serve(residency, backend, measured=True) for _ in range(3)][-1]
    assert "fl2va_dit" not in steady["resident"]
    assert "ref2va_dit" not in steady["staged"]


def test_an_unmeasured_scope_still_frees_everything(device: Device) -> None:
    residency, backend = plane(device, ("ref2va_dit", "video_vae", "audio_vae"))
    residency.open_attempt("component_staged", ENVELOPE, dict(SCOPES), ("decode_video",))
    residency.admit("decode_audio", ("audio_vae",))
    assert sorted(backend.components) == ["audio_vae"]


CELL = "assets=2,frames=243"


def _served(ledger: Ledger, cell: str, free: int) -> None:
    ledger.device_free = free
    ledger.observe_attempt(
        msgspec.convert({"activation_peaks": dict(SCOPES)}, Metrics), cell=cell, succeeded=True
    )


def test_only_an_exact_cell_measurement_may_keep_components() -> None:
    ledger = Ledger(worker_pid=1)
    ledger.begin_generation(7)
    _served(ledger, CELL, free=60 << 30)
    assert ledger.measured_scopes(CELL) == frozenset(SCOPES)
    # Another shape's peak does not describe this one.
    assert ledger.measured_scopes("assets=12,frames=363") == frozenset()
    # A property of the model and the shape: it survives an executor rebuild.
    ledger.begin_generation(8)
    assert ledger.measured_scopes(CELL) == frozenset(SCOPES)


def test_a_hosted_text_encoder_leaves_rank_0_nothing_to_move(device: Device) -> None:
    # The follower keeps the conditioner (`mirror.hosting_plan`); rank 0 frees it once, and
    # its fully hosted scope never clears the card even before anything is measured.
    residency, backend = plane(device, ("text_encoder", "ref2va_dit", "video_vae", "audio_vae"))
    residency.host("text_encoder")
    assert "text_encoder" not in backend.components
    residency.open_attempt("component_staged", ENVELOPE, dict(SCOPES))
    residency.admit("condition_text", ("text_encoder",))
    residency.release("condition_text", ("text_encoder",))
    assert residency.attempt_evictions == 0 and backend.staged == []
    for _ in range(3):
        served = serve(residency, backend, measured=True)
    assert served["resident"] == ["audio_vae", "ref2va_dit", "video_vae"]
    assert served["staged"] == [] and served["evictions"] == 0
    document = residency.document()
    assert document["hosted"] == ["text_encoder"]
    assert "text_encoder" not in document["evicted"]


#: GPU 1 of run 1560 (4xH100, H3 hosting its text encoder): 85_017_493_504 B less the vacated
#: Qwen executor's context (842 MiB) and this process's bytes outside the allocator (78.28 GiB
#: in use, 75.35 GiB reserved), so 7.57 GB stay allocatable beside the encoder and the DiT.
GPU1 = 80_981_451_079
#: What segment 2 (345 frames + a 56-frame context, ~134k rows) had allocated beyond those
#: weights when it asked for 452 MiB more and the card had 58.5 MiB left.
SEGMENT_2_AT_FAILURE = 7_648_016_643


def test_run_1560_gpu_1_gives_up_the_text_encoder_only_when_the_segment_is_measured_elsewhere(
    device: Device,
) -> None:
    device.capacity = GPU1
    residency, backend = plane(device, ("text_encoder", "ref2va_dit"))
    # The follower homing the encoder runs staged. Segment 2 shared segment 1's cell, so the
    # 6 s peak counted as measured and the encoder stayed beside a 12 s denoise.
    residency.open_attempt("component_staged", ENVELOPE, dict(SCOPES), tuple(SCOPES))
    residency.admit("sample_ref2va_turbo", ("ref2va_dit",))
    residency.release("sample_ref2va_turbo", ("ref2va_dit",))
    assert "text_encoder" in backend.components
    assert device.capacity - device.allocated < SEGMENT_2_AT_FAILURE
    # At its own, unmeasured cell the denoise frees everything else first.
    residency.open_attempt("component_staged", ENVELOPE, dict(SCOPES))
    residency.admit("sample_ref2va_turbo", ("ref2va_dit",))
    assert sorted(backend.components) == ["ref2va_dit"]
    assert device.capacity - device.allocated > 7 * SEGMENT_2_AT_FAILURE
