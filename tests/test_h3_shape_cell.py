"""H3's shape cell names what sizes its memory: clip length and reference count.

Staged admission (#756) keeps components beside a scope only when that scope's peak was
measured at the request's exact cell. H3's cell used to be `-` for every length and reference
count, so a first 15 s request after 10 s ones could run on a 10 s peak. This drives the real
`prepare` path over an H3-shaped entrypoint (bounded `duration_s` with a frames table, an
`Assets[Mixed]` parameter, a preflight) and hands the cells to the real ledger.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Annotated, Any

import msgspec
import pytest
from PIL import Image

from cozy_runtime.author import (
    App,
    AssetLimits,
    Assets,
    ConformanceError,
    Mixed,
    Preflight,
    Shape,
    describe,
    prepare,
)
from cozy_runtime.author._assets import GrantedInput
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import shape_cell

#: Stands in for H3's `official.frames_for` table (whole seconds onto its frame grid).
FRAMES = {s: 24 * s + 5 for s in range(5, 16)}
DurationSeconds = Annotated[int, msgspec.Meta(ge=5, le=15), Shape(frames=FRAMES)]
References = Annotated[Assets[Mixed], AssetLimits(images=9, videos=3, audio=3, total=12)]


class TurboInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    seed: int | None = None
    duration_s: DurationSeconds = 5


class Facts(msgspec.Struct, frozen=True):
    images: int
    total: int


class Output(msgspec.Struct):
    ok: bool


def preflight(payload: TurboInput, assets: References) -> Facts:
    kinds = [assets.info(index).kind for index in range(len(assets))]
    return Facts(kinds.count("image"), len(kinds))


app = App()


@app.entrypoint(preflight=preflight)
def ref2va_turbo(payload: TurboInput, assets: References, facts: Preflight[Facts]) -> Output:
    raise AssertionError("prepare only")


def _cell(tmp_path: Path, *, seconds: int, references: int, prompt: str = "a garden") -> str:
    image = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 90, 0)).save(image, format="PNG")
    raw = image.getvalue()
    path = tmp_path / "reference.png"
    path.write_bytes(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    rows = {
        f"assets.{index}.asset": GrantedInput(
            input_id=f"assets.{index}.asset",
            local=path,
            media_type="image/png",
            digest=digest,
            length=len(raw),
            order=index,
        )
        for index in range(references)
    }
    wire: dict[str, Any] = {
        "prompt": prompt,
        "seed": 41827,
        "duration_s": seconds,
        "assets": [{"asset": digest} for _ in range(references)],
    }
    return shape_cell(prepare(app.get("ref2va_turbo"), wire, input_metadata=rows).features.values)


def test_length_and_reference_count_are_the_cell(tmp_path: Path) -> None:
    describe(app)
    ten = _cell(tmp_path, seconds=10, references=2)
    assert ten == f"assets=2,frames={FRAMES[10]}"
    assert _cell(tmp_path, seconds=15, references=2) == f"assets=2,frames={FRAMES[15]}"
    assert _cell(tmp_path, seconds=10, references=3) == f"assets=3,frames={FRAMES[10]}"
    # Prompt and seed size nothing: the same cell.
    assert _cell(tmp_path, seconds=10, references=2, prompt="another prompt") == ten


def test_a_longer_first_request_is_unmeasured_so_staging_frees_the_card(tmp_path: Path) -> None:
    """The #756 hand-off: a 10 s serve's peaks never let a first 15 s request keep neighbours."""
    ten = _cell(tmp_path, seconds=10, references=2)
    fifteen = _cell(tmp_path, seconds=15, references=2)
    ledger = Ledger(worker_pid=1)
    ledger.begin_generation(1)
    ledger.device_free = 60 << 30
    peaks = {"condition_text": 2 << 30, "sample_ref2va_turbo": 5 << 30}
    ledger.observe_attempt({"activation_peaks": peaks}, cell=ten, succeeded=True)
    assert ledger.measured_scopes(ten) == frozenset(peaks)
    assert ledger.measured_scopes(fifteen) == frozenset()
    assert ledger.measured_scopes(_cell(tmp_path, seconds=10, references=5)) == frozenset()


class Open(msgspec.Struct):
    prompt: str
    duration_s: Annotated[int, msgspec.Meta(ge=5), Shape(frames=FRAMES)] = 5


class Short(msgspec.Struct):
    prompt: str
    duration_s: Annotated[
        int, msgspec.Meta(ge=5, le=15), Shape(frames={s: f for s, f in FRAMES.items() if s < 15})
    ] = 5


def test_a_table_over_an_open_int_is_refused_at_build() -> None:
    local = App()

    @local.entrypoint
    def render(payload: Open) -> Output:
        raise AssertionError("describe only")

    with pytest.raises(ConformanceError, match="open value domain"):
        describe(local)


def test_a_table_missing_a_bounded_value_is_refused_at_build() -> None:
    local = App()

    @local.entrypoint
    def render(payload: Short) -> Output:
        raise AssertionError("describe only")

    with pytest.raises(ConformanceError, match="no entry for 15"):
        describe(local)
