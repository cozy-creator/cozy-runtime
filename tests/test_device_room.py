"""Device memory is arbitrated across every executor on the card (shelly runs 1458/1459).

On shelly an mxfp8 SDXL load needed 7.65 GB to stage its unet (the bf16 destinations beside
the 2.97 GB encoded payload) while two idle executors of earlier runs held the card: the
worker priced the load at its stored bytes, found room, and the new executor, able to evict
only its own components, refused `device_shortfall … with [] resident`. The refusal then
poisoned the binding, so the retry failed `prepare refused: every binding`.

Every arm drives the REAL worker, trampoline, executor processes and fill plane on the real
card (`test_shared_lane`'s harness). The tenant here is FP8-encoded: staging its unet holds
the bf16 destinations beside the encoded payload, about three times the stored bytes the
worker prices a load at. A `Neighbour` process sizes the card; it is running work, which
nothing may evict.
"""

from __future__ import annotations

import dataclasses
import io
import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import tensorfs
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.derive.microscale import encode_fp8_rowwise
from cozy_runtime.internal.encoding import SPEC_PLAIN, SPEC_ROWWISE
from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_shared_lane import (
    NO_TENANT,
    Neighbour,
    TenantStore,
    TwoTenantOwner,
    _drive,
    _turn,
    _workspace,
)

needs_tenant = pytest.mark.skipif(bool(NO_TENANT), reason=NO_TENANT or "")

TENANT = Path(__file__).resolve().parent.parent / "corpus" / "tenant"


@dataclasses.dataclass(frozen=True, slots=True)
class Fp8Tenant:
    store: TenantStore
    #: the unet's staging peak (bf16 destinations beside the encoded payload) plus the rest
    peak: int


def write_fp8_tenant(root: Path, *, blocks: int, hidden: int, ff: int) -> Fp8Tenant:
    """`cozytest/tenant@1` with every unet linear stored FP8-rowwise. Blocks repeat one
    matrix per shape, so the store holds each once while the card holds every block."""
    rng = np.random.default_rng(1458)
    linear = {
        shape: encode_fp8_rowwise("w", rng.standard_normal(shape, dtype=np.float32) * 0.05)
        for shape in ((hidden, hidden), (hidden, ff))
    }
    plain: dict[tuple[str, str], np.ndarray] = {}
    encoded: dict[tuple[str, str], tuple[int, int]] = {}
    for block in range(blocks):
        prefix = f"blocks.{block}."
        encoded[("unet", prefix + "attn1.to_q.weight")] = (hidden, hidden)
        encoded[("unet", prefix + "attn1.to_out.0.weight")] = (hidden, hidden)
        encoded[("unet", prefix + "ff.net.0.proj.weight")] = (hidden, ff)
        plain[("unet", prefix + "attn1.norm_q.weight")] = np.ones(64, dtype=np.float32)
        plain[("unet", prefix + "norm1.weight")] = np.ones(hidden, dtype=np.float32)
    plain[("text_encoder", "embeddings.weight")] = rng.standard_normal((1024, 1024), np.float32)
    plain[("text_encoder", "final_norm.weight")] = np.ones(256, dtype=np.float32)
    plain[("vae", "decoder.conv_in.weight")] = rng.standard_normal((128, 128), np.float32)
    plain[("vae", "decoder.conv_out.weight")] = rng.standard_normal((128, 128), np.float32)
    f32 = {"final_norm.weight", "decoder.conv_in.weight", "decoder.conv_out.weight"}

    def dtype(key: str) -> str:
        return "f32" if key in f32 else "bf16"

    targets: dict[str, dict[str, Tensor]] = {"unet": {}, "text_encoder": {}, "vae": {}}
    for (component, key), shape in encoded.items():
        targets[component][key] = Tensor(
            "bf16",
            shape,
            SPEC_ROWWISE,
            {"data": Part("f8_e4m3fn", shape), "scale": Part("f32", (shape[0],))},
        )
    for (component, key), values in plain.items():
        kind = dtype(key)
        targets[component][key] = Tensor(
            kind, values.shape, SPEC_PLAIN, {"value": Part(kind, values.shape)}
        )
    order = tuple((component, key) for component, rows in targets.items() for key in rows)
    config = {
        "blocks": blocks,
        "hidden": hidden,
        "ff": ff,
        "norm_q": 64,
        "text_embed": 1024,
        "text_norm": 256,
        "vae": 128,
    }
    store = tensorfs.Store.ensure(str(root / "store"))
    logical = sum(h * w * 2 for h, w in encoded.values())
    native = sum(h * w + h * 4 for h, w in encoded.values())
    rest = sum(
        values.size * (4 if dtype(key) == "f32" else 2) for (_c, key), values in plain.items()
    )
    transaction = store.begin_derived(
        weights_transaction_id("device-room", "fixture", "sha256:" + "1" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={name: Target(add=rows) for name, rows in targets.items()},
            configs={"unet": NativeConfig("add")},
            order=order,
        ).native_arguments(2 * (native + rest) + (1 << 20)),
        work_fingerprint="sha256:" + "2" * 64,
    )
    for (component, key), (height, width) in encoded.items():
        shaped = linear[(height, width)]
        transaction.add_part(component, key, "data", io.BytesIO(shaped.payload))
        transaction.add_part(component, key, "scale", io.BytesIO(shaped.companions[0].raw))
    for (component, key), values in plain.items():
        body = values.astype(np.float32).tobytes() if dtype(key) == "f32" else _bf16(values)
        transaction.add_part(component, key, "value", io.BytesIO(body))
    transaction.add_config("unet", io.BytesIO(json.dumps(config, sort_keys=True).encode()))
    receipt = transaction.commit()
    manifest = "sha256:" + receipt["manifest"]["sha256"]
    length = int(receipt["manifest"]["length"])
    operation = store.begin_operation("op-tenant", "cozytest", "tenant")
    operation.hold_manifest(manifest, length)
    operation.commit_release(None, "1", "bf16", manifest, length)
    stored = sum(int(row["length"]) for row in store.walk_cozytensors(manifest))
    return Fp8Tenant(
        store=TenantStore(root=root / "store", weight_bytes=stored, config=config),
        peak=logical + native + rest,
    )


def _bf16(values: np.ndarray) -> bytes:
    """Round-to-nearest-even bf16, as the upper half of each f32."""
    bits = values.astype(np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16).tobytes()


def _package(root: Path) -> Path:
    """`corpus/tenant` that accepts encoded leaves and warms through its one scope, as
    SDXL's `warm` does: the load stages every component before the placement serves."""
    source = (TENANT / "tiny_tenant.py").read_text()
    anchor = '    @uses_components("text_encoder", "unet", "vae")\n    def touch('
    assert source.count(anchor) == 1
    source = source.replace(
        "class TenantModel(Model[TenantPipeline]):",
        'class TenantModel(Model[TenantPipeline], encoded_leaves="accept"):',
    ).replace(
        anchor, "    def warm(self, ctx: Context) -> None:\n        self.touch(0)\n\n" + anchor
    )
    project = root / "tenant_fp8"
    project.mkdir(parents=True)
    (project / "tenant_fp8.py").write_text(source)
    table = (TENANT / "package.toml").read_text().replace("tiny_tenant:app", "tenant_fp8:app")
    (project / "package.toml").write_text(table)
    return project


class SizedOwner(TwoTenantOwner):
    """Sizes the card once every executor is up (`hold`, before the first dispatch) and
    ends that running work at the first refusal (`release`), which the owner retries. A
    retry whose placement has gone OFFLINE will never be served: the run ends there."""

    hold: Callable[[], None] | None = None
    release: Callable[[], None] | None = None

    def _on_observed_state(self, report: pb.ObservedWorkerState) -> None:
        waiting = self.pending[0].placement_id if self.refusals and self.pending else ""
        for status in report.placements:
            if (
                status.placement_id == waiting
                and self.current is None
                and (status.serving == pb.ServingState.SERVING_STATE_OFFLINE)
            ):
                faults = [f"{f.reason}: {f.detail}"[:300] for f in (*report.faults, *status.faults)]
                self.ended, self.error = "worker_error", f"{waiting} is offline: {faults}"
                self.done.set()
                self.host.close()
                return
        super()._on_observed_state(report)

    def _next(self) -> None:
        hold, self.hold = self.hold, None
        if hold is not None:
            hold()
        super()._next()

    def _on_outcome(self, outcome: pb.AttemptOutcome) -> None:
        body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        release, self.release = self.release, None
        if int(body.get("status", 0)) == pb.OutcomeStatus.OUTCOME_STATUS_REFUSED and release:
            release()
        else:
            self.release = release
        super()._on_outcome(outcome)


@pytest.fixture(scope="module")
def fp8(tmp_path_factory: pytest.TempPathFactory) -> Fp8Tenant:
    if NO_TENANT:
        pytest.skip(NO_TENANT)
    return write_fp8_tenant(tmp_path_factory.mktemp("fp8"), blocks=24, hidden=2048, ff=8192)


def _room(notes: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(co-tenant, for-whom) per release the worker made at an executor's own request."""
    pattern = re.compile(r"'([^']+)' \(epoch \d+ pid \d+\) .* for '([^']+)' on its executor needs")
    return [
        (match.group(1), match.group(2))
        for kind, step in notes
        if kind == "residency" and (match := pattern.search(step))
    ]


def _short_of(fp8: Fp8Tenant) -> int:
    """Driver-free bytes at which a load's pre-check passes (twice the stored bytes: two
    entrypoints bind the construction) but staging the unet beside the rest does not."""
    assert 2 * fp8.store.weight_bytes + (256 << 20) < fp8.peak, fp8
    return (2 * fp8.store.weight_bytes + fp8.peak) // 2


@needs_tenant
def test_an_executor_short_of_memory_gets_room_from_idle_tenants(
    fp8: Fp8Tenant, tmp_path: Path
) -> None:
    """P and Q load and go idle. Running work then leaves R's load room for its stored
    bytes but not for staging its unet, the bf16 destinations beside the encoded payload.
    R's executor asks the worker, which frees idle tenants' memory LRU-first (allocator
    caches before weights) instead of R refusing; nothing is respawned."""
    project = _package(tmp_path)
    running: list[Neighbour] = []
    turns = [_turn("plc-p", "p-1", 1), _turn("plc-q", "q-1", 2), _turn("plc-r", "r-1", 3)]
    turns[1].after = lambda: running.append(Neighbour(_short_of(fp8)))
    try:
        with _workspace() as root:
            owner, _worker, notes = _drive(
                fp8.store, turns, root, package=project, owner_type=SizedOwner
            )
    finally:
        for neighbour in running:
            neighbour.close()
    assert running and running[0].free_after < fp8.peak, running
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    assert owner.refusals == [], owner.refusals
    assert [turn.outcome["status"] for turn in owner.turns] == [1, 1, 1], owner.turns
    assert {victim for victim, whom in _room(notes) if whom == "plc-r"} <= {"plc-p", "plc-q"}
    assert any(whom == "plc-r" for _victim, whom in _room(notes)), _room(notes)
    for turn in owner.turns:
        assert {row["executor_epoch"] for row in turn.rows.values()} == {1}, turn.rows


@needs_tenant
def test_a_capacity_refusal_is_not_inherited_by_the_retry(fp8: Fp8Tenant, tmp_path: Path) -> None:
    """Running work holds the card, so R's load is refused on capacity. Once that work
    ends, the retry prepares R afresh and serves: the refusal was that attempt's alone."""
    project = _package(tmp_path)
    running: list[Neighbour] = []

    def release() -> None:
        for neighbour in running:
            neighbour.close()

    owner_type = type(
        "Owner",
        (SizedOwner,),
        {
            "hold": lambda self: running.append(Neighbour(_short_of(fp8))),
            "release": lambda self: release(),
        },
    )
    try:
        with _workspace() as root:
            owner, _worker, notes = _drive(
                fp8.store, [_turn("plc-r", "r-1", 1)], root, package=project, owner_type=owner_type
            )
    finally:
        release()
    assert len(owner.refusals) == 1 and "device_shortfall" in owner.refusals[0], owner.refusals
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    assert [turn.outcome["status"] for turn in owner.turns] == [1], owner.turns
    assert not any("prepare refused: every binding" in step for _kind, step in notes)


#: SDXL's shape: one declared set per stage, and a warm that walks all three.
_SCOPES = """
    @uses_components("text_encoder")
    def encode(self, seed: int) -> float:
        return self._read("text_encoder", seed)

    @uses_components("unet")
    def denoise(self, seed: int) -> float:
        return self._read("unet", seed)

    @uses_components("vae")
    def decode(self, seed: int) -> float:
        return self._read("vae", seed)

    def _read(self, name: str, seed: int) -> float:
        import torch

        with torch.inference_mode():
            total = torch.zeros((), dtype=torch.float32, device="cuda")
            for parameter in self.pipe.components[name].parameters():
                value = parameter.flatten()[seed % parameter.numel()].float()
                total = total + torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return float(torch.tanh(total))

    def warm(self, ctx: Context) -> None:
        self.encode(0) + self.denoise(0) + self.decode(0)

"""


def _staged_package(root: Path) -> Path:
    """`corpus/tenant` as SDXL is shaped: encoded leaves accepted, encode/denoise/decode
    scopes, a warm through each, and a text encoder holding a non-persistent buffer (CLIP's
    `position_ids`) that no checkpoint stores."""
    source = (TENANT / "tiny_tenant.py").read_text()
    anchor = '    @uses_components("text_encoder", "unet", "vae")\n    def touch('
    served = "    checksum = model.touch(view._seed)\n"
    buffer = "                self.final_norm = Weight(norm, torch.float32)\n"
    assert source.count(anchor) == 1 and source.count(buffer) == 1
    source = (
        source.replace(
            "class TenantModel(Model[TenantPipeline]):",
            'class TenantModel(Model[TenantPipeline], encoded_leaves="accept"):',
        )
        .replace(anchor, _SCOPES + anchor)
        .replace(served, "    checksum = model.encode(1) + model.denoise(2) + model.decode(3)\n", 1)
        .replace(
            buffer,
            buffer + '                self.register_buffer("position_ids", torch.arange(77), '
            "persistent=False)\n",
        )
    )
    project = root / "tenant_staged"
    project.mkdir(parents=True)
    (project / "tenant_staged.py").write_text(source)
    table = (TENANT / "package.toml").read_text().replace("tiny_tenant:app", "tenant_staged:app")
    (project / "package.toml").write_text(table)
    return project


@needs_tenant
def test_models_that_fit_together_serve_all_resident(tmp_path: Path) -> None:
    """Dimitri run 1507: an fp8 SDXL beside an idle bf16 one on a 96 GB card served on the
    staged rung. Its stored bytes are below its logical ones, so the fill ceiling missed the
    text encoders' derived buffers and parked the last component, and a parked component
    denied `all_resident` for the generation's life. Two such tenants that fit together
    serve every attempt all-resident, with nothing staged or evicted."""
    fp8 = write_fp8_tenant(tmp_path / "fp8", blocks=8, hidden=2048, ff=8192)
    project = _staged_package(tmp_path)
    turns = [
        _turn("plc-p", "p-1", 1),
        _turn("plc-q", "q-1", 2),
        _turn("plc-p", "p-2", 3),
        _turn("plc-q", "q-2", 4),
    ]
    with _workspace() as root:
        owner, _worker, notes = _drive(fp8.store, turns, root, package=project)
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    assert [turn.outcome["status"] for turn in owner.turns] == [1, 1, 1, 1], owner.turns
    for turn in owner.turns:
        assert turn.rows[turn.placement_id]["walk"]["chosen"] == "all_resident", turn.rows
    built = [step for kind, step in notes if kind == "boot" and " parked=" in step]
    assert len(built) == 2 and all("parked=[]" in step for step in built), built
    for turn in owner.turns:
        # `all_resident` frees nothing, and nothing was absent to stage.
        row, walk = turn.rows[turn.placement_id], turn.rows[turn.placement_id]["walk"]
        assert walk["rungs"][0]["absent_bytes"] == 0, (turn.request.request_id, walk)
        assert set(row["resident"]) == {"text_encoder", "unet", "vae"}, row
        assert row["evicted"] == {}, row
