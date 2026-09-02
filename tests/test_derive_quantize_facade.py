"""cr-076: the `derive.plan`/`derive.quantize` facade at the dozen-line bar.

Real tensors through the REAL kernel: each arm runs a registered job via
`invoke_with_fakes`, so the production decode -> inject -> WeightsSink -> transaction
path executes end to end. The only stand-in is the TensorFS host behind the
`WeightsHostTransaction` protocol (the same seam `test_runtime_weights_exchange` stands
in for), holding real numpy tensor bytes and committing content-addressed receipts.

* behaviour identity — the facade's three lanes commit byte-for-byte the transactions
  the existing helper composition (the packages-repo `sdxl_three_lane` body) commits
  over the same source;
* both doors reach the same tensors — `quantize(model, ...)` and
  `quantize(weights.structure(model), ...)` commit identical content;
* the tier-1 tripwire refuses typed (`quantization_tripwire`) and commits nothing;
* a replayed lane reports the prior receipt and writes nothing;
* cr-082 (#695): a real fp32 safetensors source normalizes to exact RNE bf16 bytes with
  round-trip stats recorded and the plain lane marked canonical; an fp16 source passes
  through untouched; a keep-listed fp32 component stays fp32.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from cozy_runtime.author import (
    App,
    Context,
    Telemetry,
    UnsupportedInput,
    WeightsOutput,
    WeightsReceipt,
    WeightsSink,
    WeightsSource,
    WeightsSourcePart,
    WeightsSourceTensor,
    WeightsTarget,
    WeightsTransaction,
)
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author._weights import WeightsCommit, WeightsHostTransaction
from cozy_runtime.author.fakes import invoke_with_fakes
from cozy_runtime.derive import QuantizePlan, QuantizeResult, plan, quantize
from cozy_runtime.derive import safetensors_io as st
from cozy_runtime.derive.quantization import (
    ArtifactQuantizationPlan,
    ArtifactQuantizationRequest,
    Bf16Plan,
    QuantizationSource,
    bf16_targets,
    inherited_configs,
    prepare_bf16,
    prepare_source_quantization,
    quantization_additions,
    quantize_component_into,
    write_bf16,
)

MANIFEST = "sha256:" + "ab" * 32
LANES = (("bf16", None), ("fp8", "fp8-rowwise/1"), ("mxfp8", "mxfp8/1"))


# ------------------------------------------------------------------ the TensorFS stand-in


def _projection(commit: WeightsCommit) -> dict[str, object]:
    return {
        "output": commit.output_slot,
        "sources": dict(commit.sources),
        "targets": {
            component: {
                "source": target.source,
                "source_component": target.source_component,
                "drop": list(target.drop),
                "add": {
                    key: {
                        "logical_dtype": tensor.logical_dtype,
                        "shape": list(tensor.shape),
                        "encoding": tensor.encoding,
                        "parts": {
                            role: [part.dtype, list(part.shape)]
                            for role, part in sorted(tensor.parts.items())
                        },
                    }
                    for key, tensor in sorted(target.add.items())
                },
            }
            for component, target in sorted(commit.targets.items())
        },
        "configs": {
            name: [config.source, config.source_config]
            for name, config in sorted(commit.configs.items())
        },
        "order": [list(row) for row in commit.order],
        "max_new_bytes": commit.max_new_bytes,
    }


@dataclass
class _Committed:
    declaration: dict[str, object]
    parts: dict[tuple[str, str, str], bytes]
    receipt: WeightsReceipt


@dataclass
class _Store:
    """Real tensor bytes behind the host protocol; commits are content-addressed."""

    raw: dict[tuple[str, str], bytes] = field(default_factory=dict)
    tensors: list[WeightsSourceTensor] = field(default_factory=list)
    configs: tuple[str, ...] = ("model_index.json", "vae/config.json")
    committed: dict[str, _Committed] = field(default_factory=dict)
    replay: bool = False
    fenced: int = 0

    def put(self, component: str, key: str, dtype: str, shape: tuple[int, ...], raw: bytes) -> None:
        self.raw[(component, key)] = raw
        self.tensors.append(
            WeightsSourceTensor(
                component, key, dtype, shape, (WeightsSourcePart("value", dtype, shape),)
            )
        )

    def structure(self, manifest: str) -> WeightsSource:
        assert manifest == MANIFEST
        return WeightsSource(configs=self.configs, tensors=tuple(self.tensors))

    def open(self, commit: WeightsCommit) -> WeightsHostTransaction:
        if self.replay:
            prior = self.committed[commit.output_slot]
            assert _projection(commit) == prior.declaration, "replay under a changed declaration"
            return _Replayed(prior.receipt)
        return _Transaction(self, commit)


class _Replayed:
    replayed = True

    def __init__(self, receipt: WeightsReceipt) -> None:
        self.receipt: WeightsReceipt | None = receipt

    def source_read_into(
        self, source: str, component: str, key: str, role: str, offset: int, into: memoryview
    ) -> None:
        raise AssertionError("a replayed transaction reads nothing")

    def add_part(self, component: str, key: str, role: str, data: object) -> None:
        raise AssertionError("a replayed transaction writes nothing")

    def add_config(self, name: str, data: object) -> None:
        raise AssertionError("a replayed transaction writes nothing")

    def commit(self) -> WeightsReceipt:
        raise AssertionError("a replayed transaction is not committed again")

    def fence(self) -> None:
        pass


class _Transaction:
    replayed = False
    receipt: WeightsReceipt | None = None

    def __init__(self, store: _Store, declared: WeightsCommit) -> None:
        self.store = store
        self.declared = declared
        self.parts: dict[tuple[str, str, str], bytes] = {}

    def source_read_into(
        self, source: str, component: str, key: str, role: str, offset: int, into: memoryview
    ) -> None:
        assert role == "value"
        raw = self.store.raw[(component, key)]
        chunk = raw[offset : offset + len(into)]
        assert len(chunk) == len(into)
        into[:] = chunk

    def add_part(self, component: str, key: str, role: str, data: object) -> None:
        assert isinstance(data, (bytes, bytearray, memoryview))
        self.parts[(component, key, role)] = bytes(data)

    def add_config(self, name: str, data: object) -> None:
        raise AssertionError("these lanes only inherit configs")

    def commit(self) -> WeightsReceipt:
        declaration = _projection(self.declared)
        body = {
            "declaration": declaration,
            "parts": {
                f"{component}/{key}#{role}": hashlib.sha256(data).hexdigest()
                for (component, key, role), data in sorted(self.parts.items())
            },
        }
        canonical = json.dumps(body, sort_keys=True).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        receipt = WeightsReceipt(
            output_slot=self.declared.output_slot,
            weights_transaction_id="wtx-" + digest[:16],
            tensorfs_receipt_digest="sha256:" + digest,
            tensorfs_receipt=canonical,
            checkpoint_evidence_canonical_bytes=b"{}",
        )
        self.store.committed[self.declared.output_slot] = _Committed(
            declaration, dict(self.parts), receipt
        )
        return receipt

    def fence(self) -> None:
        self.store.fenced += 1


def _sdxl_like_store(*, poison: bool = False) -> _Store:
    rng = np.random.default_rng(76)
    store = _Store()

    def f16(shape: tuple[int, ...]) -> bytes:
        return (rng.standard_normal(shape).astype(np.float32) * 0.05).astype("<f2").tobytes()

    q = rng.standard_normal((8, 64)).astype(np.float32) * 0.05
    if poison:
        q[3, 7] = np.nan
    store.put("unet", "down.0.attn.to_q.weight", "f16", (8, 64), q.astype("<f2").tobytes())
    store.put("unet", "down.0.attn.to_q.bias", "f16", (8,), f16((8,)))
    proj = rng.standard_normal((4, 32)).astype(np.float32) * 0.05
    store.put("unet", "mid.proj.weight", "bf16", (4, 32), st.from_f32(proj, "BF16"))
    store.put("unet", "time_embed.weight", "f16", (6, 48), f16((6, 48)))
    norm = rng.standard_normal((8,)).astype(np.float32)
    store.put("unet", "mid.norm.weight", "f32", (8,), norm.astype("<f4").tobytes())
    store.put("unet", "position_ids", "i64", (1, 16), bytes(8 * 16))
    store.put("text_encoder", "embed.weight", "f16", (16, 32), f16((16, 32)))
    return store


# ------------------------------------------------------------------------------- the jobs

app = App()

OUTPUT_BYTES = 1 << 30
THREE = tuple(WeightsOutput(name, max_new_bytes=OUTPUT_BYTES) for name, _ in LANES)


class ThreeLane(msgspec.Struct):
    bf16_receipt_digest: str
    fp8_receipt_digest: str
    mxfp8_receipt_digest: str
    replayed_outputs: int
    source_bytes_read_this_run: int
    encoded_keys_this_run: int


def _receipt(transaction: WeightsTransaction) -> WeightsReceipt:
    value = transaction.receipt if transaction.replayed else transaction.commit()
    assert value is not None
    return value


def _sdxl_targets(
    bf16: Bf16Plan, *, encoding: str | None, quantization: ArtifactQuantizationPlan
) -> dict[str, WeightsTarget]:
    """Verbatim helper composition from the packages-repo `sdxl_three_lane` consumer."""
    if encoding is None:
        return bf16_targets(bf16, source="source")
    selected = {(tensor.component, tensor.key) for tensor in quantization.tensors}
    targets = bf16_targets(bf16, source="source", excluded=selected)
    additions = quantization_additions(encoding, quantization, "unet", target_logical_dtype="bf16")
    current = targets["unet"]
    combined = {**current.add, **additions}
    targets["unet"] = WeightsTarget(
        source="source", source_component="unet", drop=tuple(sorted(combined)), add=combined
    )
    return targets


@app.job(name="legacy-three-lane", weights=THREE)
def legacy_three_lane(
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    source: QuantizationSource,
    weights: WeightsSink,
    tel: Telemetry,
) -> ThreeLane:
    """The moved packages-repo body, verbatim: the identity baseline."""
    structure = weights.structure(source)
    bf16 = prepare_bf16(structure)
    quantization = prepare_source_quantization(structure, components=("unet",))
    selected = {(tensor.component, tensor.key) for tensor in quantization.tensors}
    receipts: dict[str, WeightsReceipt] = {}
    source_bytes = 0
    encoded = 0
    for output, encoding in LANES:
        with weights.open(
            output,
            sources={"source": source},
            targets=_sdxl_targets(bf16, encoding=encoding, quantization=quantization),
            configs=inherited_configs(bf16, source="source"),
            order=bf16.order,
        ) as transaction:
            if transaction.replayed:
                receipts[output] = _receipt(transaction)
                continue
            normalization = write_bf16(
                transaction,
                ctx,
                tel,
                plan=bf16,
                source="source",
                excluded=selected if encoding is not None else None,
            )
            source_bytes += normalization.source_bytes_read
            if encoding is not None:
                stats = quantize_component_into(
                    transaction,
                    ctx,
                    payload,
                    tel,
                    encoding=encoding,
                    plan=quantization,
                    component="unet",
                    source="source",
                    source_component="unet",
                    target_component="unet",
                )
                source_bytes += stats.source_bytes_read
                encoded += stats.encoded_keys
            receipts[output] = transaction.commit()
    return ThreeLane(
        receipts["bf16"].tensorfs_receipt_digest,
        receipts["fp8"].tensorfs_receipt_digest,
        receipts["mxfp8"].tensorfs_receipt_digest,
        sum(receipt.replayed for receipt in receipts.values()),
        source_bytes,
        encoded,
    )


@app.job(name="facade-three-lane", weights=THREE)
def facade_three_lane(
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    source: QuantizationSource,
    weights: WeightsSink,
    tel: Telemetry,
) -> ThreeLane:
    """The same three lanes at the cr-073 dozen-line bar."""
    threshold = payload.max_relative_frobenius
    lanes = {
        output: plan(
            ("unet",) if encoding else (),
            encoding,
            max_relative_frobenius=threshold if encoding else None,
        )
        for output, encoding in LANES
    }
    results = {
        output: quantize(source, spec, sink=weights, ctx=ctx, tel=tel, output=output)
        for output, spec in lanes.items()
    }
    return ThreeLane(
        results["bf16"].tensorfs_receipt_digest,
        results["fp8"].tensorfs_receipt_digest,
        results["mxfp8"].tensorfs_receipt_digest,
        sum(result.replayed for result in results.values()),
        sum(result.source_bytes_read_this_run for result in results.values()),
        sum(result.encoded_keys for result in results.values() if not result.replayed),
    )


@app.job(name="facade-model", weights=(WeightsOutput("model", max_new_bytes=OUTPUT_BYTES),))
def facade_model(
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    source: QuantizationSource,
    weights: WeightsSink,
    tel: Telemetry,
) -> QuantizeResult:
    spec = plan(("unet",), "mxfp8/1", max_relative_frobenius=payload.max_relative_frobenius)
    return quantize(source, spec, sink=weights, ctx=ctx, tel=tel)


class PlainLaneRequest(msgspec.Struct):
    keep: tuple[str, ...] = ()


@app.job(name="facade-plain", weights=(WeightsOutput("model", max_new_bytes=OUTPUT_BYTES),))
def facade_plain(
    ctx: Context,
    payload: PlainLaneRequest,
    source: QuantizationSource,
    weights: WeightsSink,
    tel: Telemetry,
) -> QuantizeResult:
    """The canonical ingest lane (#695): fp32→bf16, fp16 stays, keep list honored."""
    return quantize(source, plan(keep=payload.keep), sink=weights, ctx=ctx, tel=tel)


@app.job(name="facade-structure", weights=(WeightsOutput("model", max_new_bytes=OUTPUT_BYTES),))
def facade_structure(
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    source: QuantizationSource,
    weights: WeightsSink,
    tel: Telemetry,
) -> QuantizeResult:
    spec = plan(("unet",), "mxfp8/1", max_relative_frobenius=payload.max_relative_frobenius)
    return quantize(weights.structure(source), spec, sink=weights, ctx=ctx, tel=tel)


def _run(name: str, store: _Store, tmp_path: Path, wire: dict[str, object]) -> object:
    return invoke_with_fakes(
        app.get(name),
        wire,
        request_id=name,
        spool=tmp_path / name,
        models={"source": _derive_model(QuantizationSource, MANIFEST)},
        weights=store.open,
        weights_source_structure=store.structure,
    ).result


# ------------------------------------------------------------------------------- the arms


def test_the_facade_lanes_match_the_helper_composition_byte_for_byte(tmp_path: Path) -> None:
    """Identity evidence: same source, same declarations, same bytes, same receipts."""
    wire: dict[str, object] = {"max_relative_frobenius": 0.25}
    legacy_store = _sdxl_like_store()
    facade_store = _sdxl_like_store()

    legacy = _run("legacy-three-lane", legacy_store, tmp_path, wire)
    facade = _run("facade-three-lane", facade_store, tmp_path, wire)

    assert isinstance(legacy, ThreeLane) and isinstance(facade, ThreeLane)
    assert facade == legacy
    for output, _ in LANES:
        ours, theirs = facade_store.committed[output], legacy_store.committed[output]
        assert ours.declaration == theirs.declaration
        assert ours.parts == theirs.parts
        assert ours.receipt == theirs.receipt
    # The lanes did real work: 2 encoded keys per quantized lane, and distinct contents.
    assert facade.encoded_keys_this_run == 4
    assert facade.source_bytes_read_this_run > 0
    assert len({c.receipt.tensorfs_receipt_digest for c in facade_store.committed.values()}) == 3


def test_both_doors_reach_the_same_tensors(tmp_path: Path) -> None:
    """`quantize(model, ...)` and `quantize(structure, ...)` commit identical content."""
    by_model_store = _sdxl_like_store()
    by_structure_store = _sdxl_like_store()

    by_model = _run("facade-model", by_model_store, tmp_path, {})
    by_structure = _run("facade-structure", by_structure_store, tmp_path, {})

    assert isinstance(by_model, QuantizeResult) and by_model == by_structure
    assert by_model_store.committed["model"] == by_structure_store.committed["model"]
    assert by_model.encoded_keys == 2
    assert by_model.converted_keys == 1
    assert by_model.inherited_keys == 4
    assert by_model.kept_keys == 0
    assert not by_model.canonical
    assert 0.0 < by_model.normalization_worst_relative_frobenius <= 2.0**-8
    assert 0.0 < by_model.worst_relative_frobenius < 0.25


def test_the_tier1_tripwire_refuses_typed_and_commits_nothing(tmp_path: Path) -> None:
    store = _sdxl_like_store()
    with pytest.raises(UnsupportedInput) as tripped:
        _run("facade-model", store, tmp_path, {"max_relative_frobenius": 1e-9})
    assert tripped.value.code == "quantization_tripwire"
    assert not store.committed and store.fenced == 1

    poisoned = _sdxl_like_store(poison=True)
    with pytest.raises(UnsupportedInput) as refused:
        _run("facade-model", poisoned, tmp_path, {})
    assert refused.value.code == "unsupported_input"
    assert not poisoned.committed

    invalid: tuple[Callable[[], QuantizePlan], ...] = (
        lambda: plan(("unet",)),
        lambda: plan((), "mxfp8/1"),
        lambda: plan(("unet",), "int3/0"),
        lambda: plan(("unet",), "mxfp8/1", max_relative_frobenius=-1.0),
        lambda: plan(("unet",), "mxfp8/1", keep=("unet",)),
        lambda: plan(keep=("vae", "vae")),
    )
    for bad in invalid:
        with pytest.raises(UnsupportedInput) as refusal:
            bad()
        assert refusal.value.code == "quantization_plan"


def test_a_replayed_lane_reports_the_prior_receipt_and_writes_nothing(tmp_path: Path) -> None:
    store = _sdxl_like_store()
    first = _run("facade-model", store, tmp_path / "first", {})
    assert isinstance(first, QuantizeResult)

    store.replay = True
    again = _run("facade-model", store, tmp_path / "again", {})
    assert isinstance(again, QuantizeResult)
    assert again.replayed
    assert again.tensorfs_receipt_digest == first.tensorfs_receipt_digest
    assert again.weights_transaction_id == first.weights_transaction_id
    assert (again.source_bytes_read_this_run, again.new_bytes_written_this_run) == (0, 0)
    assert (again.encoded_keys, again.converted_keys, again.inherited_keys) == (2, 1, 4)
    assert (again.kept_keys, again.normalization_worst_relative_frobenius) == (0, 0.0)
    assert not again.canonical


# ------------------------------------------------------------------- cr-082 (#695) arms


def _declared_target(committed: _Committed, component: str) -> dict[str, Any]:
    targets = committed.declaration["targets"]
    assert isinstance(targets, dict)
    target = targets[component]
    assert isinstance(target, dict)
    return target


def test_a_real_fp32_safetensors_source_normalizes_to_bf16_with_recorded_stats(
    tmp_path: Path,
) -> None:
    """The ingest cut is a recorded derivation: exact RNE bf16 bytes, stats, canonical."""
    rng = np.random.default_rng(82)
    attn = rng.standard_normal((16, 32)).astype(np.float32) * 0.02
    norm = rng.standard_normal((16,)).astype(np.float32)
    writer = st.Writer()
    writer.add("blocks.0.attn.weight", "F32", [16, 32], attn.tobytes())
    writer.add("blocks.0.norm.weight", "F32", [16], norm.tobytes())
    path = tmp_path / "source-fp32.safetensors"
    writer.write(path)

    store = _Store()
    header, base = st.read_header(path)
    for key, row in sorted(header.items()):
        raw_source = st.read_raw(path, header, base, key)
        store.put("dit", key, row["dtype"].lower(), tuple(row["shape"]), raw_source)

    result = _run("facade-plain", store, tmp_path, {})
    assert isinstance(result, QuantizeResult)
    assert result.canonical
    assert (result.converted_keys, result.kept_keys, result.encoded_keys) == (2, 0, 0)
    assert 0.0 < result.normalization_worst_relative_frobenius <= 2.0**-8
    committed = store.committed["model"]
    for key, values in (("blocks.0.attn.weight", attn), ("blocks.0.norm.weight", norm)):
        declared = _declared_target(committed, "dit")["add"][key]
        assert declared["logical_dtype"] == "bf16"
        assert declared["parts"] == {"value": ["bf16", list(values.shape)]}
        raw = committed.parts[("dit", key, "value")]
        assert raw == st.from_f32(values, "BF16")
        round_trip = st.to_f32(raw, "BF16", list(values.shape))
        relative = np.linalg.norm(values - round_trip) / np.linalg.norm(values)
        assert 0.0 < relative <= 2.0**-8


def test_an_fp16_source_passes_through_untouched(tmp_path: Path) -> None:
    """fp16→bf16 buys nothing (#695): same bytes, three mantissa bits lost — refused."""
    rng = np.random.default_rng(83)
    store = _Store()
    raw_weight = (rng.standard_normal((8, 32)).astype(np.float32) * 0.05).astype("<f2").tobytes()
    raw_bias = (rng.standard_normal((8,)).astype(np.float32) * 0.05).astype("<f2").tobytes()
    store.put("unet", "a.weight", "f16", (8, 32), raw_weight)
    store.put("unet", "a.bias", "f16", (8,), raw_bias)

    result = _run("facade-plain", store, tmp_path, {})
    assert isinstance(result, QuantizeResult)
    assert result.canonical
    assert (result.converted_keys, result.kept_keys, result.inherited_keys) == (0, 0, 2)
    assert result.source_bytes_read_this_run == 0
    committed = store.committed["model"]
    assert committed.parts == {}
    target = _declared_target(committed, "unet")
    assert target["drop"] == [] and target["add"] == {}


def test_a_keep_listed_fp32_component_stays_fp32(tmp_path: Path) -> None:
    """The ingest config's keep list (H3's fp32 audio VAE is the canonical entry)."""
    rng = np.random.default_rng(84)
    vae = rng.standard_normal((4, 4)).astype(np.float32)

    def _store() -> _Store:
        store = _Store()
        store.put(
            "dit",
            "blocks.0.w.weight",
            "f32",
            (8, 32),
            (rng.standard_normal((8, 32)).astype(np.float32) * 0.02).tobytes(),
        )
        store.put("audio_vae", "decoder.conv.weight", "f32", (4, 4), vae.tobytes())
        return store

    store = _store()
    result = _run("facade-plain", store, tmp_path, {"keep": ["audio_vae"]})
    assert isinstance(result, QuantizeResult)
    assert (result.converted_keys, result.kept_keys, result.inherited_keys) == (1, 1, 1)
    committed = store.committed["model"]
    assert ("audio_vae", "decoder.conv.weight", "value") not in committed.parts
    kept = _declared_target(committed, "audio_vae")
    assert kept["drop"] == [] and kept["add"] == {}
    converted = _declared_target(committed, "dit")["add"]["blocks.0.w.weight"]
    assert converted["logical_dtype"] == "bf16"

    with pytest.raises(UnsupportedInput):
        _run("facade-plain", _store(), tmp_path / "ghost", {"keep": ["ghost"]})
