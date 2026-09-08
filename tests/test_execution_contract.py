"""The lane execution contract (cr-109): read, refused typed, bound to resolution, attested.

The torch-free arms run everywhere. The CUDA arm runs `testdata/execution_contract_proof.py`
in a child process against real diffusers 0.40.0 and a real sageattention 2.2.0 build on the
box's own card, and skips naming whichever of those this host lacks.
"""

from __future__ import annotations

import functools
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import canonical, execution_contract
from cozy_runtime.internal.encoding import (
    SPEC_PLAIN,
    SPEC_ROWWISE,
    DeviceFacts,
    Encoded,
    RolePart,
    RuntimeIdentity,
)
from cozy_runtime.internal.encoding.formats import TORCH_DTYPES, RowwiseDequant, Verbatim
from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf
from cozy_runtime.internal.encoding.selection import (
    Capabilities,
    CapabilityRecord,
    implementation_digest,
    launch_providers,
)
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.resolution import ConstructionFacts, PlanRefusal, Variant, resolve

CONTRACT: dict[str, Any] = {
    "device": "sm90",
    "activations": "bf16",
    "class": "quantized",
    "weights": {"route": "encoded_gemm"},
    "attention": {
        "distribution": "sageattention",
        "version": "2.2.0",
        "entry": "sageattn_qk_int8_pv_fp8_cuda_sm90",
        "kwargs": {
            "tensor_layout": "NHD",
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp32",
            "smooth_k": True,
            "is_causal": False,
        },
        "scheme": (
            "qk int8 per-thread (Q64/16, K128/128) K-mean-smoothed; v fp8-e4m3 per-channel; "
            "pv fp32+fp32"
        ),
    },
}


#: The same-class lane's contract (h3a-012): FA3 bf16 baked in the cu130 image, no descale.
FA3_CONTRACT: dict[str, Any] = {
    "device": "sm90",
    "activations": "bf16",
    "class": "same",
    "weights": {"route": "encoded_gemm"},
    "attention": {
        "distribution": "flash-attn3",
        "version": "1",
        "revision": "7cb368cf8278b583132eb72cbf312d54586df2e2",
        "variant": "torch-stable-abi29-cu130-x86_64-linux",
        "entry": "flash_attn_func",
        "kwargs": {
            "causal": False,
            "q_descale": None,
            "k_descale": None,
            "v_descale": None,
            "num_splits": 1,
            "deterministic": False,
        },
        "scheme": (
            "bf16 q/k/v with no descale (the bf16 path, not fp8); fp32 softmax and PV "
            "accumulation; full non-causal attention, no window, no softcap"
        ),
    },
}


def _configs(document: dict[str, Any]) -> dict[str, bytes]:
    return {"transformer": b'{"num_layers":50}', "execution": canonical.write(document)}


def _refusal(run: Any) -> execution_contract.ExecutionContractRefusal:
    with pytest.raises(execution_contract.ExecutionContractRefusal) as refused:
        run()
    return refused.value


def _device(sm: int, kind: str = "cuda") -> DeviceFacts:
    return DeviceFacts(kind=kind, name="card", sm=sm, driver="580", configuration="", index=0)


# ------------------------------------------------------------------------ the contract


def test_the_header_without_an_execution_config_carries_no_contract() -> None:
    assert execution_contract.read(None) is None
    assert execution_contract.read({"transformer": b'{"num_layers":50}'}) is None


def test_the_design_contract_reads_verbatim_and_digests_stably() -> None:
    contract = execution_contract.read(_configs(CONTRACT))
    assert contract is not None
    assert contract.document() == CONTRACT
    assert contract.device == "sm90" and contract.weights.route == "encoded_gemm"
    assert contract.attention.entry == "sageattn_qk_int8_pv_fp8_cuda_sm90"
    assert contract.attention.kwargs["smooth_k"] is True
    again = execution_contract.read(_configs(json.loads(json.dumps(CONTRACT))))
    assert again is not None and again.digest() == contract.digest()
    assert contract.digest().startswith("sha256:")


@pytest.mark.parametrize(
    ("mutate", "names"),
    [
        (lambda d: d["attention"].__setitem__("extra", 1), "extra"),
        (lambda d: d.__setitem__("device", 90), "device"),
        (lambda d: d.__setitem__("device", "hopper"), "device"),
        (lambda d: d.__setitem__("activations", "int8"), "activations"),
        (lambda d: d.__setitem__("class", "faster"), "class"),
        (lambda d: d.pop("class"), "class"),
        (lambda d: d["attention"].__setitem__("revision", "7cb368cf"), "revision"),
        (lambda d: d["weights"].__setitem__("route", "decode_floor"), "weights.route"),
        (lambda d: d["attention"].__setitem__("entry", "a-b"), "attention.entry"),
        (lambda d: d["attention"].__setitem__("version", "latest!"), "attention.version"),
        (lambda d: d["attention"].__setitem__("kwargs", {"x": [1]}), "kwargs"),
        (lambda d: d["attention"].pop("scheme"), "scheme"),
    ],
)
def test_a_malformed_contract_refuses_invalid_naming_the_field(mutate: Any, names: str) -> None:
    document = json.loads(json.dumps(CONTRACT))
    mutate(document)
    refused = _refusal(lambda: execution_contract.read(_configs(document)))
    assert refused.code == "execution_contract_invalid"
    assert names.split(".")[-1] in str(refused)


def test_the_baked_kernel_contract_reads_verbatim_and_pins_its_snapshot() -> None:
    contract = execution_contract.read(_configs(FA3_CONTRACT))
    assert contract is not None
    assert contract.document() == FA3_CONTRACT
    assert contract.class_ == "same"
    assert contract.attention.revision == "7cb368cf8278b583132eb72cbf312d54586df2e2"
    assert contract.attention.kwargs["q_descale"] is None
    assert contract.digest() != execution_contract.read(_configs(CONTRACT)).digest()  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("mutate", "names"),
    [
        (lambda d: d["attention"].__setitem__("revision", "v1"), "revision"),
        (lambda d: d["attention"].pop("revision"), "revision"),
        (lambda d: d["attention"].__setitem__("variant", ""), "variant"),
        (lambda d: d["attention"].pop("variant"), "variant"),
    ],
)
def test_a_baked_kernel_without_its_pin_refuses_invalid(mutate: Any, names: str) -> None:
    document = json.loads(json.dumps(FA3_CONTRACT))
    mutate(document)
    refused = _refusal(lambda: execution_contract.read(_configs(document)))
    assert refused.code == "execution_contract_invalid" and names in str(refused)


def test_a_non_canonical_or_external_config_refuses_invalid() -> None:
    loose = json.dumps(CONTRACT, indent=1).encode()
    refused = _refusal(lambda: execution_contract.read({"execution": loose}))
    assert refused.code == "execution_contract_invalid" and "canonical" in str(refused)
    external = {"execution": {"sha256": "1" * 64, "length": 1, "media_type": "application/json"}}
    assert _refusal(lambda: execution_contract.read(external)).code == "execution_contract_invalid"


# --------------------------------------------------------------------- device + kernel


def test_the_device_class_is_binding_and_refuses_before_any_kernel_is_looked_up() -> None:
    contract = execution_contract.read(_configs(CONTRACT))
    assert contract is not None
    for facts in (_device(89), _device(0, "cpu"), _device(100)):
        refused = _refusal(functools.partial(execution_contract.check_device, contract, facts))
        assert refused.code == "execution_contract_unsupported"
        assert "sm90" in str(refused)
    execution_contract.check_device(contract, _device(90))


def test_an_absent_distribution_refuses_kernel_absent_naming_it() -> None:
    contract = execution_contract.read(_configs(CONTRACT))
    assert contract is not None
    if importlib.util.find_spec("sageattention") is not None:
        pytest.skip("sageattention is installed here; the absent arm needs it absent")
    refused = _refusal(lambda: execution_contract.resolve_kernel(contract))
    assert refused.code == "execution_contract_kernel_absent"
    assert "sageattention==2.2.0" in str(refused) and "not installed" in str(refused)


def test_the_baked_kernel_lane_refuses_off_sm90_and_where_nothing_is_baked() -> None:
    """The local arms of the FA3 lane (sm90-only kernel, RTX 4070 box): the device class
    refuses before any lookup, and off the worker image the snapshot is not baked."""
    contract = execution_contract.read(_configs(FA3_CONTRACT))
    assert contract is not None
    refused = _refusal(functools.partial(execution_contract.check_device, contract, _device(89)))
    assert refused.code == "execution_contract_unsupported" and "sm90" in str(refused)
    if (execution_contract.IMAGE_KERNELS_DIR / "flash-attn3").exists():
        pytest.skip("flash-attn3 is baked here; the absent arm needs it absent")
    refused = _refusal(lambda: execution_contract.resolve_kernel(contract))
    assert refused.code == "execution_contract_kernel_absent"
    assert "flash-attn3@7cb368cf8278/torch-stable-abi29-cu130-x86_64-linux" in str(refused)
    if not execution_contract.KERNELS_ABSENT:
        assert str(execution_contract.IMAGE_KERNELS_DIR / "flash-attn3") in str(refused)


def test_a_wrong_version_or_missing_entry_refuses_kernel_absent() -> None:
    installed = importlib.metadata.version("msgspec")
    document = json.loads(json.dumps(CONTRACT))
    document["attention"].update(distribution="msgspec", version="0.0.1", entry="json.decode")
    contract = execution_contract.read(_configs(document))
    assert contract is not None
    refused = _refusal(lambda: execution_contract.resolve_kernel(contract))
    assert refused.code == "execution_contract_kernel_absent"
    assert "msgspec==0.0.1" in str(refused) and installed in str(refused)

    document["attention"].update(version=installed, entry="json.no_such_entry")
    contract = execution_contract.read(_configs(document))
    assert contract is not None
    refused = _refusal(lambda: execution_contract.resolve_kernel(contract))
    assert refused.code == "execution_contract_kernel_absent"
    assert "does not exist" in str(refused)

    # A real callable that no diffusers attention backend binds (or no diffusers at all).
    document["attention"].update(entry="json.decode")
    contract = execution_contract.read(_configs(document))
    assert contract is not None
    refused = _refusal(lambda: execution_contract.resolve_kernel(contract))
    assert refused.code == "execution_contract_kernel_absent"
    assert "diffusers" in str(refused)


# --------------------------------------------------------------- weights.route binding


class _Row:
    def __init__(self, key: str, dtype: str, shape: tuple[int, ...], encoded: Encoded) -> None:
        self.key, self.component, self.dtype, self.shape, self.encoded = (
            key,
            "dit",
            dtype,
            shape,
            encoded,
        )


def _rows() -> list[_Row]:
    rowwise = Encoded(
        SPEC_ROWWISE,
        "fp8-rowwise/1",
        (RolePart("data", "f8_e4m3fn", (64, 64), 4096), RolePart("scale", "f32", (64,), 256)),
    )
    plain = Encoded(SPEC_PLAIN, "plain/1", (RolePart("value", "bf16", (64,), 128),))
    return [
        _Row("dit.proj.weight", "bf16", (64, 64), rowwise),
        _Row("dit.norm.weight", "bf16", (64,), plain),
    ]


def _record(provider: Any, spec: str) -> CapabilityRecord:
    return CapabilityRecord(
        encoding_spec_id=spec,
        implementation_digest=implementation_digest(provider),
        runtime_release="0.4.1",
        device_predicate="cuda.sm89",
        output_dtype="bfloat16",
        delivery_route=str(provider.route),
        evidence_digest="sha256:" + "0" * 64,
        resident_bits_per_element=16.0 if provider.route != "encoded_gemm" else 8.0,
        kernel_numeric_error=0.0,
    )


def _resolve(
    capabilities: Capabilities,
    *,
    route: str = "",
    encoded_leaves: str = "accept",
    rows: list[_Row] | None = None,
) -> Any:
    return resolve(
        variants=[
            Variant(name="ref", store="store", snapshot="sha256:" + "1" * 64, reference=True)
        ],
        rows_for={"ref": _rows() if rows is None else rows},
        providers=launch_providers(),
        capabilities=capabilities,
        device=_device(89),
        runtime=RuntimeIdentity(
            release="0.4.1", source_digest="sha256:" + "2" * 64, torch_build="2.13.0"
        ),
        construction=ConstructionFacts(
            release="0.4.1", model_class="h3:H3Model", components=("dit",)
        ),
        facts=PlanFacts(),
        encoded_leaves=encoded_leaves,
        dtype_name=lambda name: TORCH_DTYPES.get(str(name), str(name)),
        route=route,
    )


def _floor_only() -> Capabilities:
    """This card qualified the fp8-rowwise DECODE and the plain copy, not the native leaf."""
    return Capabilities(
        [
            _record(RowwiseDequant(SPEC_ROWWISE), SPEC_ROWWISE),
            _record(Verbatim(SPEC_PLAIN), SPEC_PLAIN),
        ]
    )


def test_a_contract_less_lane_still_serves_the_decode_floor_with_its_confession() -> None:
    plan = _resolve(_floor_only())
    assert plan.route_census() == {"decoded_float": 1, "verbatim": 1}
    assert plan.confession.startswith("UNCALIBRATED") and "DECODE FLOOR" in plan.confession


def test_a_contract_lane_refuses_where_the_floor_confession_would_appear() -> None:
    with pytest.raises(PlanRefusal) as refused:
        _resolve(_floor_only(), route="encoded_gemm")
    assert refused.value.code == "execution_contract_unsupported"
    assert "dit.proj.weight" in refused.value.detail and "decoded_float" in refused.value.detail
    assert "weights.route='encoded_gemm'" in refused.value.detail


def test_a_contract_lane_needs_the_package_consent_it_binds() -> None:
    with pytest.raises(PlanRefusal) as refused:
        _resolve(_floor_only(), route="encoded_gemm", encoded_leaves="refuse")
    assert refused.value.code == "execution_contract_unsupported"
    assert "encoded_leaves='refuse'" in refused.value.detail


def test_a_contract_lane_serves_its_bound_route_unmeasured_and_confesses_the_contract() -> None:
    qualified = _floor_only()
    qualified.add(_record(RowwiseNativeLeaf(SPEC_ROWWISE), SPEC_ROWWISE))
    plan = _resolve(qualified, route="encoded_gemm")
    assert plan.route_census() == {"encoded_gemm": 1, "verbatim": 1}
    assert plan.policy == "encoded_gemm"
    assert "weights.route='encoded_gemm'" in plan.confession
    assert [row["verdict"] for row in plan.walk] == ["chosen_uncalibrated"]
    # Without the contract the same records reach the same plan through cr-086's encoded lane;
    # the contract's difference is the floor-only case above, where cr-086 would decode.
    free = _resolve(qualified)
    assert free.digest() == plan.digest() and "ENCODED-GEMM" in free.confession


def test_a_route_no_tensor_stores_bytes_for_refuses_unsupported() -> None:
    plain_only = [row for row in _rows() if row.encoded.encoding == SPEC_PLAIN]
    with pytest.raises(PlanRefusal) as refused:
        _resolve(_floor_only(), route="encoded_gemm", rows=plain_only)
    assert refused.value.code == "execution_contract_unsupported"
    assert "no tensor of this checkpoint" in refused.value.detail
    # A float route binds only the tensors that offer it; the rest keep their own resolution.
    plan = _resolve(_floor_only(), route="verbatim")
    assert plan.route_census() == {"decoded_float": 1, "verbatim": 1}


# ------------------------------------------------------------------------- the CUDA arm


def _cuda_proof_skip() -> str:
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed"
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device"
    for distribution, version in (("diffusers", "0.40.0"), ("sageattention", "2.2.0")):
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return f"{distribution}=={version} is not installed"
        if installed != version:
            return f"{distribution} is {installed}, the proof needs {version}"
    try:
        __import__("sageattention")
    except Exception as exc:
        return f"sageattention does not import: {exc}"
    return ""


def test_the_contract_is_pinned_and_attested_on_a_real_card() -> None:
    why = _cuda_proof_skip()
    if why:
        pytest.skip(why)
    child = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "testdata/execution_contract_proof.py")],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    out = json.loads(child.stdout.splitlines()[-1])
    assert out["unsupported"]["code"] == "execution_contract_unsupported", out["unsupported"]
    assert out["allocated_after_unsupported"] == 0
    assert out["design_backend"] == "_sage_qk_int8_pv_fp8_cuda_sm90"
    assert out["design_live"] == {"is_causal": "is_causal"}
    for arm, refused in out["kernel_absent"].items():
        assert refused["code"] == "execution_contract_kernel_absent", (arm, refused)
    assert "installs 2.2.0" in out["kernel_absent"]["version"]["detail"]
    assert "per_thread" in out["kernel_absent"]["kwarg_default"]["detail"]
    assert "no_such_kwarg" in out["kernel_absent"]["kwarg_unknown"]["detail"]
    assert out["hosts"] == {"fl2va_dit": 3} and out["pinned_processors"] == 3
    green = out["green"]
    assert green["refusal"] == "", green
    assert green["calls"] == green["expected_calls"] == 3 * out["steps"]
    assert green["forwards"] == {"fl2va_dit": out["steps"]}
    assert "qk_int_sv_f8_attn_kernel" in green["kernel_symbol"], green["kernels"]
    assert green["kernel_self_ms"] > 0
    assert out["prepare_facts_after"]["kernel_symbol"] == green["kernel_symbol"]
    assert out["dropped_site"]["refusal"].startswith("execution_contract_unobserved")
    assert "called 2 time(s) where the pinned sites predict 3" in out["dropped_site"]["refusal"]
    assert "called 1 time(s) where the pinned sites predict 0" in out["extra_call"]["refusal"]
    assert "is_causal=True" in out["scheme_mismatch"]["refusal"]
    assert "float16 activations" in out["activations_mismatch"]["refusal"]
    assert out["registry_slot_is_proxy"] is True
