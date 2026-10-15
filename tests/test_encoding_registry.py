"""Runtime providers are reviewed against TensorFS 0.3 identities, never aliases."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.derive import quantization
from cozy_runtime.internal.encoding import (
    SPEC_INT8_CONVROT256,
    SPEC_NVFP4,
    SPEC_PLAIN,
    SPEC_ROWWISE,
    SPEC_ROWWISE_KEEPDIM,
    SPEC_SCALED_SCALAR,
    SPEC_SCALED_SCALAR_WEIGHT_ONLY,
    SPEC_VECTORS,
    Capabilities,
    DeviceFacts,
    Encoded,
    EncodingRefusal,
    RolePart,
    Selector,
    launch_providers,
    selection,
)

VECTORS = Path(__file__).resolve().parents[1] / "src/cozy_runtime/internal/encoding/vectors"
RETIRED = {
    "sha256:71409d585d82c513f364ac730d2f573f0261828c4700ce653333a758d15e14bd",
    "sha256:8c86b26daec5bcd287401d9873b70aa3594855f97ac4d01cf75cc2bafd5d50b5",
    "sha256:91ddc2a73ba2778b344afc5e78b4addee9174a015cfff02f17cfbb629ad5ef78",
    "sha256:c3f4ff643d6ce6cce6843d2cddd756b1e9fe91b8b6294e51a48b3339c42e7a7f",
    "sha256:5fb5ffc9e387cd1e54fb9877058294f6965fdc0e504ab8799ac310d90f3fe489",
    "sha256:796a50ed4d63e0b7cfc70a8cd797a5fe796bba64cb3143de4d35f059653f5d80",
}
#: `mxfp8/1`, removed from this Runtime; TensorFS may still seed it.
MXFP8 = "sha256:7e9b1ad8f2e5ddd236a4d4303042d632a96eadef4f25d44eb0fb63124cec7cfd"


def test_every_provider_is_reviewed_against_the_exact_tensorfs_03_seed() -> None:
    seeds = {digest: alias for alias, digest in tensorfs.seed_digests()}
    reviewed = {
        SPEC_PLAIN,
        SPEC_ROWWISE,
        SPEC_ROWWISE_KEEPDIM,
        SPEC_SCALED_SCALAR,
        SPEC_SCALED_SCALAR_WEIGHT_ONLY,
        SPEC_INT8_CONVROT256,
        SPEC_NVFP4,
    }
    # TensorFS before 0.9.2 carries no int8-convrot256/1; that one provider is then not served.
    assert reviewed - {SPEC_INT8_CONVROT256} <= set(seeds)
    assert RETIRED.isdisjoint(seeds)
    assert quantization.PLAIN_SPEC == SPEC_PLAIN
    assert set(quantization._ARTIFACT_SPECS.values()) == {
        SPEC_ROWWISE,
        SPEC_INT8_CONVROT256,
        SPEC_NVFP4,
    }

    providers = launch_providers()
    assert set(providers) == reviewed & set(seeds)
    assert all(
        provider.reviewed_specs <= set(seeds)
        for implementations in providers.values()
        for provider in implementations
    )
    # nvfp4's three other role sets (input_scale, pre_quant_scale) have no mined vectors.
    unsupported = {
        digest
        for digest, alias in seeds.items()
        if alias == "svdq-microscale/1" or (alias == "nvfp4-w4a4/1" and digest != SPEC_NVFP4)
    }
    assert unsupported and unsupported.isdisjoint(providers)


def test_reviewed_roles_and_vector_bytes_did_not_move() -> None:
    providers = launch_providers()
    expected_roles = {
        SPEC_PLAIN: ({"value"}, set()),
        SPEC_ROWWISE: ({"data", "scale"}, set()),
        SPEC_ROWWISE_KEEPDIM: ({"data", "scale"}, set()),
        SPEC_SCALED_SCALAR: ({"data", "weight_scale"}, {"input_scale"}),
        SPEC_SCALED_SCALAR_WEIGHT_ONLY: ({"data", "weight_scale"}, {"input_scale"}),
        SPEC_INT8_CONVROT256: ({"data", "scale"}, set()),
        SPEC_NVFP4: ({"weight", "weight_scale", "weight_scale_2"}, set()),
    }
    for digest, implementations in providers.items():
        required, optional = expected_roles[digest]
        assert all(provider.required == required for provider in implementations)
        assert all(provider.optional == optional for provider in implementations)
    for _digest, (filename, expected) in SPEC_VECTORS.items():
        measured = "sha256:" + hashlib.sha256((VECTORS / filename).read_bytes()).hexdigest()
        assert measured == expected


def test_an_mxfp8_tensor_refuses_at_bind() -> None:
    providers = launch_providers()
    assert MXFP8 not in providers
    selector = Selector(providers, Capabilities(), DeviceFacts("cpu", "CPU", 0, "", "", 0))
    parts = (RolePart("data", "f8_e4m3fn", (2, 64), 128), RolePart("scale", "u8", (2, 2), 4))
    with pytest.raises(EncodingRefusal) as refused:
        selector.candidates("w", Encoded(MXFP8, "mxfp8/1", parts), "bf16", (2, 64))
    assert refused.value.code in {"spec_unreviewed", "unknown_encoding"}


def test_a_spec_the_installed_tensorfs_lacks_is_skipped_with_one_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    older = selection.registry_digests() - {SPEC_INT8_CONVROT256}
    monkeypatch.setattr(selection, "registry_digests", lambda: older)
    monkeypatch.setattr(selection, "_ABSENT", set())
    for _ in range(2):
        providers = launch_providers()
        assert SPEC_INT8_CONVROT256 not in providers and SPEC_ROWWISE in providers
    warned = capsys.readouterr().err.splitlines()
    assert len(warned) == 1 and SPEC_INT8_CONVROT256 in warned[0]  # once, not per call
