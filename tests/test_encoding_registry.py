"""Runtime providers are reviewed against TensorFS 0.3 identities, never aliases."""

from __future__ import annotations

import hashlib
from pathlib import Path

import tensorfs

from cozy_runtime.derive import quantization
from cozy_runtime.internal.encoding import (
    SPEC_MXFP8,
    SPEC_PLAIN,
    SPEC_ROWWISE,
    SPEC_ROWWISE_KEEPDIM,
    SPEC_SCALED_SCALAR,
    SPEC_SCALED_SCALAR_WEIGHT_ONLY,
    SPEC_VECTORS,
    RolePart,
    launch_providers,
)
from cozy_runtime.internal.encoding.leaves import MicroScaledNativeLeaf

VECTORS = Path(__file__).resolve().parents[1] / "src/cozy_runtime/internal/encoding/vectors"
RETIRED = {
    "sha256:71409d585d82c513f364ac730d2f573f0261828c4700ce653333a758d15e14bd",
    "sha256:8c86b26daec5bcd287401d9873b70aa3594855f97ac4d01cf75cc2bafd5d50b5",
    "sha256:91ddc2a73ba2778b344afc5e78b4addee9174a015cfff02f17cfbb629ad5ef78",
    "sha256:c3f4ff643d6ce6cce6843d2cddd756b1e9fe91b8b6294e51a48b3339c42e7a7f",
    "sha256:5fb5ffc9e387cd1e54fb9877058294f6965fdc0e504ab8799ac310d90f3fe489",
    "sha256:796a50ed4d63e0b7cfc70a8cd797a5fe796bba64cb3143de4d35f059653f5d80",
}


def test_every_provider_is_reviewed_against_the_exact_tensorfs_03_seed() -> None:
    seeds = {digest: alias for alias, digest in tensorfs.seed_digests()}
    reviewed = {
        SPEC_PLAIN,
        SPEC_ROWWISE,
        SPEC_ROWWISE_KEEPDIM,
        SPEC_SCALED_SCALAR,
        SPEC_SCALED_SCALAR_WEIGHT_ONLY,
        SPEC_MXFP8,
    }
    assert reviewed <= set(seeds)
    assert RETIRED.isdisjoint(seeds)
    assert quantization.PLAIN_SPEC == SPEC_PLAIN
    assert set(quantization._ARTIFACT_SPECS.values()) == {SPEC_ROWWISE, SPEC_MXFP8}

    providers = launch_providers()
    assert set(providers) == reviewed
    assert all(
        provider.reviewed_specs <= set(seeds)
        for implementations in providers.values()
        for provider in implementations
    )
    unsupported = {
        digest for digest, alias in seeds.items() if alias in {"svdq-microscale/1", "nvfp4-w4a4/1"}
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
        SPEC_MXFP8: ({"data", "scale"}, set()),
    }
    for digest, implementations in providers.items():
        required, optional = expected_roles[digest]
        assert all(provider.required == required for provider in implementations)
        assert all(provider.optional == optional for provider in implementations)
    for _digest, (filename, expected) in SPEC_VECTORS.items():
        measured = "sha256:" + hashlib.sha256((VECTORS / filename).read_bytes()).hexdigest()
        assert measured == expected


def test_mxfp8_padded_scale_storage_has_one_provider_price() -> None:
    provider = MicroScaledNativeLeaf(SPEC_MXFP8)
    parts = {
        "data": RolePart("data", "f8_e4m3fn", (2, 1056), 2112),
        "scale": RolePart("scale", "u8", (2, 33), 66),
    }
    # cuBLAS scales occupy complete 128x4 tiles: 2x33 becomes 128x36.
    assert provider.resident_bytes(parts) == 2112 + 128 * 36
    assert provider.fill_scratch_bytes(parts) == 3 * 128 * 36
    assert provider.resident_bits_per_element(parts, 2112) == 8 * (2112 + 128 * 36) / 2112
