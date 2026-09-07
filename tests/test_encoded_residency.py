"""Native leaf residency preserves its staging peak after logical modules are replaced."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal.executor import residence_ceiling
from cozy_runtime.internal.fill import ComponentMemory, complete_device_envelope, price_envelope


def test_native_component_peaks_do_not_sum_parked_payloads() -> None:
    # Exact verified header77c18fdb...16cc62 and H100 req-17aeedad3088e21d8fd7e753.
    # The 313 rowwise weights per DiT hold their FP8 data plus f32 row scales.
    logical, native = 143_380_369_572, 40_157_014_016
    authorized, allocatable = 84_460_896_256, 84_263_763_968
    rows = []
    for name in ("fl2va_dit", "ref2va_dit"):
        rows.extend(
            [
                (name, 40_131_624_960, native // 2, "encoded_gemm", native // 2, 0),
                (name, 295_072_768, 295_072_768, "verbatim", 0, 0),
            ]
        )
    rows.extend(
        (name, size, size, "verbatim", 0, 0)
        for name, size in (
            ("text_encoder", 51_506_191_840),
            ("video_vae", 10_415_475_936),
            ("audio_vae", 605_306_340),
        )
    )
    envelope = complete_device_envelope(price_envelope(rows, 512 << 20), 103_274_585_285)
    assert envelope["destination_bytes"] == logical
    assert envelope["native_bytes"] == native
    assert allocatable + native == 124_420_777_984  # observed old refusal
    overlap = envelope["fill_overhead_bytes"]
    budget = residence_ceiling(
        destinations=envelope["base_bytes"],
        authorized=authorized - overlap,
        allocatable=allocatable - overlap,
    )
    assert budget + overlap <= min(authorized, allocatable)
    assert overlap == native // 2
    assert budget == 64_185_256_960
    assert envelope["components"]["fl2va_dit"]["destinations"] == 40_426_697_728
    assert (
        residence_ceiling(destinations=logical, authorized=overlap - 1, allocatable=overlap - 1)
        < overlap
    )


def test_settled_measurement_does_not_erase_restore_peak_or_derived_buffers() -> None:
    memory = ComponentMemory(
        destinations=1200, native_destinations=1000, native=510, native_resident=510, decode=90
    )
    assert memory.settled == 710  # includes 200 bytes not replaced
    assert memory.peak == 1800
    memory.resident = 710
    assert memory.admission(100) == 1800
    assert memory.admission(2000) == 2710
    envelope = complete_device_envelope(
        price_envelope(
            [
                ("a", 1000, 510, "encoded_gemm", 510, 0),
            ],
            100,
        ),
        1200,
    )
    assert envelope["derived_destination_bytes"] == 200
    assert envelope["base_bytes"] + envelope["fill_overhead_bytes"] == 1710


def test_expanding_native_roles_and_swizzle_temporaries_are_charged() -> None:
    memory = ComponentMemory(
        destinations=8, native_destinations=8, native=20, native_resident=24, native_scratch=12
    )
    assert memory.base == 24 and memory.peak == 40
    assert memory.overhead == 16
    assert memory.admission(0) == 40


def test_native_tensorfs_encoded_component_lifecycle(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("actual native FP8 component lifecycle requires CUDA")
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "testdata/encoded_residency_proof.py"),
            str(tmp_path / "encoded"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(child.stdout.splitlines()[-1])
    assert result["phase"] == "passed", child.stdout + child.stderr
    assert result["staging_peak_bytes"] == result["staging_bound_bytes"]
    assert result["failure"] == "poisoned"


def test_mxfp8_padded_scale_storage_has_one_provider_price() -> None:
    from cozy_runtime.internal.encoding import SPEC_MXFP8, RolePart
    from cozy_runtime.internal.encoding.leaves import MicroScaledNativeLeaf

    provider = MicroScaledNativeLeaf(SPEC_MXFP8)
    parts = {
        "data": RolePart("data", "f8_e4m3fn", (2, 1056), 2112),
        "scale": RolePart("scale", "u8", (2, 33), 66),
    }
    # cuBLAS scales occupy complete128x4 tiles: 2x33 becomes128x36.
    assert provider.resident_bytes(parts) == 2112 + 128 * 36
    assert provider.fill_scratch_bytes(parts) == 3 * 128 * 36
    assert provider.resident_bits_per_element(parts, 2112) == 8 * (2112 + 128 * 36) / 2112
