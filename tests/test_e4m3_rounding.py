"""The fast encoder must preserve the reference table's exact rounding and bytes."""

import numpy as np
import pytest

from cozy_runtime.derive import microscale


def table_encoding(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=0.0, posinf=448.0, neginf=-448.0
    )
    magnitudes = np.minimum(np.abs(values).astype(np.float64), 448.0)
    index = microscale._nearest_even(microscale.E4M3_MAG, magnitudes)
    result: np.ndarray = index | (np.signbit(values).astype(np.uint8) << 7)
    return result


def test_e4m3_matches_every_source_16bit_value_and_rounding_boundary() -> None:
    words = np.arange(65536, dtype=np.uint32)
    midpoints = ((microscale.E4M3_MAG[:-1] + microscale.E4M3_MAG[1:]) / 2).astype(np.float32)
    edges = np.concatenate(
        [
            midpoints,
            np.nextafter(midpoints, np.float32(-np.inf)),
            np.nextafter(midpoints, np.float32(np.inf)),
        ]
    )
    random_words = np.random.default_rng(20260906).integers(
        0, 2**32, size=1_000_000, dtype=np.uint32
    )
    for values in (
        (words << 16).view(np.float32),
        words.astype(np.uint16).view(np.float16).astype(np.float32),
        edges,
        -edges,
        random_words.view(np.float32),
        np.array([[-0.0, 0.0], [448.0, -448.0]], dtype=np.float32).T,
        np.empty((0, 32), dtype=np.float32),
    ):
        np.testing.assert_array_equal(microscale.e4m3_encode(values), table_encoding(values))


def test_complete_quantization_bytes_scales_and_statistics_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exact binary fractions cover zero rows, large dynamic range and both signs.
    values: np.ndarray = ((np.arange(256 * 128, dtype=np.int32) % 1021) - 510).astype(
        np.float32
    ).reshape(256, 128) / np.float32(32)
    values[0] = 0
    values[1::3] *= np.float32(2**-12)
    values[2::3] *= np.float32(2**8)
    encode = microscale.encode_fp8_rowwise
    actual = encode("weight", values)
    monkeypatch.setattr(
        microscale, "_e4m3_signed", lambda values, _magnitude: table_encoding(values)
    )
    expected = encode("weight", values)
    assert actual == expected


def reference_fp8_rowwise(values: np.ndarray) -> tuple[bytes, bytes]:
    """The pre-optimization encoder, one expression per step, as the byte oracle."""
    scale = np.maximum(np.max(np.abs(values), axis=-1) / np.float32(448), np.float32(1e-12))
    return table_encoding(np.clip(values / scale[:, None], -448, 448)).tobytes(), scale.tobytes()


@pytest.mark.parametrize("shape", [(1280, 1280), (257, 2048), (3, 64)])
def test_encoders_emit_the_reference_bytes_on_weight_like_tensors(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(1303)
    values = (rng.standard_normal(shape, dtype=np.float32) * np.float32(0.03)).astype(np.float16)
    values = values.astype(np.float32)
    values[0] = 0
    values[1, ::3] *= np.float32(2**-14)
    encoded = microscale.encode_fp8_rowwise("weight", values)
    assert (encoded.payload, encoded.companions[0].raw) == reference_fp8_rowwise(values)
