"""CPU arithmetic check using verified imported LoRA parts and a generated rowwise base."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cozy_runtime.internal.encoding.formats import SPEC_ROWWISE
from cozy_runtime.internal.lora_merge import MergeAdapter, merge_rowwise_fp8


def digest(value: Any) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    output = args.evidence / "prepared-merge"
    output.mkdir(exist_ok=True)
    torch.set_num_threads(2)  # Only this standalone measurement process.
    prefix = "transformer_blocks.0.attn.to_k"
    adapters, provenance = [], []
    for label, header, strength in (
        ("spatial", "spatial.candidate.json", 0.3),
        ("wushu", "wushu.header.json", 0.7),
    ):
        document = json.loads((args.evidence / header).read_text())
        rows = {row[0]: row for _, component in document["components"] for row in component}
        factors, parts = [], []
        for role in ("A", "B"):
            row = rows[f"{prefix}.lora_{role}.weight"]
            _, dtype, shape, body = row[4][0]
            assert len(body["segments"]) == 1
            expected, length = body["segments"][0]
            path = output / f"{label}-{role}.bin"
            if not path.exists():
                subprocess.run(
                    [
                        "tfs",
                        "get",
                        str(args.store),
                        expected.removeprefix("sha256:"),
                        "--out",
                        str(path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            raw = path.read_bytes()
            assert len(raw) == length
            assert hashlib.sha256(raw).hexdigest() == expected.removeprefix("sha256:")
            factor = np.frombuffer(raw, dtype="<f2" if dtype == "f16" else "<f4").reshape(shape)
            factors.append(torch.from_numpy(factor.copy()))
            parts.append({"key": row[0], "dtype": dtype, "shape": shape, "sha256": expected})
        a, b = factors
        selected = [0, 13, 127, 2047, b.shape[0] - 1]
        b = b[selected].contiguous()
        alpha = None
        if prefix + ".alpha" in rows:
            raw_alpha = bytes.fromhex(rows[prefix + ".alpha"][4][0][3]["inline_hex"])
            alpha = float(np.frombuffer(raw_alpha, dtype="<f4")[0])
        adapters.append(MergeAdapter(a, b, strength, alpha))
        provenance.append(
            {"label": label, "parts": parts, "rows": selected, "strength": strength, "alpha": alpha}
        )

    width = adapters[0].a.shape[1]
    base = torch.linspace(-2, 2, 5 * width).reshape(5, width).to(torch.float8_e4m3fn)
    scales = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05])
    inputs = [base, scales, *(factor for adapter in adapters for factor in (adapter.a, adapter.b))]
    before = [digest(value) for value in inputs]
    reference = base.double() * scales.double()[:, None]
    for adapter in adapters:
        alpha = adapter.a.shape[0] if adapter.alpha is None else adapter.alpha
        reference += (
            adapter.strength
            * (alpha / adapter.a.shape[0])
            * (adapter.b.double() @ adapter.a.double())
        )
    started = time.perf_counter()
    iterator = merge_rowwise_fp8(
        base, scales, adapters, encoding_spec=SPEC_ROWWISE, tile_rows=2, max_scratch_bytes=8 << 20
    )
    assert iterator is not None
    tiles = list(iterator)
    seconds = time.perf_counter() - started
    actual = (
        torch.cat([tile.data for tile in tiles]).double()
        * torch.cat([tile.scale for tile in tiles]).double()[:, None]
    )
    difference = actual - reference
    update = reference - base.double() * scales.double()[:, None]
    after = [digest(value) for value in inputs]
    assert before == after and bool(torch.isfinite(actual).all())
    source = Path(__file__).resolve().parents[2] / "src/cozy_runtime/internal/lora_merge.py"
    result = {
        "scope": (
            "Actual immutable canonical factors on five selected output rows; "
            "generated rowwise base; no H3 execution or full-weight quality claim"
        ),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "factors": provenance,
        "shape": list(base.shape),
        "tile_starts": [tile.start for tile in tiles],
        "scratch_budget": 8 << 20,
        "seconds": seconds,
        "relative_l2_vs_fp64_dense": float(
            torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)
        ),
        "max_abs_vs_fp64_dense": float(difference.abs().max()),
        "update_l2": float(torch.linalg.vector_norm(update)),
        "error_l2": float(torch.linalg.vector_norm(difference)),
        "quantization_squared_error": sum(tile.error.squared_error for tile in tiles),
        "merged_squared_norm": sum(tile.error.squared_reference for tile in tiles),
        "input_sha256_before": before,
        "input_sha256_after": after,
    }
    (output / "cpu-real-factors.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
