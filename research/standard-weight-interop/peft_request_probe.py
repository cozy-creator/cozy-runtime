"""One full-size Spatial/Wushu projection through Runtime's native PEFT lifecycle.

CPU forward intentionally uses TorchAO weight-only execution. This is numerical
and ownership evidence, not W8A8 emulation, GPU performance or video quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("CPU qualification requires CUDA_VISIBLE_DEVICES=''")

    import torch

    from cozy_runtime.author._loader import census
    from cozy_runtime.internal.base_weights import BaseWeights
    from cozy_runtime.internal.encoding.torchao_rowwise import rowwise_tensor
    from cozy_runtime.internal.lora import AdapterInput, LinearAdapters

    torch.set_num_threads(2)
    torch.manual_seed(7381)
    torch.set_float32_matmul_precision("highest")
    base = torch.nn.Linear(5376, 7168, bias=False, dtype=torch.bfloat16, device="meta")
    model = torch.nn.Sequential(base)
    owned = BaseWeights("dit", model, census(SimpleNamespace(components={"dit": model})))
    payload = torch.randint(-100, 101, (7168, 5376)).to(torch.float8_e4m3fn)
    scales = torch.linspace(1e-4, 1e-2, 7168, dtype=torch.float32).reshape(-1, 1)

    def adopt() -> None:
        base.weight = torch.nn.Parameter(
            rowwise_tensor(payload, scales, dtype=torch.bfloat16), requires_grad=False
        )

    adopt()
    x = torch.randn(8, 5376, dtype=torch.bfloat16)
    baseline = base(x).detach().clone()
    adapters = {}
    factors = {}
    provenance = {}
    for row in json.loads((args.factor_dir / "cpu-real-factors.json").read_text())["factors"]:
        pair = []
        for letter, info in zip(("A", "B"), row["parts"], strict=True):
            data = bytearray((args.factor_dir / f"{row['label']}-{letter}.bin").read_bytes())
            assert hashlib.sha256(data).hexdigest() == info["sha256"].removeprefix("sha256:")
            dtype = {"f16": torch.float16, "f32": torch.float32}[info["dtype"]]
            pair.append(torch.frombuffer(data, dtype=dtype).clone().reshape(info["shape"]))
        label = row["label"]
        factors[label] = pair
        provenance[label] = row["parts"]
        adapters[label] = AdapterInput(
            label,
            "dit",
            "base",
            dict(zip(("0.lora_A.weight", "0.lora_B.weight"), pair, strict=True)),
            scale={"spatial": 0.3, "wushu": 0.8}[label],
        )
    binding = LinearAdapters({"dit": model}, {"dit": "base"}, {"dit": ["0"]})
    report = {"scope": __doc__, "wrapper": type(model[0]).__name__, "adapters": []}
    first = None
    try:
        for label in ("spatial", "wushu", "spatial"):
            capacity = []
            adapter = adapters[label]
            with binding.request(
                [adapter],
                device="cpu",
                max_device_bytes=8 << 20,
                check_cancelled=lambda: None,
                check_capacity=capacity.append,
            ):
                actual = model(x)
                repeated = model(x)
                assert torch.equal(actual, repeated)
                a, b = factors[label]
                # Independent FP32 residual reference only; Runtime executes PEFT.
                delta32 = (
                    torch.nn.functional.linear(
                        torch.nn.functional.linear(x.float(), a.float()), b.float()
                    )
                    * adapter.scale
                )
                expected32 = baseline.float() + delta32
                layer = model[0]
                native_delta = (
                    layer.lora_B["request_0"](layer.lora_A["request_0"](x)) * adapter.scale
                )
                factor_objects = [
                    *layer.lora_A["request_0"].parameters(),
                    *layer.lora_B["request_0"].parameters(),
                ]
                pointers = [p.data_ptr() for p in factor_objects]
                owned.evict()
                assert all(
                    p.data_ptr() == pointer
                    for p, pointer in zip(factor_objects, pointers, strict=True)
                )
                owned.reserve("cpu")
                adopt()
                assert torch.equal(model(x), actual)
                row = {
                    "label": label,
                    "source_parts": provenance[label],
                    "factor_dtype": str(factor_objects[0].dtype),
                    "preparation": binding.preparation_facts(),
                    "factor_bytes": sum(p.nbytes for p in factor_objects),
                    "rank": int(a.shape[0]),
                    "strength": adapter.scale,
                    "full_output_relative_rmse": float(
                        (actual.float() - expected32).norm() / expected32.norm()
                    ),
                    "rounded_fp32_reference_relative_rmse": float(
                        (actual.float() - expected32.to(torch.bfloat16).float()).norm()
                        / expected32.norm()
                    ),
                    "adapter_delta_relative_rmse": float(
                        (native_delta.float() - delta32).norm() / delta32.norm()
                    ),
                    "adapter_delta_cosine": float(
                        torch.nn.functional.cosine_similarity(
                            native_delta.float().flatten(), delta32.flatten(), dim=0
                        )
                    ),
                    "repeat_exact": True,
                    "base_evict_reload_exact": True,
                    "factor_objects_survived_base_eviction": True,
                    "capacity_bytes": capacity,
                    "output_sha256": hashlib.sha256(
                        actual.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
                report["adapters"].append(row)
                if first is None:
                    first = actual.clone()
                elif label == "spatial":
                    assert torch.equal(first, actual)
        assert torch.equal(model(x), baseline)
        report["switch_back_exact"] = True
        report["base_after_requests_exact"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        binding.close()


if __name__ == "__main__":
    main()
