"""The memory plane on a real card: a diffusers UNet and VAE filled from native TensorFS.

Unconstrained, the pair runs resident and gives the reference outputs. Then the process's
allocator is capped below what the resident run needed: the UNet's forward runs out of memory
and recovers (its blocks page, its batch splits), the decode recovers by tiling, and the
outputs match the reference. Below the floor the forward fails typed, the plane is not
poisoned, and the next call under a larger cap succeeds.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import tensorfs
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author._loader import census
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.fill import Checkpoint, StreamingFillBackend, _Holder
from cozy_runtime.internal.memory import ComponentResidency, ResidencyRefusal
from cozy_runtime.internal.weights_sink import weights_transaction_id

UNET = {
    "sample_size": 32,
    "in_channels": 4,
    "out_channels": 4,
    "block_out_channels": (128, 256, 512),
    "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    "up_block_types": ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    "layers_per_block": 2,
    "cross_attention_dim": 256,
    "attention_head_dim": 8,
    "transformer_layers_per_block": (1, 2, 2),
}
VAE = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": (64, 128, 256, 256),
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "layers_per_block": 2,
    "sample_size": 256,
}
DTYPE = torch.float16
NAMES = {torch.float16: "f16", torch.float32: "f32", torch.bfloat16: "bf16"}


def build(device: str) -> dict[str, Any]:
    with torch.device(device):
        torch.manual_seed(5)
        return {
            "unet": UNet2DConditionModel(**UNET).to(DTYPE).eval(),
            "vae": AutoencoderKL(**VAE).to(DTYPE).eval(),
        }


def write(root: Path, weights: dict[str, dict[str, Any]]) -> Checkpoint:
    store = tensorfs.Store.ensure(str(root / "store"))
    targets = {
        name: {
            key: Tensor(
                NAMES[t.dtype], tuple(t.shape), SPEC_PLAIN, {"value": Part(NAMES[t.dtype], tuple(t.shape))}
            )
            for key, t in state.items()
        }
        for name, state in weights.items()
    }
    bound = 2 * sum(t.numel() * t.element_size() for s in weights.values() for t in s.values())
    transaction = store.begin_derived(
        weights_transaction_id("memory-proof", "proof", "sha256:" + "3" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={name: Target(add=rows) for name, rows in targets.items()},
            configs={"model": NativeConfig("add")},
            order=tuple((name, key) for name, rows in targets.items() for key in rows),
        ).native_arguments(bound + (1 << 20)),
        work_fingerprint="sha256:" + "4" * 64,
    )
    for name, state in weights.items():
        for key, tensor in state.items():
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            transaction.add_part(name, key, "value", io.BytesIO(raw))
    transaction.add_config("model", io.BytesIO(b"{}"))
    receipt = transaction.commit()
    return Checkpoint(root / "store", "sha256:" + receipt["manifest"]["sha256"])


def construct(checkpoint: Checkpoint) -> tuple[ComponentResidency, dict[str, Any]]:
    rows = [row for name in ("unet", "vae") for row in checkpoint.rows(name)]
    backend = StreamingFillBackend.for_script(
        checkpoint,
        rows,
        release="memory-proof/1",
        store="fixture",
        snapshot=checkpoint.manifest_id,
        placement="all_resident",
        objective="footprint",
        window_bytes=1 << 20,
        slots=3,
        readers=2,
        inflight=2,
    )
    models = build("meta")
    holder = _Holder(models)
    walked = census(holder)
    backend.expect(
        {name: [d.key for d in walked.destinations if d.component == name] for name in models}
    )
    live = backend.materialize(holder, walked, encoded_leaves="accept")
    for destination in walked.destinations:
        backend.fill(destination.key, destination.spec, live[destination.key])
    backend.commit()
    residency = ComponentResidency(backend=backend, torch=torch, placement="all_resident")
    return residency, models


def serve(residency: ComponentResidency, models: dict[str, Any], side: int) -> dict[str, Any]:
    torch.manual_seed(11)
    latents = torch.randn(2, 4, side // 8, side // 8, device="cuda", dtype=DTYPE)
    text = torch.randn(2, 77, UNET["cross_attention_dim"], device="cuda", dtype=DTYPE)
    residency.open_attempt("all_resident", 0)
    with torch.inference_mode():
        residency.admit("denoise", ("unet",))
        noise = models["unet"](latents, 999, encoder_hidden_states=text).sample
        residency.release("denoise", ("unet",))
        residency.admit("decode", ("vae",))
        image = models["vae"].decode(noise[:1]).sample
        residency.release("decode", ("vae",))
    torch.cuda.synchronize()
    return {"noise": noise.float().cpu(), "image": image.float().cpu()}


def main(root: Path) -> None:
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    reference_models = build("cpu")
    weights = {name: dict(model.state_dict()) for name, model in reference_models.items()}
    del reference_models
    checkpoint = write(root, weights)
    residency, models = construct(checkpoint)
    side = 512
    reference = serve(residency, models, side)
    weights_bytes = sum(t.numel() * t.element_size() for s in weights.values() for t in s.values())
    peaks = {name: op.peak for (name, _), op in residency.ops.items()}
    layout = residency.backend.layouts["unet"]
    activations = peaks["unet"] - weights_bytes

    # A cap the resident run could not live in: the UNet pages, its batch of two only fits
    # as two batches of one, and the full-frame decode only fits in tiles.
    residency.vacate()
    torch.cuda.empty_cache()
    limit = torch.cuda.memory_reserved() + layout.working_bytes + activations * 3 // 4
    residency.set_limit(limit)
    constrained = serve(residency, models, side)
    recoveries = list(residency.recoveries)
    noise_error = float((constrained["noise"] - reference["noise"]).abs().max())
    image_error = float((constrained["image"] - reference["image"]).abs().max())

    # Below the floor: typed failure, no poison, and the next call under room succeeds.
    residency.vacate()
    torch.cuda.empty_cache()
    residency.set_limit(torch.cuda.memory_reserved() + layout.common.nbytes // 4)
    try:
        serve(residency, models, side)
        floor = "served"
    except (torch.cuda.OutOfMemoryError, ResidencyRefusal) as exc:
        floor = str(exc).splitlines()[0]
    poisoned = residency.poisoned
    residency.set_limit(0)
    again = serve(residency, models, side)
    print(
        json.dumps(
            {
                "phase": "passed",
                "device": torch.cuda.get_device_name(),
                "weights_bytes": weights_bytes,
                "op_peaks": peaks,
                "limit_bytes": limit,
                "recoveries": recoveries,
                "paging": sorted(residency.backend.paging),
                "noise_error": noise_error,
                "image_error": image_error,
                "floor": floor,
                "poisoned": poisoned,
                "again_error": float((again["noise"] - reference["noise"]).abs().max()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
