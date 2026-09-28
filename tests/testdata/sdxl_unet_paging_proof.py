"""SDXL's own UNet (the checkpoint's config and weights, from a TensorFS store) runs one CFG
denoise step resident and then paged through its innermost no-split blocks: the outputs are
bit-identical, and the paged run holds a fraction of the weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel

from cozy_runtime.author._loader import census
from cozy_runtime.internal.fill import Checkpoint, StreamingFillBackend, _Holder
from cozy_runtime.internal.memory import ComponentResidency, partition


def build(checkpoint: Checkpoint) -> tuple[ComponentResidency, UNet2DConditionModel]:
    rows = list(checkpoint.rows("unet"))
    config = json.loads(checkpoint.header["configs"]["unet"])
    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(config).to(torch.float16).eval()
    backend = StreamingFillBackend.for_script(
        checkpoint,
        rows,
        release="sdxl-unet/1",
        store="proof",
        snapshot=checkpoint.manifest_id,
        placement="all_resident",
        objective="footprint",
    )
    holder = _Holder({"unet": unet})
    walked = census(holder)
    backend.expect({"unet": [d.key for d in walked.destinations]})
    live = backend.materialize(holder, walked, encoded_leaves="accept")
    for destination in walked.destinations:
        backend.fill(destination.key, destination.spec, live[destination.key])
    backend.commit()
    return ComponentResidency(backend=backend, torch=torch, placement="all_resident"), unet


def step(plane: ComponentResidency, unet: UNet2DConditionModel) -> torch.Tensor:
    torch.manual_seed(3)
    latents = torch.randn(2, 4, 128, 128, device="cuda", dtype=torch.float16)
    text = torch.randn(2, 77, 2048, device="cuda", dtype=torch.float16)
    added = {
        "text_embeds": torch.randn(2, 1280, device="cuda", dtype=torch.float16),
        "time_ids": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]] * 2, device="cuda").half(),
    }
    plane.open_attempt("all_resident", 0)
    with torch.inference_mode():
        plane.admit("denoise", ("unet",))
        out = unet(latents, 500, encoder_hidden_states=text, added_cond_kwargs=added).sample
        plane.release("denoise", ("unet",))
    torch.cuda.synchronize()
    return out.float().cpu()


def main(store: Path, manifest: str) -> None:
    checkpoint = Checkpoint(store, manifest)
    plane, unet = build(checkpoint)
    declared = list(UNet2DConditionModel._no_split_modules)
    layout = partition(unet)
    assert layout is not None
    resident, again = step(plane, unet), step(plane, unet)
    resident_peak = torch.cuda.max_memory_allocated()
    # The control: the same resident weights copied to new addresses, no paging at all.
    with torch.no_grad():
        for module in unet.modules():
            for name, param in list(module._parameters.items()):
                if param is not None:
                    module._parameters[name] = torch.nn.Parameter(param.clone(), False)
    moved = step(plane, unet)
    plane.vacate()
    torch.cuda.empty_cache()
    plane.backend.paging["unet"] = plane.backend.layouts["unet"]
    torch.cuda.reset_peak_memory_stats()
    paged = [step(plane, unet) for _ in range(2)]
    print(
        json.dumps(
            {
                "phase": "passed",
                "declared": declared,
                "blocks": len(layout.blocks),
                "working_bytes": layout.working_bytes,
                "total_bytes": layout.total_bytes,
                "resident_peak_bytes": resident_peak,
                "paged_peak_bytes": plane.attempt_absolute_peak_bytes,
                "resident_repeatable": torch.equal(resident, again),
                "paged_repeatable": torch.equal(paged[0], paged[1]),
                "bit_identical": all(torch.equal(resident, run) for run in paged),
                "max_error": max(float((resident - run).abs().max()) for run in paged),
                "moved_error": float((resident - moved).abs().max()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
