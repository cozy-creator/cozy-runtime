"""CUDA proof of the lane execution contract (cr-109) against real diffusers and a real kernel.

Run in a process with torch, diffusers 0.40.0 and sageattention 2.2.0 on a CUDA card. The
stand-in for the H100 lane is the same distribution's sm89 entry — the box has an RTX 4070 —
under a contract naming this card; the design's own sm90 contract is read verbatim and refuses
`execution_contract_unsupported` here before a byte is allocated. Every other arm is real:
a tiny `MiniMaxH3Transformer3DModel` is pinned through `set_attention_backend`, its forwards
attest calls == sites x forwards with the profiled CUDA symbol, and each planted violation
(dropped site, extra call, wrong scheme kwarg, wrong activation dtype) fails typed.

Prints one JSON document on the last line.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from cozy_runtime.internal import canonical, execution_contract
from cozy_runtime.internal.encoding import measure_device

DESIGN_CONTRACT: dict[str, Any] = {
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

STANDIN_CONTRACT: dict[str, Any] = {
    "device": "sm89",
    "activations": "bf16",
    "class": "quantized",
    "weights": {"route": "verbatim"},
    "attention": {
        "distribution": "sageattention",
        "version": "2.2.0",
        "entry": "sageattn_qk_int8_pv_fp8_cuda",
        "kwargs": {
            "tensor_layout": "NHD",
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "is_causal": False,
        },
        "scheme": (
            "stand-in sm89: qk int8 per-thread K-mean-smoothed; v fp8-e4m3 per-channel; "
            "pv fp32+fp16"
        ),
    },
}


def contract(document: dict[str, Any], **attention: Any) -> execution_contract.ExecutionContract:
    body = json.loads(json.dumps(document))
    body["attention"].update(attention)
    parsed = execution_contract.read({"execution": canonical.write(body)})
    assert parsed is not None
    return parsed


def refusal(run: Any) -> dict[str, str]:
    try:
        run()
    except execution_contract.ExecutionContractRefusal as exc:
        return {"code": exc.code, "detail": str(exc)}
    return {"code": "", "detail": "no refusal"}


def main() -> None:
    import torch

    out: dict[str, Any] = {"device": torch.cuda.get_device_name(0)}
    facts = measure_device(torch, 0)
    out["sm"] = facts.sm
    assert torch.cuda.memory_allocated() == 0

    # RED: the design's own contract, verbatim, on this card — refused before allocation.
    design = contract(DESIGN_CONTRACT)
    out["design_digest"] = design.digest()
    out["unsupported"] = refusal(lambda: execution_contract.check_device(design, facts))
    out["allocated_after_unsupported"] = torch.cuda.memory_allocated()
    # The design contract's KERNEL is bindable here: same distribution, sm90 entry present.
    design_kernel = execution_contract.resolve_kernel(design)
    out["design_backend"] = design_kernel.backend
    out["design_live"] = dict(design_kernel.live)

    standin = contract(STANDIN_CONTRACT)
    execution_contract.check_device(standin, facts)
    out["kernel_absent"] = {
        "version": refusal(
            lambda: execution_contract.resolve_kernel(contract(STANDIN_CONTRACT, version="2.9.9"))
        ),
        "distribution": refusal(
            lambda: execution_contract.resolve_kernel(
                contract(STANDIN_CONTRACT, distribution="sageattention-nowhere")
            )
        ),
        "entry": refusal(
            lambda: execution_contract.resolve_kernel(
                contract(STANDIN_CONTRACT, entry="sageattn_x")
            )
        ),
        "kwarg_default": refusal(
            lambda: execution_contract.resolve_kernel(
                contract(
                    STANDIN_CONTRACT,
                    kwargs={**STANDIN_CONTRACT["attention"]["kwargs"], "qk_quant_gran": "per_warp"},
                )
            )
        ),
        "kwarg_unknown": refusal(
            lambda: execution_contract.resolve_kernel(
                contract(
                    STANDIN_CONTRACT,
                    kwargs={**STANDIN_CONTRACT["attention"]["kwargs"], "no_such_kwarg": 1},
                )
            )
        ),
        "sdpa_unbound": refusal(
            lambda: execution_contract.resolve_kernel(
                contract(
                    STANDIN_CONTRACT,
                    distribution="torch",
                    version=torch.__version__.split("+")[0],
                    entry="nn.functional.scaled_dot_product_attention",
                    kwargs={},
                )
            )
        ),
    }
    kernel = execution_contract.resolve_kernel(standin)
    out["backend"] = kernel.backend
    out["live"] = dict(kernel.live)

    # GREEN: a tiny real MiniMax-H3 transformer, pinned and attested.
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
        dispatch_attention_fn,
    )
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    torch.manual_seed(7)
    dit = (
        MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=64,
            hidden_size=128,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=256,
            in_channels=4,
            audio_in_channels=8,
            patch_size=(1, 2, 2),
            text_dim=32,
            freq_dim=32,
            time_embed_hidden_dim=128,
            time_embed_dim=64,
            rope_freq_dim=8,
        )
        .to("cuda", torch.bfloat16)
        .eval()
    )

    # The H3 package's shape: `H3Model.pipe.components["fl2va_dit"]` is the diffusers model.
    model_like = SimpleNamespace(pipe=SimpleNamespace(components={"fl2va_dit": dit}))

    text, video, audio = 128, 1792, 128
    seq = text + video + audio
    device = torch.device("cuda")
    text_indices = torch.arange(0, text, device=device)
    video_indices = torch.arange(text, text + video, device=device)
    audio_indices = torch.arange(text + video, seq, device=device)
    token_tags = torch.zeros(seq, dtype=torch.long, device=device)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    timestep_indices = torch.zeros(seq, dtype=torch.long, device=device)
    timestep_indices[audio_indices] = 1
    position_ids = torch.stack(
        [
            torch.arange(seq, device=device) // 64,
            torch.arange(seq, device=device) % 8,
            torch.arange(seq, device=device) % 3,
        ],
        dim=-1,
    )
    inputs = dict(
        hidden_states=torch.randn(1, video, 16, device=device, dtype=torch.bfloat16),
        audio_hidden_states=torch.randn(1, audio, 8, device=device, dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(1, text, 32, device=device, dtype=torch.bfloat16),
        timestep=torch.tensor([0.5, 0.0], device=device),
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        return_dict=False,
    )

    def forward() -> Any:
        with torch.inference_mode():
            return dit(**inputs)[0]

    reference = forward().float()
    applied = execution_contract.apply(standin, kernel, model_like)
    out["hosts"] = dict(applied.hosts)
    out["prepare_facts"] = applied.document()
    member = AttentionBackendName(kernel.backend)
    out["pinned_processors"] = sum(
        1
        for module in dit.modules()
        if getattr(getattr(module, "processor", None), "_attention_backend", None) == member
    )
    attestation = execution_contract.Attestation([applied])

    steps = 3
    attested_output, green = attestation.observe(lambda: [forward() for _ in range(steps)][-1])
    out["green"] = green
    out["steps"] = steps
    out["max_abs_diff_vs_native"] = float((attested_output.float() - reference).abs().max())
    out["prepare_facts_after"] = applied.document()

    # RED: one site dropped on one forward.
    block = dit.transformer_blocks[0]
    block.attn.processor._attention_backend = AttentionBackendName.NATIVE
    _, dropped = attestation.observe(forward)
    block.attn.processor._attention_backend = member
    out["dropped_site"] = dropped

    # RED: one extra call outside any forward.
    q = torch.randn(1, 64, 2, 64, device=device, dtype=torch.bfloat16)
    _, extra = attestation.observe(lambda: dispatch_attention_fn(q, q, q, backend=member))
    out["extra_call"] = extra

    # RED: the scheme's `is_causal` violated by a caller.
    _, causal = attestation.observe(
        lambda: dispatch_attention_fn(q, q, q, is_causal=True, backend=member)
    )
    out["scheme_mismatch"] = causal

    # RED: activations not the contract's dtype.
    dit.to(torch.float16)
    half = {
        k: (
            v.half()
            if torch.is_tensor(v) and v.is_floating_point() and v.dtype == torch.bfloat16
            else v
        )
        for k, v in inputs.items()
    }
    inputs.update(half)
    _, activations = attestation.observe(forward)
    out["activations_mismatch"] = activations
    dit.to(torch.bfloat16)

    out["registry_slot_is_proxy"] = (
        _AttentionBackendRegistry._backends[member] is not kernel.wrapper
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
