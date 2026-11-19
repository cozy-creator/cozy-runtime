"""cr-342's cell (`bench.lm_opening_kernel`): a decode whose first step meets a kernel still building completes, eager
until it lands (cozy-runtime#2289).

The escaped bug: the launch walkthrough's first run on a fresh RTX 4090 failed 11 s in, `Run PLA-2 failed
[unhandled_exception]: Building: attention cuda:0 sm89 sageattention` (Runtime 0.48.0.dev2346+g051d106f); an A40's
first decode met `llama.cpp-ggml`'s probe the same way (44424148955). `lm._Engine`'s warm-up opened its graph domain
(`domain.initialize()`, under `kernel_ready.recording()`) outside the capture's `except Building`. Here a probe that
takes 20 s is primed, then a small Qwen3.5 of random weights (Transformers', built in the call: three gated-delta
layers and one attention layer) decodes 32 greedy tokens through `lm.generate`, cold, while it opens: red on
051d106fa, green with #2289. Transformers is imported here, in the caller's environment, never where the builtins
are registered. Told: the tokens, the engine's eager steps, the wall time and whether the probe was still opening at
the end; a decode that fails raises.
"""

from __future__ import annotations

import time
from typing import cast

#: The primed probe's time to open: far longer than the decode, so the decode's first steps meet it opening.
PROBE_S = 20.0
NEW = 32


def measure() -> dict[str, object]:
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from cozy_runtime import lm
    from cozy_runtime.internal import kernel_ready

    config = Qwen3_5TextConfig(
        vocab_size=4096, hidden_size=512, intermediate_size=1024, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
        num_key_value_heads=2, head_dim=128, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 0.25,
                         "mrope_section": [6, 5, 5]},
        tie_word_embeddings=False,
    )  # fmt: skip
    torch.manual_seed(213)
    model = Qwen3_5ForCausalLM(config)  # type: ignore[no-untyped-call]  # unannotated in Transformers
    cast("torch.nn.Module", model).to(device="cuda", dtype=torch.bfloat16).eval()  # in place, as engine-step moves it
    name = f"xs213-slow-probe cuda:{torch.cuda.current_device()}"

    def slow(wait: bool) -> object:
        time.sleep(PROBE_S)
        return object()

    kernel_ready.prime(name, slow)
    ids = torch.arange(1, 39, device="cuda").unsqueeze(0)  # 38 tokens, ggml-decode's prompt length
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    opening = kernel_ready.opening(name)
    torch.cuda.synchronize()
    began = time.perf_counter()
    out = lm.generate(model, inputs, max_new_tokens=NEW, stop=[], pad=0, temperature=0.0)
    torch.cuda.synchronize()
    seconds = round(time.perf_counter() - began, 3)
    engine = lm._ENGINES.get(model)
    still = kernel_ready.opening(name)
    # Returned while the probe still opened, every `recording()` in this executor would refuse (`Building`) for the rest
    # of its 20 s: lm's captures refused, and a timed decode right after measuring eager steps. So the cell ends only
    # once the probe's opener has ended.
    deadline = time.monotonic() + PROBE_S + 60
    while kernel_ready.opened(name) is None and kernel_ready.failed(name) is None:
        if time.monotonic() > deadline:
            raise RuntimeError(f"{name} still opening {PROBE_S + 60:.0f} s after it was primed")
        time.sleep(0.1)
    return {
        "tokens": int(out.shape[-1]),
        "s": seconds,
        "opening_at_start": opening,
        "still_opening": still,
        "eager_steps": None if engine is None else engine.eager,
    }
