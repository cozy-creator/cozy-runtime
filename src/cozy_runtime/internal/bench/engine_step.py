"""The engine's decode step on a GPU (the varena suite's `engine-step` case; xs-213 phase 2):
`engine.decoder.Decoder` on a small Qwen3.5 of random weights (three gated-delta layers and one
attention layer, a multi-token-prediction head), against Transformers' own forward.

- Plain: the decoder's greedy tokens, held against Transformers' forward over the prompt and
  those tokens in one pass (teacher-forced), at every position where its choice is clear.
- Drafted: the decoder's drafted greedy tokens against its plain ones (a draft only changes how
  many tokens a step keeps, never which).
- Rows: four prompts of different lengths in one bucket, each row against itself alone.
- Sampled (temperature 0.7, top-k 20, top-p 0.95, a seed): drafted against plain and plain against
  itself, reported (each draw's noise is hashed from the seed and the token's index; a near tie in the
  scores can still turn a draw, so equality is told, not required of the real model).
- The step's graph: recorded once and replayed (or why not).

The head is scaled so greedy choices are decisive with random weights. It reports and fails on a
wrong token.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import cast

import torch

type JSON = None | bool | int | float | str | Sequence[JSON] | Mapping[str, JSON]

#: A clear choice: the top logit leads the next by at least this.
MARGIN = 0.5
NEW = 48


def _model() -> torch.nn.Module:
    import sys

    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from cozy_runtime import lm

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
    made = Qwen3_5ForCausalLM(config)  # type: ignore[no-untyped-call]  # unannotated in Transformers
    model = cast("torch.nn.Module", made).to(device="cuda", dtype=torch.bfloat16).eval()
    head = lm.Draft(made.config, sys.modules[type(made).__module__]).to(device="cuda", dtype=torch.bfloat16)
    for parameter in head.parameters():
        parameter.data.normal_(0, 0.02)
    model.mtp = head.mtp  # as `vlm.construct` sets it
    with torch.no_grad():
        cast("torch.nn.Linear", model.lm_head).weight.mul_(40.0)
    return model


def _forced(model: torch.nn.Module, ids: list[int]) -> tuple[list[int], list[float]]:
    """Transformers' greedy choice after each position of `ids`, and its lead over the next."""
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids], device="cuda")).logits[0].float()
    top = logits.topk(2, dim=-1)
    return top.indices[:, 0].tolist(), (top.values[:, 0] - top.values[:, 1]).tolist()


def measure() -> dict[str, JSON]:
    from cozy_runtime.engine.decoder import Decoder, Request

    model = _model()
    g = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, 4096, (n,), generator=g).tolist() for n in (37, 5, 120, 64)]
    row: dict[str, JSON] = {}
    runs: dict[str, list[list[int]]] = {}
    head = model.mtp
    sampled = Request(prompts=prompts[:2], max_new=NEW, stops=[], temperature=0.7, top_k=20, top_p=0.95, seed=213)
    for name, drafting in (("plain", False), ("drafted", True)):
        if drafting:
            model.mtp = head
        else:
            del model.mtp
        decoder = Decoder(model, max_rows=4, max_tokens=1024)
        began = time.perf_counter()
        call = decoder(Request(prompts=prompts, max_new=NEW, stops=[]))
        row[name] = {
            "captured": call.captured, "steps": call.steps, "notes": list(call.notes),
            "prefill_s": call.prefill_s, "decode_s": call.decode_s, "wall_s": time.perf_counter() - began,
            "tokens": [len(t) for t in call.tokens],
        }  # fmt: skip
        runs[name] = call.tokens
        alone = decoder(Request(prompts=prompts[:1], max_new=NEW, stops=[]))
        runs[f"{name}-alone"] = alone.tokens
        drawn = decoder(sampled)
        runs[f"{name}-sampled"] = drawn.tokens
        kept = [k for row in drawn.kept for k in row]
        row[f"{name}-sampled"] = {"kept_a_step": sum(kept) / max(len(kept), 1), "steps": drawn.steps}
        kept = [k for r in call.kept for k in r]
        row[name]["kept_a_step"] = sum(kept) / max(len(kept), 1)  # type: ignore[index]  # the dict just made
        if not drafting:
            runs["plain-sampled-again"] = decoder(sampled).tokens
        # The same step replayed by the host, in place of the launcher's WHILE node: the same tokens.
        replayed = Decoder(model, max_rows=4, max_tokens=1024, launcher=False)
        runs[f"{name}-replayed"] = replayed(Request(prompts=prompts, max_new=NEW, stops=[])).tokens
        # The step's small ops as torch's (the reference the fused kernels are held against).
        unfused = Decoder(model, max_rows=4, max_tokens=1024, fused=False)
        runs[f"{name}-unfused"] = unfused(Request(prompts=prompts, max_new=NEW, stops=[])).tokens
        # The prompts prefilled 64 tokens a forward (two of them span chunks), against Transformers too.
        chunked = Decoder(model, max_rows=4, max_tokens=1024, prefill_chunk=64)
        runs[f"{name}-chunked"] = chunked(Request(prompts=prompts, max_new=NEW, stops=[])).tokens
        # The card holding 32 tokens of each row's K/V: every older whole block on the host tier,
        # prefilled there and read there each step. The same tokens as all on the card.
        card = 4 * 32 * decoder.kv.layout.token_bytes
        hosted = Decoder(model, max_rows=4, max_tokens=1024, kv_card_bytes=card)  # copied ahead to the card
        runs[f"{name}-hosted"] = hosted(Request(prompts=prompts, max_new=NEW, stops=[])).tokens
        in_place = Decoder(model, max_rows=4, max_tokens=1024, kv_card_bytes=card, in_place=True)  # the fallback
        runs[f"{name}-in-place"] = in_place(Request(prompts=prompts, max_new=NEW, stops=[])).tokens
        row[name]["launcher"] = decoder.launcher  # type: ignore[index]  # the dict just made
    checks: dict[str, JSON] = {}
    wrong: list[str] = []

    def forced(prompt: list[int], made: list[int]) -> tuple[int, int, int | None]:
        choice, lead = _forced(model, list(prompt) + made[:-1])
        clear = agreed = 0
        first_wrong: int | None = None
        for t, token in enumerate(made):
            at = len(prompt) - 1 + t
            if lead[at] >= MARGIN:
                clear += 1
                if choice[at] == token:
                    agreed += 1
                elif first_wrong is None:
                    first_wrong = t
        return clear, agreed, first_wrong

    for i, prompt in enumerate(prompts):
        plain = runs["plain"][i]
        clear, agreed, first_wrong = forced(prompt, plain)
        _, unfused_agreed, unfused_wrong = forced(prompt, runs["plain-unfused"][i])
        _, chunked_agreed, chunked_wrong = forced(prompt, runs["plain-chunked"][i])
        same_drafted = runs["drafted"][i] == plain
        checks[f"row{i}"] = {
            "prompt": len(prompt), "clear": clear, "agreed": agreed, "first_wrong": first_wrong,
            "drafted_equal": same_drafted, "unfused_agreed": unfused_agreed, "unfused_first_wrong": unfused_wrong,
            "unfused_equal": runs["plain-unfused"][i] == plain,
            "chunked_agreed": chunked_agreed, "chunked_first_wrong": chunked_wrong,
            "chunked_equal": runs["plain-chunked"][i] == plain,
            "drafted_first_diff": next((t for t, (a, b) in enumerate(zip(plain, runs["drafted"][i], strict=False)) if a != b), None),
        }  # fmt: skip
        if first_wrong is not None:
            wrong.append(f"row {i}: plain token {first_wrong} is not Transformers' clear choice "
                         f"(unfused: first wrong {unfused_wrong})")
        if chunked_wrong is not None:
            wrong.append(f"row {i}: prefilled 64 tokens a forward, plain token {chunked_wrong} is not Transformers' clear choice")
    checks["sampled_drafted_equal"] = runs["drafted-sampled"] == runs["plain-sampled"]
    checks["sampled_repeat_equal"] = runs["plain-sampled-again"] == runs["plain-sampled"]
    checks["sampled_differs_from_greedy"] = runs["plain-sampled"][0] != runs["plain"][0]
    checks["alone_plain_equal"] = runs["plain-alone"][0] == runs["plain"][0]
    checks["alone_drafted_equal"] = runs["drafted-alone"][0] == runs["drafted"][0]
    for name in ("plain", "drafted"):
        checks[f"{name}_replayed_equal"] = runs[f"{name}-replayed"] == runs[name]
        if runs[f"{name}-replayed"] != runs[name]:
            wrong.append(f"{name}: the launcher's tokens are not the host's replays' for the same step")
        for arm in ("hosted", "in-place"):
            checks[f"{name}_{arm}_equal"] = runs[f"{name}-{arm}"] == runs[name]
            if runs[f"{name}-{arm}"] != runs[name]:
                wrong.append(f"{name}: with K/V on the host tier ({arm}), other tokens than with it all on the card")
    row["checks"] = checks
    assert not wrong, wrong
    return row
