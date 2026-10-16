"""`lm.generate` with a draft head gives Transformers' greedy tokens exactly: with a draft that is
mostly right (most drafts kept, some cut) and with a random one (almost all cut), on a plain
full-attention LM and on Qwen3.5's hybrid linear/full attention, across a window boundary, with
and without a presence penalty, call after call. On CUDA the steps replay as graphs."""

from __future__ import annotations

import sys
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cozy_runtime import lm  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = 48  # below the hidden size: an orthonormal head lets a draft name any token
STOP = [7, 9]


def _qwen3() -> Any:
    config = transformers.AutoConfig.for_model(
        "qwen3",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=VOCAB,
    )
    return _orthonormal(transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32))


def _qwen3_5() -> Any:
    config = transformers.AutoConfig.for_model(
        "qwen3_5",
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "vocab_size": VOCAB,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
        },
        image_token_id=45,
        vision_start_token_id=46,
        vision_end_token_id=47,
    )
    model = transformers.AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
    return _orthonormal(model)


def _orthonormal(model: Any) -> Any:
    weight = model.get_output_embeddings().weight
    rows, _ = torch.linalg.qr(torch.randn(weight.shape[1], weight.shape[0]))
    with torch.no_grad():
        weight.copy_(rows.T * 4)
    return model.to(DEVICE).eval()


class _Oracle(torch.nn.Module):
    """A draft that knows the model's continuation and gets every `miss`-th token wrong."""

    def __init__(self, model: Any, prompt: int, reference: Any, miss: int) -> None:
        super().__init__()
        self.rows = model.get_output_embeddings().weight
        self.prompt, self.miss = prompt, miss
        self.reference = torch.cat([reference[0], torch.zeros(64, dtype=torch.long, device=DEVICE)])

    def forward(self, embeds: Any, hidden: Any, positions: Any, cache: Any, mask: Any) -> Any:
        at = (positions.reshape(-1)[:1] + 1 - self.prompt).clamp(min=0)
        token = self.reference.index_select(0, at)
        token = torch.where(at % self.miss == 0, (token + 1) % VOCAB, token)
        return self.rows.index_select(0, token)[None] * 4, torch.zeros_like(hidden)


class _Random(torch.nn.Module):
    """A Qwen3.5-style head of random weights: its drafts are almost never kept."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        text = model.config.get_text_config(decoder=True)
        size = text.hidden_size
        self.fc = torch.nn.Linear(2 * size, size, bias=False)
        self.layer = torch.nn.Linear(size, size)

    def forward(self, embeds: Any, hidden: Any, positions: Any, cache: Any, mask: Any) -> Any:
        out = torch.tanh(self.layer(self.fc(torch.cat([embeds, hidden], dim=-1))))
        return out, out


def _head(model: Any) -> Any:
    """Qwen3.5's own head (`lm.Draft`) of random weights; a plain LM gets a random stand-in."""
    space = sys.modules[type(model).__module__]
    if hasattr(space, "Qwen3_5DecoderLayer"):
        return lm.Draft(model.config, space).to(DEVICE).eval()
    return _Random(model).to(DEVICE)


def _reference(model: Any, ids: Any, new: int, penalty: float) -> Any:
    prompt = int(ids.shape[1])

    def presence(input_ids: Any, scores: Any) -> Any:
        generated = input_ids[:, prompt:]
        if not generated.shape[1] or not penalty:
            return scores
        return scores - penalty * torch.zeros_like(scores).scatter_(1, generated, 1.0)

    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=new,
            do_sample=False,
            eos_token_id=STOP,
            pad_token_id=0,
            logits_processor=transformers.LogitsProcessorList([presence]),
        )
    return out[:, prompt:]


@pytest.mark.parametrize("make", [_qwen3, _qwen3_5], ids=["qwen3", "qwen3_5"])
@pytest.mark.parametrize("penalty", [0.0, 1.5])
def test_drafted_tokens_are_the_greedy_tokens(
    make: Any, penalty: float, capfd: pytest.CaptureFixture[str]
) -> None:
    torch.manual_seed(5)
    model = make()
    for call, prompt in enumerate((250, 251, 30)):  # the first two cross the 256 window
        ids = torch.randint(10, 44, (1, prompt), device=DEVICE)
        want = _reference(model, ids, 40, penalty)
        for draft in (_Oracle(model, prompt, want, miss=5), _head(model)):
            got = lm.generate(
                model,
                {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                max_new_tokens=40,
                stop=STOP,
                pad=0,
                presence_penalty=penalty,
                draft=draft,
            )
            assert torch.equal(got, want), (call, type(draft).__name__)
    lines = [line for line in capfd.readouterr().err.splitlines() if "tokens a step" in line]
    rates = [float(line.split("drafting 3: ")[1].split(" ")[0]) for line in lines]
    assert len(rates) == 6 and max(rates[0::2]) > 2.0  # the oracle's drafts are mostly kept
