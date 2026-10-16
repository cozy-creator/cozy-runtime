"""`lm.generate` with a draft head gives Transformers' greedy tokens exactly: with a draft that is
mostly right (most drafts kept, some cut) and with a random one (almost all cut), on a plain
full-attention LM and on Qwen3.5's hybrid linear/full attention, across a window boundary, with
and without a presence penalty, call after call. On CUDA the steps replay as graphs."""

from __future__ import annotations

import contextvars
import sys
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cozy_runtime import lm  # noqa: E402
from cozy_runtime.author._model import WARNINGS  # noqa: E402
from cozy_runtime.internal import kernel_cache, kernel_compile, triton_kernels  # noqa: E402
from cozy_runtime.lm import _kernels  # noqa: E402

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
        text = positions[0] if positions.dim() == 3 else positions  # Qwen3.5's mrope rows
        at = (text + 1 - self.prompt).clamp(min=0)  # each row's next token
        token = self.reference[at]
        token = torch.where(at % self.miss == 0, (token + 1) % VOCAB, token)
        return self.rows[token] * 4


class _Random(torch.nn.Module):
    """A Qwen3.5-style head of random weights: its drafts are almost never kept."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        text = model.config.get_text_config(decoder=True)
        size = text.hidden_size
        self.fc = torch.nn.Linear(2 * size, size, bias=False)
        self.layer = torch.nn.Linear(size, size)

    def forward(self, embeds: Any, hidden: Any, positions: Any, cache: Any, mask: Any) -> Any:
        return torch.tanh(self.layer(self.fc(torch.cat([embeds, hidden], dim=-1))))


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
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_drafted_tokens_are_the_greedy_tokens(
    make: Any,
    penalty: float,
    depth: int,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lm, "DEPTH", depth)
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
    rates = [float(line.split(f"drafting {depth}: ")[1].split(" ")[0]) for line in lines]
    # The oracle's drafts are mostly kept, the random head's hardly ever.
    assert len(rates) == 6 and min(rates[0::2]) > 1.25 > max(rates[1::2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels need a card")
def test_a_verify_pass_steps_its_tokens_as_decode_does(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verify pass's one-launch conv and recurrence give each token exactly what decoding
    it alone through the one-token kernels gives: outputs, and the state after each token."""
    store = kernel_cache.Store(tmp_path / "u")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(store.own / "triton"))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [store])
    major, minor = torch.cuda.get_device_capability()
    if triton_kernels.ready(lm.KERNELS, device="cuda") is None:
        job = triton_kernels.job(lm.KERNELS, major * 10 + minor)
        assert kernel_compile.wait(store, job).state == "ready"
    launch = triton_kernels.ready(lm.KERNELS, device="cuda")
    monkeypatch.setattr(lm, "_STEPS", [launch])
    space = sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"]
    verify = lm._Verify()
    monkeypatch.setattr(lm, "_VERIFY", contextvars.ContextVar("verify", default=verify))
    torch.manual_seed(0)
    tokens, heads, width, channels = 4, 8, _kernels.KEY_DIM, 512
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}

    step = lm._recurrent_slot(None, launch)
    q, k, v = (torch.randn(1, tokens, heads, width, **bf16) for _ in range(3))
    g = -torch.rand(1, tokens, heads, device="cuda")
    beta = torch.rand(1, tokens, heads, **bf16)
    state = torch.randn(1, heads, width, width, device="cuda")
    many = lm._stepped_recurrence(space, None)
    out, last = many(q, k, v, g, beta, initial_state=state, use_qk_l2norm_in_kernel=True)
    alone = state.clone()
    for t in range(tokens):
        one, alone = step(
            q[:, t : t + 1],
            k[:, t : t + 1],
            v[:, t : t + 1],
            g[:, t : t + 1].contiguous(),
            beta[:, t : t + 1].contiguous(),
            alone.clone(),
            True,
            True,
        )
        assert torch.equal(out[:, t : t + 1], one)
        assert torch.equal(verify.recurrent[-1][1][t], alone)
    assert torch.equal(last, alone)

    conv = lm._conv_slot(None, launch)
    weight = torch.randn(channels, _kernels.CONV, **bf16)
    x = torch.randn(1, channels, _kernels.CONV + tokens, **bf16)
    got = lm._stepped_conv(space, None)(x, weight, None, activation="silu")
    window = x[..., : _kernels.CONV].contiguous()
    for t in range(tokens):
        one = conv(x[..., _kernels.CONV + t :][..., :1].contiguous(), window, weight, None, "silu")
        assert torch.equal(got[..., _kernels.CONV + t], one[..., 0])


def test_a_head_that_does_not_fit_decodes_without_drafting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where the model's decode set fits at home only without the head (a 16 GB card holding
    Klein and its upsampler), the graphed decode runs undrafted, not Transformers' loop."""
    torch.manual_seed(5)
    model = _qwen3_5()
    ids = torch.randint(10, 44, (1, 30), device=DEVICE)
    want = _reference(model, ids, 40, 0.0)
    static = lm._static
    monkeypatch.setattr(lm, "_static", lambda *a: None if a[-1] is not None else static(*a))
    WARNINGS.clear()
    got = lm.generate(
        model,
        {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
        max_new_tokens=40,
        stop=STOP,
        pad=0,
        draft=_head(model),
    )
    assert torch.equal(got, want)
    assert [message for message, _ in WARNINGS] == ["decoded without drafting"]
