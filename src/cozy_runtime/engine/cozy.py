"""`CozyEngine` (xs-213 phase 2): `vlm.Engine` on the engine's own decoder where it serves the
call, Transformers' engine where it doesn't, with the reason in the call's timings note.

It serves a CUDA Qwen3.5 text call: text-only questions, no repeat-n-gram cut, at most the
decoder's rows. The questions are read as Transformers' engine reads them (its chat template and
tokenizer), each row's text is handed on as it comes (at most every `STREAM_S`), and stops are
cut as `VisionLM.ask` cuts them.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, cast

import torch

from cozy_runtime.engine.decoder import Decoder, Request
from cozy_runtime.engine.qwen35 import text_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import ProcessorMixin

    from cozy_runtime.vlm import Delta, Final, Params, Pipeline, Question

#: Text handed on at most this often, a row at a time.
STREAM_S = 0.1


class _Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...


def serves(model: torch.nn.Module) -> str:
    """Why the engine can't decode `model`, or "" where it can."""
    if not torch.cuda.is_available():
        return "the engine decodes on CUDA"
    try:
        text = text_of(model)
    except (AttributeError, KeyError, TypeError) as exc:
        return f"the engine decodes Qwen3.5's architecture: {exc}"
    if not text.attention or not text.delta:
        return "the engine decodes Qwen3.5's hybrid layers"
    if (text.heads // text.kv_heads) * text.kv_heads != text.heads:
        return "its query heads are not a multiple of its K/V heads"
    return ""


class CozyEngine:
    """The engine of a loaded Qwen3.5 checkpoint, with `TransformersEngine` beside it."""

    def __init__(self, pipe: Pipeline, processor: ProcessorMixin) -> None:
        from cozy_runtime.vlm import COMPONENT, TransformersEngine

        self.pipe, self.processor = pipe, processor
        self.fallback = TransformersEngine(pipe, processor)
        self.model = pipe.components[COMPONENT]
        self.decoder: Decoder | None = None

    def _why(self, questions: list[Question], params: Params) -> str:
        if any(not isinstance(p, str) for q in questions for p in q.read()):
            return "the engine's prefill reads text only (xs-213 phase 2)"
        if params.no_repeat_ngram:
            return "the engine has no repeat-n-gram cut"
        if len(questions) > 16:
            return "the engine decodes at most 16 rows at once"
        return serves(self.model)

    def generate(
        self,
        questions: list[Question],
        params: Params,
        *,
        on_delta: Callable[[Delta], None],
        progress: Callable[[int], None] | None = None,
    ) -> Final:
        from cozy_runtime.vlm import Final, Usage, _Told

        if self._why(questions, params):
            served = self.fallback.generate(questions, params, on_delta=on_delta, progress=progress)
            served.timings["engine_transformers"] = 1.0
            return served
        tokenizer = cast("_Tokenizer", self.processor.tokenizer)
        texts = [self.fallback._chat(q, params.system, params.thinking, params.reasoning_effort) for q in questions]
        prompts = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        configured = self.model.generation_config.eos_token_id
        stop = [] if params.thinking else params.stop
        single = [ids[0] for ids in (tokenizer.encode(s, add_special_tokens=False) for s in stop) if len(ids) == 1]
        stops = [t for t in dict.fromkeys([*(configured if isinstance(configured, list) else [configured]), tokenizer.eos_token_id, *single]) if t is not None]
        if self.decoder is None:
            self.decoder = Decoder(self.model)
        rows = len(prompts)
        made: list[list[int]] = [[] for _ in range(rows)]
        told = [_Told(params.thinking) for _ in range(rows)]
        last = [0.0]
        count = [0]

        def tell(row: int, ended: bool = False) -> None:
            text = tokenizer.decode(made[row], skip_special_tokens=True)
            for delta in told[row].more(row, text, ended):
                on_delta(delta)

        def tokens(row: int, new: list[int]) -> None:
            made[row].extend(new)
            count[0] += len(new)
            now = time.perf_counter()
            if now - last[0] >= STREAM_S:
                last[0] = now
                for r in range(rows):
                    tell(r)
                if progress is not None:
                    progress(count[0])

        request = Request(
            prompts=prompts, max_new=params.max_new_tokens, stops=stops, temperature=params.temperature,
            top_p=params.top_p, top_k=params.top_k, penalty=params.presence_penalty, seed=params.seed,
        )  # fmt: skip
        out = self.decoder(request, told=tokens)
        reasons: list[str] = []
        for row in range(rows):
            ids = out.tokens[row]
            end = next((i for i, t in enumerate(ids) if t in stops), None)
            made[row] = ids if end is None else ids[: end + 1]
            reasons.append("length" if end is None else "stop")
            tell(row, ended=True)
        return Final(
            texts=[t.answer for t in told],
            thoughts=[t.thought for t in told],
            finish_reasons=reasons,
            usage=Usage(sum(len(p) for p in prompts), sum(len(m) for m in made)),
            timings={"prefill": out.prefill_s, "decode": out.decode_s, "steps": float(out.steps), "captured": float(out.captured)},
        )
