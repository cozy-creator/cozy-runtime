"""`CozyEngine` (xs-213 phase 2): `vlm.Engine` on the engine's own decoder for the calls it can
hold, Transformers' engine for every other. Which serves is decided in one place: `_why` (the
platform, the model, the call), then the decoder's `Short` (its weights or K/V refused room at the
call's start, before anything is told). Once a token is told nothing falls back. Each call says
which engine served it, and why not this one, in the run record's `<model>.engine` note.

It serves a CUDA Qwen3.5 text call: text-only questions, at most `ROWS` rows, no repeat-n-gram cut,
on a dense model whose step linears are all of `LANES`, with a TensorFS that has the K/V cache, once
the model's kernel set has landed (`install`: built while it loads, never on a request). The
questions are read as Transformers' engine reads them (its chat template and tokenizer), each row's
text is handed on as it comes (at most every `STREAM_S`), and stops are cut as `VisionLM.ask` cuts
them.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Protocol, cast

import torch

from cozy_runtime.engine.decoder import Decoder, Request, Short, answered, short_of_room
from cozy_runtime.engine.launch import Run
from cozy_runtime.engine.qwen35 import text_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import ProcessorMixin

    from cozy_runtime.accel.kernels import Launch, Plan

    from cozy_runtime.vlm import Delta, Final, Params, Pipeline, Question

#: Text handed on at most this often, a row at a time.
STREAM_S = 0.1
#: The most rows a call decodes at once (the decoder's).
ROWS = 16
#: The encodings of the step's linears the engine's GPU cells have passed (the real 9B's teacher-
#: forced check and today's path on the same pod, sm86 and sm89); a lane joins when its cell
#: passes (q4_k_m's `ggml.q4_k` and `ggml.q6_k` next).
LANES = frozenset({"bf16", "ggml.q8_0", "int8.convrot"})
_DTYPES = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}


class _Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...


def serves(model: torch.nn.Module) -> str:
    """Why the engine can't decode `model`, or "" where it can."""
    from cozy_runtime.engine._kernels import TRITON_ABSENT

    if not torch.cuda.is_available():
        return "the engine decodes on CUDA"
    from tensorfs.varena import _runtime as varena

    if TRITON_ABSENT:
        return f"the engine's kernels need Triton: {TRITON_ABSENT}"
    if not hasattr(varena.Device, "cache"):
        return "this TensorFS has no K/V cache (varena's Device.cache)"
    try:
        text = text_of(model)
    except (AttributeError, KeyError, TypeError) as exc:
        return f"the engine decodes Qwen3.5's architecture: {exc}"
    if not text.attention or not text.delta:
        return "the engine decodes Qwen3.5's hybrid layers"
    if (text.heads // text.kv_heads) * text.kv_heads != text.heads:
        return "its query heads are not a multiple of its K/V heads"
    root = model.get_decoder()  # type: ignore[operator]  # a Transformers model's
    head = getattr(model, "mtp", None)
    layers = [*root.layers, *(head.layers if head is not None else [])]
    if not all(hasattr(layer.mlp, part) for layer in layers for part in ("gate_proj", "up_proj", "down_proj")):
        return "its MLPs are experts: the engine's step has none"
    reads = (root, model.get_output_embeddings(), head)  # type: ignore[operator]  # as above
    lanes = {_lane(m) for read in reads if read is not None for m in read.modules()} - {""}
    if unproven := sorted(lanes - LANES):
        return f"its linears are {', '.join(unproven)}, which the engine's cells have not passed"
    return ""


def _lane(module: torch.nn.Module) -> str:
    """A linear's encoding (an encoded leaf's), or its dtype's name (a plain one's); "" otherwise."""
    encoding = getattr(module, "encoding", None)
    if isinstance(encoding, str):
        return encoding
    if isinstance(module, torch.nn.Linear):
        return _DTYPES.get(module.weight.dtype, str(module.weight.dtype))
    return ""


class CozyEngine:
    """The engine of a loaded Qwen3.5 checkpoint, with `TransformersEngine` beside it."""

    def __init__(self, pipe: Pipeline, processor: ProcessorMixin) -> None:
        from cozy_runtime.vlm import COMPONENT, TransformersEngine

        self.pipe, self.processor = pipe, processor
        self.fallback = TransformersEngine(pipe, processor)
        self.model = pipe.components[COMPONENT]
        self.decoder: Decoder | None = None
        #: why the engine can't decode the model ("" where it can), judged as it installs
        self.unserved: str | None = None
        #: the model's kernel set as it launches, once it landed (`install`)
        self.run: Run | None = None
        #: that set's name, while it builds
        self.building = ""

    def install(self, plan: Plan) -> str:
        """The model's kernel set (`engine.kernels`), built while the model loads: the engine serves
        from when it lands (`plan.when_ready`), Transformers' engine until then, so a request never
        compiles one. Returns the set's identity, "" where the engine can't serve the model."""
        from cozy_runtime.accel.kernels import KernelsUnavailable
        from cozy_runtime.engine import kernels

        self.unserved = serves(self.model)
        if self.unserved:
            return ""
        try:
            shapes = kernels.shapes_of(self.model)
            made = kernels.kernels_for(shapes)
        except ValueError as exc:
            self.unserved = f"the engine's kernels: {exc}"
            return ""
        self.building = made.name
        print(f"[engine] {made.name}: {len(made.entries)} kernels for {shapes}", file=sys.stderr)

        def activate(launch: Launch) -> None:
            self.run = Run(launch, made)

        try:
            if plan.when_ready is None:
                activate(plan.load(made))
            else:
                plan.when_ready(made, activate)
        except KernelsUnavailable as exc:
            self.unserved = f"the engine's kernels ({made.name}): {exc.code}: {exc}"
            return ""
        return f"{made.name}:{made.identity}"

    def _why(self, questions: list[Question], params: Params) -> str:
        if self.unserved is None:
            self.unserved = serves(self.model) or "the engine's kernels were never installed (no execution plan)"
        if self.unserved:
            return self.unserved
        if self.run is None:
            return f"the engine's kernels ({self.building}) are building"
        if any(not isinstance(p, str) for q in questions for p in q.everything()):
            return "the engine's prefill reads text only"
        if params.allowed is not None:
            return "the engine has no grammar-constrained decode"
        if len(questions) > ROWS:
            return f"the engine decodes at most {ROWS} rows at once"
        if params.no_repeat_ngram:
            return "the engine has no repeat-n-gram cut"
        return ""

    def generate(
        self,
        questions: list[Question],
        params: Params,
        *,
        on_delta: Callable[[Delta], None],
        progress: Callable[[int], None] | None = None,
    ) -> Final:
        why = self._why(questions, params)
        if not why:
            try:
                return self._served(questions, params, on_delta, progress)
            except Short as short:
                from cozy_runtime.internal.execution import catch

                why = str(short)
                catch._unwind(short)
        self._note(f"Transformers' engine: {why}")
        return self.fallback.generate(questions, params, on_delta=on_delta, progress=progress)

    def _served(
        self,
        questions: list[Question],
        params: Params,
        on_delta: Callable[[Delta], None],
        progress: Callable[[int], None] | None,
    ) -> Final:
        from cozy_runtime.vlm import Final, Usage, _Told

        tokenizer = cast("_Tokenizer", self.processor.tokenizer)
        texts = [self.fallback._chat(q, params.system, params.thinking, params.reasoning_effort) for q in questions]
        prompts = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        configured = self.model.generation_config.eos_token_id
        stop = [] if params.thinking else params.stop
        single = [ids[0] for ids in (tokenizer.encode(s, add_special_tokens=False) for s in stop) if len(ids) == 1]
        stops = [t for t in dict.fromkeys([*(configured if isinstance(configured, list) else [configured]), tokenizer.eos_token_id, *single]) if t is not None]
        if self.decoder is None:
            assert self.run is not None  # `_why` serves no call before the set lands
            try:
                self.decoder = Decoder(self.model, max_rows=ROWS, run=self.run)
            except Exception as exc:
                if not short_of_room(exc):
                    raise
                answered(exc)
                raise Short(f"the engine's tables don't fit ({str(exc)[:200]})") from exc
        ceiling, longest = self.decoder.kv.ceiling, max(len(p) for p in prompts)
        if longest >= ceiling:
            raise Short(f"its prompt of {longest} tokens fills the {ceiling} a row holds (the model's context)")
        new = min(params.max_new_tokens, ceiling - longest)
        if new < params.max_new_tokens:
            from cozy_runtime.author._model import WARNINGS

            WARNINGS.append((
                f"max_tokens {params.max_new_tokens} runs past the model's context after a {longest}-token "
                f"prompt: clamped to {new}",
                {"max_tokens": params.max_new_tokens, "clamped": new, "context": ceiling},
            ))  # fmt: skip
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
            prompts=prompts, max_new=new, stops=stops, temperature=params.temperature,
            top_p=params.top_p, top_k=params.top_k, min_p=params.min_p, presence=params.presence_penalty,
            frequency=params.frequency_penalty, repetition=params.repetition_penalty, seed=params.seed,
        )  # fmt: skip
        out = self.decoder(request, told=tokens)
        reasons: list[str] = []
        for row in range(rows):
            ids = out.tokens[row]
            end = next((i for i, t in enumerate(ids) if t in stops), None)
            made[row] = ids if end is None else ids[: end + 1]
            reasons.append("length" if end is None else "stop")
            tell(row, ended=True)
        kept = [k for row in out.kept for k in row]
        made_tokens = sum(len(m) for m in made)
        self._note(
            f"{made_tokens} tokens after {sum(len(p) for p in prompts)} by the engine: prefill {out.prefill_s:.3f} s, "
            f"then {1000 * out.decode_s / max(out.steps, 1):.2f} ms a step over {out.steps} steps "
            f"({sum(kept) / max(len(kept), 1):.2f} tokens a step a row, {'captured' if out.captured else 'eager'})"
        )
        return Final(
            texts=[t.answer for t in told],
            thoughts=[t.thought for t in told],
            finish_reasons=reasons,
            usage=Usage(sum(len(p) for p in prompts), made_tokens),
            timings={"prefill": out.prefill_s, "decode": out.decode_s, "steps": float(out.steps), "captured": float(out.captured)},
        )

    def _note(self, text: str) -> None:
        """The call's engine in the run record, beside lm's `<model>.lm`, and the executor's log."""
        from cozy_runtime.author._model import PLACED

        print(f"[engine] {text}", file=sys.stderr)
        if (placed := PLACED.get(self.model)) is not None:
            placed[0].note(f"{placed[1]}.engine", text)
