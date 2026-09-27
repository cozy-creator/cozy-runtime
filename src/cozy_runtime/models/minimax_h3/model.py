"""Reusable MiniMax H3 models using Runtime's ordinary public Model contract."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, Literal

from cozy_runtime.author import Context, Loader, Model, sequence_parallel, uses_components

from .continuation import AVContext
from .official import (
    NumericalChecks,
    OfficialH3Pipeline,
    OfficialH3TurboLoRA,
    ScheduleFacts,
    Task,
    _validate_sol_dense_steps,
    build_h3_pipeline,
    build_h3_turbo_base,
    build_h3_turbo_lora,
)
from .turbo import ATTENTION_KWARG, OVERLAY_KWARG, TURBO_BANK


@sequence_parallel(degrees=(2, 4, 7, 8))
class H3Model(Model[OfficialH3Pipeline], encoded_leaves="accept", fusion="accept"):
    """H3 is head-shardable at every divisor of its 56 attention heads up to eight GPUs.

    The upstream `_cp_plan` shards the packed sequence itself with `tensor_split`, so any
    length splits and every block's GEMMs, norms, RoPE, AdaLN rows and SwiGLU run on the
    local rows; only attention exchanges, per head. `EncodedLinear` quantizes per TOKEN with
    rowwise `_scaled_mm` scales, so a row's numbers do not depend on which rank holds it; a
    per-tensor activation scale would be derived from the local shard and is refused by the
    runtime rather than served. Conditioning, sampling and both VAEs stay on the leader.

    Degrees 2 and 4 reproduce the degree-1 video and audio velocities to 1.2e-7 and 2.4e-7
    max absolute error on this pipeline's own AdaLN-pruned DiT (2026-09-08), which is
    float32 round-off. 7 and 8 match one rank on CPU Gloo ranks (`test_parallel_calls`) and
    have not yet run on GPUs (h3a-019).
    """

    pipe: OfficialH3Pipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(OfficialH3Pipeline, factory=build_h3_pipeline)

    def unload(self, loader: Loader) -> None:
        return None

    def warm(self, ctx: Context) -> None:
        """One dry DiT forward per entrypoint DiT this device ADMITS, before serving.

        Both DiTs are offered, because one construction carries both entrypoints and a
        switch between them must not pay a first call either (h3a-018). The runtime has
        already applied the fused glue and loaded its cubins for this device by the time
        `warm` runs (h3a-015, `fusion="accept"` above); the dry forward is what pays their
        first launches, the rotary tables and the projections' first GEMM plans.

        Offered, not required. Warm uses the same staged component-use admission as a
        request: unrelated completed components are evicted before each DiT runs. A
        parked DiT can therefore warm if its own scope fits. A measured device_shortfall
        remains optional and is recorded and skipped; it is not a requirement to fit
        both DiTs simultaneously. Later requests select their own placement rung from
        the actual post-warm resident set. Warming a parked DiT may add checkpoint I/O.
        """
        warmers: dict[Task, Callable[[], None]] = {
            "fl2va": self.warm_fl2va,
            "ref2va": self.warm_ref2va,
        }
        for warm_one in warmers.values():
            ctx.raise_if_cancelled()
            try:
                warm_one()
            except Exception as exc:
                if getattr(exc, "code", "") != "device_shortfall":
                    raise
                print(
                    f"[minimax-h3] {warm_one.__name__} not applied: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    @uses_components("fl2va_dit")
    def warm_fl2va(self) -> None:
        self.pipe.warm_dit("fl2va")

    @uses_components("ref2va_dit")
    def warm_ref2va(self) -> None:
        self.pipe.warm_dit("ref2va")

    @uses_components("text_encoder")
    def condition_text(self, task: Task, state: Any, *, checks: NumericalChecks) -> None:
        checks.component("text_encoder", self.pipe.components["text_encoder"])
        self.pipe.condition_text(task, state, checks=checks)

    @uses_components("audio_vae")
    def decode_audio(
        self, task: Task, state: Any, *, checks: NumericalChecks | None = None
    ) -> tuple[Any, int]:
        if checks is not None:
            checks.component("audio_vae", self.pipe.components["audio_vae"])
        audio = self.pipe.decode_audio(task, state)
        if checks is not None:
            checks.tensors("decode_audio", [("audio", audio)])
        return audio, int(state.sampling_rate)

    @uses_components("video_vae")
    def decode_video(
        self,
        task: Task,
        state: Any,
        *,
        on_chunk: Callable[[Any], None],
        checks: NumericalChecks | None = None,
    ) -> int:
        """Hand every decoded temporal chunk to `on_chunk` inside the VAE's component scope;
        a generator would run its body after the scope had closed."""
        if checks is not None:
            checks.component("video_vae", self.pipe.components["video_vae"])

        def observed(chunk: Any) -> None:
            if checks is not None:
                checks.tensors("decode_video", [("video", chunk)])
            on_chunk(chunk)

        return self.pipe.decode_video_chunks(task, state, observed)

    @uses_components("video_vae", "audio_vae")
    def export_completed_av_tail(self, state: Any, *, provenance: str) -> AVContext:
        """Re-encode the delivered tail while both native VAE components are resident."""
        return self.pipe.export_completed_av_tail(state, provenance=provenance)

    @uses_components("video_vae")
    def condition_fl2va_media(self, task: Task, state: Any, *, checks: NumericalChecks) -> None:
        checks.component("video_vae", self.pipe.components["video_vae"])
        self.pipe.condition_media(task, state, checks=checks)

    @uses_components("video_vae", "audio_vae")
    def condition_ref2va_media(self, task: Task, state: Any, *, checks: NumericalChecks) -> None:
        for name in ("video_vae", "audio_vae"):
            checks.component(name, self.pipe.components[name])
        self.pipe.condition_media(task, state, checks=checks)

    @uses_components("fl2va_dit")
    def sample_fl2va(
        self, state: Any, *, on_step: Any, cancel: Any, checks: NumericalChecks
    ) -> ScheduleFacts:
        root = self.pipe.components["fl2va_dit"]
        checks.component("fl2va_dit", root)
        with checks.forwards(root, "fl2va_dit"):
            return self.pipe.denoise("fl2va", state, on_step=on_step, cancel=cancel, checks=checks)

    @uses_components("ref2va_dit")
    def sample_ref2va(
        self, state: Any, *, on_step: Any, cancel: Any, checks: NumericalChecks
    ) -> ScheduleFacts:
        root = self.pipe.components["ref2va_dit"]
        checks.component("ref2va_dit", root)
        with checks.forwards(root, "ref2va_dit"):
            return self.pipe.denoise("ref2va", state, on_step=on_step, cancel=cancel, checks=checks)


class H3TurboBase(H3Model, encoded_leaves="accept", fusion="accept"):
    """The five-root base checkpoint with construction-time turbo consumers.

    ``sol_dense_steps`` counts initial transformer evaluations, not scheduler
    sigma points or CFG branches. The default 4 keeps evaluations 1-4 dense and
    allows sparse video attention in 5-8 when Sol is explicitly selected. Callers
    can still request 3, 8 or 10; no value selects Sol, changes its protected prefix,
    or removes the always-dense refiner and first two transformer blocks.
    """

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(OfficialH3Pipeline, factory=build_h3_turbo_base)

    @uses_components("fl2va_dit")
    def sample_fl2va_turbo(
        self,
        state: Any,
        *,
        turbo_lora: H3TurboLoRA,
        on_step: Any,
        cancel: Any,
        checks: NumericalChecks,
        sol_dense_steps: int = 4,
    ) -> ScheduleFacts:
        return turbo_lora.sample_fl2va(
            self,
            state,
            on_step=on_step,
            cancel=cancel,
            checks=checks,
            sol_dense_steps=sol_dense_steps,
        )

    @uses_components("ref2va_dit")
    def sample_ref2va_turbo(
        self,
        state: Any,
        *,
        turbo_lora: H3TurboLoRA,
        on_step: Any,
        cancel: Any,
        checks: NumericalChecks,
        sol_dense_steps: int = 4,
    ) -> ScheduleFacts:
        return turbo_lora.sample_ref2va(
            self,
            state,
            on_step=on_step,
            cancel=cancel,
            checks=checks,
            sol_dense_steps=sol_dense_steps,
        )


@sequence_parallel(degrees=(2, 4, 7, 8))
class H3TurboLoRA(Model[OfficialH3TurboLoRA], encoded_leaves="accept"):
    """Independent PDD weights, replicated alongside the base on every CP rank."""

    pipe: OfficialH3TurboLoRA

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(OfficialH3TurboLoRA, factory=build_h3_turbo_lora)

    def unload(self, loader: Loader) -> None:
        return None

    @uses_components("fl2va_turbo")
    def sample_fl2va(
        self,
        base: H3Model,
        state: Any,
        *,
        on_step: Any,
        cancel: Any,
        checks: NumericalChecks,
        sol_dense_steps: int = 4,
    ) -> ScheduleFacts:
        return self._sample(
            base,
            "fl2va",
            state,
            on_step=on_step,
            cancel=cancel,
            checks=checks,
            sol_dense_steps=sol_dense_steps,
        )

    @uses_components("ref2va_turbo")
    def sample_ref2va(
        self,
        base: H3Model,
        state: Any,
        *,
        on_step: Any,
        cancel: Any,
        checks: NumericalChecks,
        sol_dense_steps: int = 4,
    ) -> ScheduleFacts:
        return self._sample(
            base,
            "ref2va",
            state,
            on_step=on_step,
            cancel=cancel,
            checks=checks,
            sol_dense_steps=sol_dense_steps,
        )

    def _sample(
        self,
        base: H3Model,
        trunk: Literal["fl2va", "ref2va"],
        state: Any,
        *,
        on_step: Any,
        cancel: Any,
        checks: NumericalChecks,
        sol_dense_steps: int,
    ) -> ScheduleFacts:
        _validate_sol_dense_steps(sol_dense_steps)
        task: Task = "fl2va_turbo" if trunk == "fl2va" else "ref2va_turbo"
        overlay = self.pipe.overlay(base.pipe, trunk)
        root = base.pipe.components[f"{trunk}_dit"]
        checks.component(f"{trunk}_dit", root)
        checks.component(f"{trunk}_turbo", overlay)
        original = state.get("attention_kwargs")
        state.set(
            "attention_kwargs",
            {
                **(original or {}),
                ATTENTION_KWARG: TURBO_BANK,
                OVERLAY_KWARG: overlay,
            },
        )
        try:
            with checks.forwards(root, f"{trunk}_dit"):
                return base.pipe.denoise(
                    task,
                    state,
                    on_step=on_step,
                    cancel=cancel,
                    checks=checks,
                    sol_dense_steps=sol_dense_steps,
                )
        finally:
            state.set("attention_kwargs", original)
