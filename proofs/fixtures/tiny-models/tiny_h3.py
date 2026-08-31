"""The component-staged package — MiniMax H3 shaped, weightless but real.

Modeled on `cozy-runtime-h3.md`: three typed request/result structs, an ordered
discriminated reference union, plain orchestration functions (module code — they touch no
components), ONE shared implementation base with TWO thin task-stamped operation classes,
and two entrypoints. The pipeline comes from `tiny_diffusers`; everything else is the
production author surface.

What this fixture proves that SDXL does not:
  · two thin classes over one base, each declaring ONLY its own transformer — which is what
    lets a NARROW artifact validate while a dual artifact binds from both classes;
  · role-scoped factories: the class, its factory role, the artifact stamp and the derived
    construction record must all agree;
  · granular `@uses_components` per semantic operation, and a private helper inheriting its
    caller's active scope;
  · a declared `DerivedCache` whose entries are built INSIDE the transformer's scope, on a
    full topology only — a baked construction reads checkpoint components instead;
  · fresh per-request video and audio schedulers from `view.make_scheduler`.

Video/audio ENCODING is cr-017's: `out.save_video` exists and refuses typed, so this
fixture returns a poster frame plus the decoded counts. Nothing here is stubbed silently.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, TypeAlias

import msgspec
from tiny_diffusers import MiniMaxH3ModularPipeline, build_h3_pipeline

from cozy_runtime.author import (
    App,
    AudioAsset,
    Budget,
    Checkpoints,
    Context,
    DerivedCache,
    FileAsset,
    ImageAsset,
    ImageFrame,
    InvalidRequest,
    Loader,
    Model,
    ModelDefault,
    Outputs,
    Preflight,
    ProductionContract,
    ProductionNode,
    ProductionOutput,
    RequestView,
    Resources,
    Scheduler,
    Scratch,
    Settings,
    Shape,
    Telemetry,
    Tree,
    UnsupportedInput,
    VideoAsset,
    uses_components,
)

app = App()

H3Task: TypeAlias = Literal["fl2va", "ref2va"]
StepPreset = Literal[2, 4, 8]
_FPS = 24


def _frames_for(seconds: int) -> int:
    """The smallest 17n+5 frame count covering the duration — the model's own grid."""
    n = -(-(seconds * _FPS - 5) // 17)
    return 17 * n + 5


DurationS = Literal[1, 2, 3]
_FRAMES = {d: _frames_for(d) for d in (1, 2, 3)}


class AspectPreset(Enum):
    W16_9 = "16:9"
    SQUARE = "1:1"
    T9_16 = "9:16"


# THE ONE CANONICAL FAMILY TABLE — it feeds the request schema (Shape), demand features and
# geometry alike; a copied number elsewhere is never a second authority.
_PIXELS = {
    AspectPreset.W16_9: (84, 48),
    AspectPreset.SQUARE: (64, 64),
    AspectPreset.T9_16: (48, 84),
}


class ImageReference(msgspec.Struct, tag="image", tag_field="kind"):
    image: ImageAsset


class VideoReference(msgspec.Struct, tag="video", tag_field="kind"):
    video: VideoAsset
    soundtrack: bool = False  # container metadata, never bytes — preflight may read it
    fps: float | None = None  # override incorrect/missing container metadata
    sample_rate: int | None = None


class AudioReference(msgspec.Struct, tag="audio", tag_field="kind"):
    audio: AudioAsset
    sample_rate: int | None = None


#: ORDER IS SEMANTIC — never three modality-grouped lists: it controls the presentation's
#: <Picture i>/<Video i>/<Audio i> labels AND the shared rotary clock.
Reference: TypeAlias = ImageReference | VideoReference | AudioReference


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    first_frame: ImageAsset | None = None  # when present, fixes output geometry
    last_frame: ImageAsset | None = None
    aspect_ratio: Annotated[AspectPreset, Shape(pixels=_PIXELS)] = AspectPreset.W16_9
    duration_s: Annotated[DurationS, Shape(frames=_FRAMES)] = 1
    num_inference_steps: ModelDefault[StepPreset] = 4
    mute: bool = False  # an output toggle, never a speed win
    seed: int | None = None


class RefGenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    references: Annotated[list[Reference], msgspec.Meta(min_length=1, max_length=12)]
    # target-clock anchors, never reference images; the combined door is gated in preflight
    first_frame: ImageAsset | None = None
    last_frame: ImageAsset | None = None
    aspect_ratio: Annotated[AspectPreset, Shape(pixels=_PIXELS)] = AspectPreset.W16_9
    duration_s: Annotated[DurationS, Shape(frames=_FRAMES)] = 1
    num_inference_steps: ModelDefault[StepPreset] = 4
    mute: bool = False
    seed: int | None = None


class GenerateOutput(msgspec.Struct):
    poster: ImageAsset  # cr-017 owns the A/V mux: save_video(frames, audio=) is ONE mp4
    frames: int
    audio_samples: int
    checkpoint: str


class RefServeSettings(msgspec.Struct):
    """Deployment-owned toggle: keyframes+references is output-UNVERIFIED (se-001)."""

    combined_keyframes: bool = False


# ------------------------------------------------------------------ plain orchestration
# Pure orchestration touches no components, so it is module code, never Model methods.


@dataclass(frozen=True, slots=True)
class PreparedReference:
    """Both legs of ONE ordered reference: a soundtracked video contributes two
    presentation segments (its <Audio i> label precedes <Video i>) and two row blocks."""

    index: int
    visual: ImageAsset | VideoAsset | None
    audio: AudioAsset | VideoAsset | None
    has_audio: bool
    fps: float | None = None
    sample_rate: int | None = None

    @classmethod
    def from_reference(cls, ref: Reference, *, index: int) -> PreparedReference:
        if isinstance(ref, ImageReference):
            return cls(index, ref.image, None, has_audio=False)
        if isinstance(ref, VideoReference):
            return cls(
                index,
                ref.video,
                ref.video if ref.soundtrack else None,
                has_audio=ref.soundtrack,
                fps=ref.fps,
                sample_rate=ref.sample_rate,
            )
        return cls(index, None, ref.audio, has_audio=True, sample_rate=ref.sample_rate)


@dataclass(frozen=True, slots=True)
class Keyframe:
    image: ImageAsset
    clock: int  # a target-clock anchor: first or last frame of the GENERATED clip


@dataclass(frozen=True, slots=True)
class LatentGrid:
    width: int
    height: int
    frames: int


@dataclass(frozen=True, slots=True)
class H3Plan:
    prompt: str
    references: tuple[PreparedReference, ...]
    keyframes: tuple[Keyframe, ...]
    frames: int
    grid: LatentGrid


@dataclass(frozen=True, slots=True)
class PackedLayout:
    """Per-row offsets, modality tags and rotary time. Offsets need the ENCODED geometries —
    semantic order alone cannot place rows."""

    rows: tuple[tuple[str, int], ...]

    @classmethod
    def build(cls, plan: H3Plan, *, visual: Sequence[int], audio: Sequence[int]) -> PackedLayout:
        rows = [("visual", v) for v in visual] + [("audio", a) for a in audio]
        return cls((*rows, ("target", plan.frames)))


@dataclass(frozen=True, slots=True)
class H3Conditioning:
    prompt_embeds: int
    text_token_tags: int
    visual_rows: tuple[int, ...]
    audio_rows: tuple[int, ...]
    packed_layout: PackedLayout


@dataclass(frozen=True, slots=True)
class H3Latents:
    video: int
    audio: int


_TABLE_FORMAT = "adaln-tables/1"  # table-format + baker-numerics version


@dataclass(frozen=True, slots=True)
class TimestepPlan:
    """The EXACT plan admission keys on. `steps=4` alone is display metadata (§1.1.1)."""

    steps: tuple[tuple[float, float], ...]
    digest: str

    @classmethod
    def derive(
        cls,
        *,
        steps: int,
        task: H3Task,
        has_visual: bool,
        has_audio: bool,
        video: Scheduler,
        audio: Scheduler,
    ) -> TimestepPlan:
        v = video.set_timesteps(steps) if hasattr(video, "set_timesteps") else ()
        a = audio.set_timesteps(steps) if hasattr(audio, "set_timesteps") else ()
        pairs = tuple(zip(v, a, strict=True))
        subject = (
            f"{task}|{has_visual}|{has_audio}|{pairs}|{video.config}|{audio.config}|{_TABLE_FORMAT}"
        )
        return cls(pairs, hashlib.blake2b(subject.encode(), digest_size=12).hexdigest())


@dataclass(frozen=True, slots=True)
class ModulationKey:
    """Never the bare plan digest: the exact snapshot, role, plan and adapter set."""

    checkpoint: str
    task: H3Task
    plan_digest: str
    adapters: tuple[tuple[str, float], ...]  # conservative: ANY adapter invalidates
    dtype: str = "bf16"
    table_format: str = _TABLE_FORMAT


@dataclass(frozen=True, slots=True)
class H3Run:
    """Every mutable execution object dies with this request (§1.1.2)."""

    view: RequestView
    task: H3Task
    timestep_plan: TimestepPlan
    modulation_key: ModulationKey
    video_scheduler: Scheduler
    audio_scheduler: Scheduler


_REF_CAPS: dict[type, int] = {ImageReference: 9, VideoReference: 3, AudioReference: 3}


def validate_references(refs: Sequence[Reference]) -> None:
    """The wire caps the list at 12; per-kind caps are semantic — typed and pre-model."""
    for kind, cap in _REF_CAPS.items():
        n = sum(isinstance(r, kind) for r in refs)
        if n > cap:
            raise InvalidRequest(
                f"{kind.__name__}: {n} > {cap}",
                code="too_many_references",
                fields=["references"],
            )


def prepare(payload: GenerateInput | RefGenerateInput, *, frames: int) -> H3Plan:
    refs: tuple[PreparedReference, ...] = ()
    if isinstance(payload, RefGenerateInput):
        refs = tuple(
            PreparedReference.from_reference(r, index=i) for i, r in enumerate(payload.references)
        )
    keys: list[Keyframe] = []
    if payload.first_frame is not None:
        keys.append(Keyframe(payload.first_frame, clock=0))
    if payload.last_frame is not None:
        keys.append(Keyframe(payload.last_frame, clock=frames - 1))
    width, height = _PIXELS[payload.aspect_ratio]
    return H3Plan(
        prompt=payload.prompt,
        references=refs,
        keyframes=tuple(keys),
        frames=frames,
        grid=LatentGrid(width, height, frames),
    )


def has_audio_conditions(plan: H3Plan) -> bool:
    """Audio references AND video-reference soundtracks — a silent video is visual-only."""
    return any(r.has_audio for r in plan.references)


def pack(
    plan: H3Plan, text: tuple[int, int], visual: tuple[int, ...], audio: tuple[int, ...]
) -> H3Conditioning:
    return H3Conditioning(
        prompt_embeds=text[0],
        text_token_tags=text[1],
        visual_rows=visual,
        audio_rows=audio,
        packed_layout=PackedLayout.build(plan, visual=visual, audio=audio),
    )


def start_run(
    view: RequestView, plan: H3Plan, *, task: H3Task, steps: StepPreset, checkpoint: str
) -> H3Run:
    video = view.make_scheduler("video_scheduler")  # fresh mutable instances —
    audio = view.make_scheduler("audio_scheduler")  #   they die with the request
    tplan = TimestepPlan.derive(
        steps=steps,
        task=task,
        has_visual=bool(plan.references or plan.keyframes),
        has_audio=has_audio_conditions(plan),
        video=video,
        audio=audio,
    )
    key = ModulationKey(
        checkpoint=checkpoint,
        task=task,
        plan_digest=tplan.digest,
        adapters=tuple((a.ref, a.scale) for a in view.adapters),
    )
    return H3Run(view, task, tplan, key, video, audio)


class H3AdmissionFacts(msgspec.Struct, frozen=True):
    """The frozen preflight record: ONLY the cross-field facts derived RequestFeatures
    cannot express. Frames/pixels/steps are already derived from the declared Shape tables,
    and a hook field duplicating a derivable feature refuses at describe."""

    has_visual_condition: bool
    has_audio_condition: bool


def preflight_fl(payload: GenerateInput) -> H3AdmissionFacts:
    """Pure, metadata-only: no Context, Model or bytes. A refusal here happens BEFORE
    hydration and before any GPU work."""
    return H3AdmissionFacts(
        has_visual_condition=(payload.first_frame is not None or payload.last_frame is not None),
        has_audio_condition=False,
    )


def preflight_ref(payload: RefGenerateInput, settings: RefServeSettings) -> H3AdmissionFacts:
    validate_references(payload.references)  # counts and kinds only, never bytes
    if (
        payload.first_frame is not None or payload.last_frame is not None
    ) and not settings.combined_keyframes:
        # The combined door ships DISABLED until output verification (se-001). Refusing
        # HERE means a refused request never hydrates an asset.
        raise UnsupportedInput(
            "keyframes with references is not enabled on this deployment",
            code="combined_conditioning_unverified",
            fields=["first_frame", "last_frame"],
        )
    prepared = tuple(
        PreparedReference.from_reference(r, index=i) for i, r in enumerate(payload.references)
    )
    return H3AdmissionFacts(
        has_visual_condition=(
            any(p.visual is not None for p in prepared)
            or payload.first_frame is not None
            or payload.last_frame is not None
        ),
        has_audio_condition=any(p.has_audio for p in prepared),
    )


# ------------------------------------------------------------------ modulation source
# Exactly ONE of the two routes exists per bound topology (§1.1.1) — a CONSTRUCTION fact,
# never a request-time mode.


@dataclass(frozen=True, slots=True)
class LiveAdaLN:
    """Full topology: the projections are part of the leased transformer, so a miss builds
    under that SAME lease and caches the immutable table."""

    cache: DerivedCache
    pipe: MiniMaxH3ModularPipeline

    def for_run(self, run: H3Run) -> tuple[int, ...]:
        return self.cache.get(
            run.modulation_key,
            build=lambda: self.pipe.blocks.build_modulation_tables(run.timestep_plan.digest),
        )


@dataclass(frozen=True, slots=True)
class BakedModulationTables:
    """Baked topology: the tables are ordinary checkpoint components and coverage was proven
    at admission — never through the cache."""

    pipe: MiniMaxH3ModularPipeline

    def for_run(self, run: H3Run) -> tuple[int, ...]:
        return self.pipe.blocks.baked_tables(run.timestep_plan.digest)


ModulationSource: TypeAlias = LiveAdaLN | BakedModulationTables


# ------------------------------------------------------------------ the model boundary


class _H3Base(Model[MiniMaxH3ModularPipeline]):
    """One shared implementation base. The thin classes below add only their transformer."""

    pipe: MiniMaxH3ModularPipeline  # declared state — mypy-strict checked (§1.7)
    modulation: ModulationSource

    def _load(self, loader: Loader, *, task: H3Task) -> None:
        # ROLE-SCOPED factory: the constructed graph contains exactly the components this
        # class declares — a stamp constrains ADMISSION, never factory behavior (§1.1).
        self.pipe = loader.construct(
            MiniMaxH3ModularPipeline,
            factory=lambda config: build_h3_pipeline(config, task=task),
        )
        self.modulation = (
            BakedModulationTables(self.pipe)
            if self.pipe.topology.baked
            else LiveAdaLN(loader.cache("adaln_modulation_tables"), self.pipe)
        )

    def unload(self, loader: Loader) -> None:
        del self.modulation, self.pipe

    @uses_components("text_encoder")
    def encode_presentation(self, plan: H3Plan) -> tuple[int, int]:
        presentation = self.pipe.blocks.build_presentation(
            plan.prompt, plan.references, plan.keyframes
        )
        return self.pipe.blocks.text_encoder(presentation)

    @uses_components("visual_vae")
    def encode_visual_conditions(self, plan: H3Plan) -> tuple[int, ...]:
        refs = tuple(
            self.pipe.blocks.encode_visual(r.visual, fps=r.fps)
            for r in plan.references
            if r.visual is not None
        )
        keys = tuple(self.pipe.blocks.encode_visual(k.image, clock=k.clock) for k in plan.keyframes)
        return refs + keys

    @uses_components("audio_vae")
    def encode_audio_conditions(self, plan: H3Plan) -> tuple[int, ...]:
        return tuple(
            self.pipe.blocks.encode_audio(r.audio, sample_rate=r.sample_rate)
            for r in plan.references
            if r.has_audio
        )

    def _denoise(
        self,
        run: H3Run,
        cond: H3Conditioning,
        transformer: str,
        *,
        frames: int,
        on_step: Callable[[int], None] | None,
    ) -> H3Latents:
        """A private helper: it INHERITS its caller's active scope (§1.1)."""
        latents = self.pipe.blocks.prepare_latents(frames=frames, generator=run.view.generator)
        # Exactly one source exists per construction: LiveAdaLN builds under THIS
        # transformer lease; BakedModulationTables reads checkpoint components.
        mod = self.modulation.for_run(run)
        for i, (t_v, t_a) in enumerate(run.timestep_plan.steps):
            latents = self.pipe.blocks.step(
                transformer, latents, cond.packed_layout, t_v, t_a, mod[i % len(mod)]
            )
            if on_step:
                on_step(i)
        return H3Latents(video=latents, audio=latents ^ 0xA5A5)

    @uses_components("visual_vae")
    def decode_video(self, latents: H3Latents) -> int:
        return self.pipe.blocks.decode_video(latents.video)

    @uses_components("audio_vae")
    def decode_audio(self, latents: H3Latents) -> tuple[int, int]:
        return self.pipe.blocks.decode_audio(latents.audio)


# TWO THIN CLASSES, not a task argument and not one class carrying both methods: each
# declares only the transformer its construction contains, so a NARROW artifact validates
# and a DUAL artifact binds from both classes. Instances are always separate (construction
# identity includes the class); shared component BYTES dedup in the resident registry.
# `task=` joins the checkpoint's VERIFIED stamps, fail-closed, stated once per role.


class Fl2VAModel(_H3Base, task="fl2va"):
    def load(self, loader: Loader) -> None:
        self._load(loader, task="fl2va")  # class, factory role, stamp and record must agree

    @uses_components("fl2va_transformer")
    def denoise(
        self,
        run: H3Run,
        cond: H3Conditioning,
        *,
        frames: int,
        on_step: Callable[[int], None] | None = None,
    ) -> H3Latents:
        return self._denoise(run, cond, "fl2va_transformer", frames=frames, on_step=on_step)


class Ref2VAModel(_H3Base, task="ref2va"):
    def load(self, loader: Loader) -> None:
        self._load(loader, task="ref2va")

    @uses_components("ref2va_transformer")
    def denoise(
        self,
        run: H3Run,
        cond: H3Conditioning,
        *,
        frames: int,
        on_step: Callable[[int], None] | None = None,
    ) -> H3Latents:
        return self._denoise(run, cond, "ref2va_transformer", frames=frames, on_step=on_step)


def _poster(grid: LatentGrid, pixels: int) -> ImageFrame:
    """Pure packing — cr-017 owns the real video encoder and the A/V mux (§3.4)."""
    rows = bytearray()
    for y in range(grid.height):
        for x in range(grid.width):
            rows += bytes(((x * 5 + pixels) & 0xFF, (y * 3 + pixels) & 0xFF, pixels & 0xFF))
    return ImageFrame(grid.width, grid.height, bytes(rows))


@app.entrypoint(preflight=preflight_fl)
def generate(
    ctx: Context,
    payload: GenerateInput,
    facts: Preflight[H3AdmissionFacts],
    model: Fl2VAModel,
    out: Outputs,
    tel: Telemetry,
) -> GenerateOutput:
    ctx.raise_if_cancelled()
    del facts  # admission already keyed on it; the handler re-validates nothing

    steps = payload.num_inference_steps
    frames = _FRAMES[payload.duration_s]
    plan = prepare(payload, frames=frames)

    text = model.encode_presentation(plan)
    visual = model.encode_visual_conditions(plan)
    audio: tuple[int, ...] = ()
    if has_audio_conditions(plan):
        audio = model.encode_audio_conditions(plan)
    condition = pack(plan, text, visual, audio)

    view = model.for_request(ctx, seed=payload.seed)
    run = start_run(view, plan, task="fl2va", steps=steps, checkpoint=model.checkpoint_ref)
    latents = model.denoise(
        run, condition, frames=frames, on_step=tel.step_callback(steps, stage="denoise")
    )

    waveform, sample_rate = model.decode_audio(latents)  # small decode first — peak residency
    pixels = model.decode_video(latents)
    return GenerateOutput(
        poster=out.save_image(_poster(plan.grid, pixels), format="png"),
        frames=frames,
        audio_samples=0 if payload.mute else (waveform & 0xFFFF) * (sample_rate // 48_000),
        checkpoint=model.checkpoint_ref,
    )


@app.entrypoint(preflight=preflight_ref)
def generate_ref(
    ctx: Context,
    payload: RefGenerateInput,
    facts: Preflight[H3AdmissionFacts],
    model: Ref2VAModel,
    out: Outputs,
    tel: Telemetry,
    settings: Settings[RefServeSettings],
) -> GenerateOutput:
    ctx.raise_if_cancelled()
    del facts, settings  # the caps and the combined gate ran in preflight_ref

    steps = payload.num_inference_steps
    frames = _FRAMES[payload.duration_s]
    plan = prepare(payload, frames=frames)

    text = model.encode_presentation(plan)
    visual = model.encode_visual_conditions(plan)
    audio: tuple[int, ...] = ()
    if has_audio_conditions(plan):  # audio references reach the same flow
        audio = model.encode_audio_conditions(plan)
    condition = pack(plan, text, visual, audio)

    view = model.for_request(ctx, seed=payload.seed)
    run = start_run(view, plan, task="ref2va", steps=steps, checkpoint=model.checkpoint_ref)
    latents = model.denoise(
        run, condition, frames=frames, on_step=tel.step_callback(steps, stage="denoise")
    )

    waveform, sample_rate = model.decode_audio(latents)
    pixels = model.decode_video(latents)
    return GenerateOutput(
        poster=out.save_image(_poster(plan.grid, pixels), format="png"),
        frames=frames,
        audio_samples=0 if payload.mute else (waveform & 0xFFFF) * (sample_rate // 48_000),
        checkpoint=model.checkpoint_ref,
    )


# ---------------------------------------------------------------------------- the job
# Structurally separate from serving — one immutable build, one bounded attempt, terminal,
# reclaim — but the SAME kernel and the same one service set (§2). What makes this a job is
# `@app.job` plus which services the signature requests, never a second context class.


class DistillInput(msgspec.Struct, forbid_unknown_fields=True):
    steps: Annotated[int, msgspec.Meta(ge=1, le=10_000)]
    learning_rate: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1e-4
    resume_from: Tree | None = None
    """Just another verified input tree — not a reserved payload name (§2). Its PATH rides
    the field value, so there is no context accessor to reach a checkpoint through."""


class DistillResult(msgspec.Struct):
    steps_run: int
    resumed_at: int
    final_loss: float
    receipt: str
    manifest: FileAsset


@app.job(
    publishes=True, emits_media=False, resources=Resources(gpu_count=2, placement="single_node")
)
def distill(
    ctx: Context,
    payload: DistillInput,
    base: Fl2VAModel,
    scratch: Scratch,
    ckpt: Checkpoints,
    budget: Budget,
    out: Outputs,
    tel: Telemetry,
) -> DistillResult:
    """`gpu_count` is the override door's canonical case: a structural FLOOR neither code
    nor measurement can discover pre-run. Everything else about this job is derived.

    The loop is the shape every training job has: resumable scratch, a durable checkpoint
    per interval, a spend check, and a run-to-completion return. Nothing here serves.
    """
    state = scratch.checkpoint_dir(key="distill")
    marker = state / "step"
    start = int(marker.read_text()) if marker.is_file() else 0
    loss = 1.0 - payload.learning_rate * start
    for step in range(start, payload.steps):
        ctx.raise_if_cancelled()
        budget.raise_if_exhausted()
        loss *= 1.0 - payload.learning_rate
        tel.training_metric("loss", loss, step=step)
        marker.write_text(str(step + 1))
    receipt = ckpt.declare(
        "final", f"{loss:.9f}".encode(), operation_key=f"distill/{payload.steps}"
    )
    return DistillResult(
        steps_run=payload.steps - start,
        resumed_at=start,
        final_loss=loss,
        receipt=receipt.declaration_id,
        manifest=out.save_bytes(
            f"{base.checkpoint_ref} {payload.steps} {loss:.9f}\n".encode(),
            media_type="text/plain",
        ),
    )


# Descriptor-only orchestration over an ordinary installed job. The fixture deliberately
# keeps this separate from ``distill``: registering the graph executes nothing.
app.production(
    "four-lane",
    source="source",
    nodes=(
        ProductionNode(
            name="distill-full",
            callable="tensorhub/h3/distill",
            models={"base": "source"},
            outputs=("bf16-full",),
            assets={"plan": "plans/full.json"},
            resources=Resources(gpu_count=2, placement="single_node", requires="ram64g"),
        ),
    ),
    outputs=(
        ProductionOutput(
            name="bf16-full",
            source="distill-full.bf16-full",
            lane_key="bf16-full",
            required=ProductionContract(
                tasks=("fl2va", "ref2va"),
                structure="h3-full",
                encodings=("plain",),
                stamps={"family": "h3"},
            ),
        ),
    ),
)
