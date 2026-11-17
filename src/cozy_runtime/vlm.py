"""Qwen-VL vision-language models served from their as-is upload: one request (`AskRequest`), one
result (`AskResult`), the model class (`VisionLM`) and the call (`ask`) a package's entrypoint
returns; and a box of words (`TextBox`, `TextResult`) a tool writes (`write`). Qwen3.5 and Qwen3.6,
dense and mixture-of-experts, load through it.

    @app.entrypoint(defaults={"model": "qwen3.6-35b-a3b@0.1.0"}, memoize=True)
    def ask(ctx: Context, payload: AskRequest, model: VisionLM, tel: Telemetry) -> AskResult:
        return vlm.ask(ctx, payload, model, tel)

    @app.tool(label="Write", named="text", defaults={"model": ...}, memoize=True)
    def write(ctx: Context, payload: TextBox, model: VisionLM, tel: Telemetry) -> TextResult:
        return vlm.write(ctx, payload, model, tel)

The construction config is the repository's `config.json`; the tokenizer, processor and chat
template are the checkpoint's assets; parameters load in their stored dtype. Fused experts are
split into per-expert blocks (`moe.split`): the checkpoint stores them per expert.
"""

from __future__ import annotations

import json
import secrets
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

import msgspec
import torch
import transformers
from PIL import Image as PILImage
from PIL.Image import Image
from tokenizers import Tokenizer
from transformers import (
    AddedToken,
    AutoConfig,
    AutoModelForImageTextToText,
    GenerationConfig,
    PreTrainedTokenizerFast,
)
from transformers.initialization import no_init_weights
from transformers.video_utils import VideoMetadata

from cozy_runtime import lm
from cozy_runtime.accel import moe
from cozy_runtime.author import (
    Config,
    Context,
    InvalidRequest,
    Loader,
    Model,
    ModelAssets,
    Seed,
    Serving,
    Telemetry,
    UnsupportedInput,
    uses_components,
)
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._model import WARNINGS
from cozy_runtime.author._tool import spoken
from cozy_runtime.fields import MENTION, Advanced, FileBound, Prompt, Video
from cozy_runtime.fields import Image as Picture
from cozy_runtime.media import DecodedVideoFrame, ImageFile, VideoFile

if TYPE_CHECKING:
    from transformers import BatchFeature, PreTrainedModel, ProcessorMixin

    from cozy_runtime.accel.kernels import Plan

COMPONENT = "model"
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
#: Where a thought ends: `ask` returns what follows it.
THOUGHT_END = "</think>"
#: A saved tokenizer's bookkeeping, not settings its constructor takes.
TOKENIZER_METADATA = (
    "added_tokens_decoder",
    "tokenizer_file",
    "processor_class",
    "chat_template",
    "name_or_path",
    "backend",
    "is_local",
    "local_files_only",
)

Images = Annotated[
    list[ImageFile],
    FileBound(max_decoded_bytes=256 << 20),
    msgspec.Meta(max_length=16),
]
Videos = Annotated[
    list[VideoFile],
    FileBound(max_decoded_bytes=16 << 30),
    msgspec.Meta(max_length=4),
]
Text = Annotated[str, msgspec.Meta(max_length=65536)]


class AskPrompt(msgspec.Struct):
    text: Annotated[str, msgspec.Meta(min_length=1, max_length=65536)]
    images: Images = []
    videos: Videos = []


class AskRequest(msgspec.Struct):
    #: One question, shown `images` then `videos`; or `prompts`, several decoded together.
    prompt: Text = ""
    images: Images = []
    videos: Videos = []
    prompts: Annotated[list[AskPrompt], msgspec.Meta(max_length=16)] = []
    system: Text = ""
    #: Think before answering (the chat template's switch); only the answer after the thought is
    #: returned. A thought `max_tokens` cuts off is returned as it stands, with a warning.
    thinking: bool = False
    max_tokens: Annotated[int, msgspec.Meta(ge=1, le=16384)] = 1024
    #: 0 answers greedily; above it, each token is drawn from the `top_p` nucleus at this
    #: temperature, from `seed`: the same request and seed give the same answer.
    temperature: Annotated[float, msgspec.Meta(ge=0, le=2)] = 0.0
    top_p: Annotated[float, msgspec.Meta(gt=0, le=1)] = 1.0
    seed: Annotated[int, msgspec.Meta(ge=0, le=(1 << 53) - 1)] = 0
    #: Subtracted from every token already generated (vLLM's).
    presence_penalty: Annotated[float, msgspec.Meta(ge=-2, le=2)] = 0.0
    #: An answer ends before the first of these.
    stop: Annotated[
        list[Annotated[str, msgspec.Meta(min_length=1, max_length=256)]], msgspec.Meta(max_length=8)
    ] = []
    #: Video reading, as Qwen-VL's: about `fps` frames a second, at least `min_frames` and at most
    #: `max_frames`; null takes the checkpoint's video processor, else (2, 4, 768).
    fps: Annotated[float, msgspec.Meta(gt=0, le=30)] | None = None
    min_frames: Annotated[int, msgspec.Meta(ge=2, le=768)] | None = None
    max_frames: Annotated[int, msgspec.Meta(ge=2, le=768)] | None = None
    #: Frames longer than this on their long edge are downscaled first (Lanczos); null: as decoded.
    max_edge: Annotated[int, msgspec.Meta(ge=32, le=4096)] | None = None
    #: Each video is watched up to this second; null: to its end.
    video_end: Annotated[float, msgspec.Meta(gt=0)] | None = None


class AskResult(msgspec.Struct):
    #: One answer per prompt, in order.
    texts: list[str]


class TextBox(msgspec.Struct, kw_only=True):
    """A box of words a model writes: one answer to its instructions (`write`)."""

    #: What to write, and the pictures and videos it mentions, each shown where first mentioned.
    instructions: Prompt[Picture | Video] = Prompt()
    #: Think before answering (`VisionLM._chat`); only the answer is the box's (`VisionLM.ask`).
    thinking: Annotated[bool, Advanced()] = False
    #: The system turn (`VisionLM._chat`).
    system: Annotated[Text, Advanced()] = ""
    #: `lm.generate`'s `max_new_tokens`.
    max_tokens: Annotated[int, msgspec.Meta(ge=1, le=16384), Advanced()] = 1024
    #: `lm.generate`'s sampling. The defaults are Qwen's for instruct VL tasks (its model cards);
    #: 0 answers greedily.
    temperature: Annotated[float, msgspec.Meta(ge=0, le=2), Advanced()] = 0.7
    top_p: Annotated[float, msgspec.Meta(gt=0, le=1), Advanced()] = 0.8
    presence_penalty: Annotated[float, msgspec.Meta(ge=-2, le=2), Advanced()] = 1.5
    #: `lm.generate`'s draw; unset, a new one each render.
    seed: Annotated[Seed | None, Advanced()] = None


class TextResult(msgspec.Struct):
    text: str


@dataclass(frozen=True)
class Watched:
    """A video as the model watches it: its frames and the second each shows."""

    frames: list[Image]
    seconds: list[float]


@dataclass(frozen=True)
class Question:
    """One single-turn question: its text after its images and videos, or `parts`, its text,
    pictures and videos in the order the model reads them."""

    text: str = ""
    images: list[Image] = field(default_factory=list)
    videos: list[Watched] = field(default_factory=list)
    parts: Sequence[str | Image | Watched] = ()

    def read(self) -> Sequence[str | Image | Watched]:
        """What the model reads, in order."""
        return self.parts or [*self.images, *self.videos, self.text]


def ask(ctx: Context, payload: AskRequest, model: VisionLM, tel: Telemetry) -> AskResult:
    """Answer each prompt about its text, images and videos, all prompts in one pass."""
    ctx.raise_if_cancelled()
    if payload.prompts and (payload.prompt or payload.images or payload.videos):
        tel.log("prompts wins: prompt, images and videos beside it are ignored", level="warning")
    prompts = payload.prompts or [AskPrompt(payload.prompt, payload.images, payload.videos)]
    if not prompts[0].text:
        raise InvalidRequest("prompt is empty", code="ask_prompts")
    return AskResult(
        _answers([[*p.images, *p.videos, p.text] for p in prompts], payload, model, tel)
    )


def write(ctx: Context, payload: TextBox, model: VisionLM, tel: Telemetry) -> TextResult:
    """The box's words: its instructions answered as one question, each picture or video it
    mentions shown right after its first mention (Qwen-VL's chat template takes them anywhere)."""
    ctx.raise_if_cancelled()
    if not payload.instructions.text.strip():
        raise InvalidRequest("instructions are empty", code="ask_prompts")
    asked = AskRequest(
        system=payload.system,
        thinking=payload.thinking,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        presence_penalty=payload.presence_penalty,
        seed=secrets.randbits(53) if payload.seed is None else payload.seed,
    )
    (text,) = _answers([_parts(payload.instructions)], asked, model, tel)
    return TextResult(text)


def _parts(prompt: Prompt[Picture | Video]) -> list[str | ImageFile | VideoFile]:
    """The prompt's text, mentions as the model reads them (`spoken`), each reference's file right
    after its first mention."""
    parts: list[str | ImageFile | VideoFile] = []
    shown: set[str] = set()
    at = 0
    for found in MENTION.finditer(prompt.text):
        key = found[1]
        if key in shown or key not in prompt.references:
            continue
        shown.add(key)
        parts += [spoken(prompt.text[at : found.end()]), prompt.references[key].file]
        at = found.end()
    parts.append(spoken(prompt.text[at:]))
    return [part for part in parts if part != ""]


def _answers(
    questions: list[list[str | ImageFile | VideoFile]],
    payload: AskRequest,
    model: VisionLM,
    tel: Telemetry,
) -> list[str]:
    """Each question's pictures read and videos watched as `payload` says, then all answered in
    one pass."""
    # The checkpoint's video reading, else Qwen-VL's (2, 4, 768): Gemma 4's processor states none.
    video = model.processor.video_processor
    fps = payload.fps or float(getattr(video, "fps", None) or 2)
    lo, hi = (
        payload.min_frames or int(getattr(video, "min_frames", None) or 4),
        payload.max_frames or int(getattr(video, "max_frames", None) or 768),
    )
    if lo > hi:
        tel.log(f"min_frames {lo} is above max_frames {hi}: swapped", level="warning")
        lo, hi = hi, lo

    def read(part: str | ImageFile | VideoFile) -> str | Image | Watched:
        if isinstance(part, VideoFile):
            return watch(part, fps, (lo, hi), payload.max_edge, payload.video_end)
        return part.read() if isinstance(part, ImageFile) else part

    with tel.stage("reading"):
        read_questions = [Question(parts=[read(part) for part in parts]) for parts in questions]
    with tel.stage("answering"):
        return model.ask(
            read_questions,
            system=payload.system,
            thinking=payload.thinking,
            stop=list(payload.stop),
            max_new_tokens=payload.max_tokens,
            presence_penalty=payload.presence_penalty,
            temperature=payload.temperature,
            top_p=payload.top_p,
            seed=payload.seed,
            progress=lambda tokens: tel._tokens(tokens, payload.max_tokens, stage="answering"),
        )


def watch(
    video: VideoFile, fps: float, frames: tuple[int, int], max_edge: int | None, end: float | None
) -> Watched:
    """About `fps` frames a second, `frames` at least and at most, an even count (Qwen-VL reads
    frame pairs), each the frame showing at the middle of its slot, downscaled to `max_edge`."""
    duration = float(video.probe().duration)
    if end is not None:
        duration = min(duration, end)
    count = min(max(round(duration * fps), frames[0]), frames[1])
    count += count % 2
    seconds = [(index + 0.5) * duration / count for index in range(count)]
    shown: list[Image] = []
    with video.stream(end=duration, audio=False) as events:
        for event in events:
            if not isinstance(event, DecodedVideoFrame):
                continue
            until = float(event.start_time + event.duration * event.time_base)
            picture = None
            while len(shown) < count and seconds[len(shown)] < until:
                if picture is None:
                    picture = PILImage.frombuffer(
                        "RGB", (event.width, event.height), event.rgb, "raw", "RGB", 0, 1
                    )
                    scale = max_edge / max(picture.size) if max_edge else 1
                    if scale < 1:
                        picture = picture.resize(
                            (round(picture.width * scale), round(picture.height * scale)),
                            PILImage.Resampling.LANCZOS,
                        )
                shown.append(picture)
    if not shown:
        raise UnsupportedInput("the video has no frames", code="media_decode")
    shown += [shown[-1]] * (count - len(shown))
    return Watched(shown, seconds)


def construct(model_config: Mapping[str, object], config: Config) -> PreTrainedModel:
    """The model, experts split per expert; every parameter in its stored dtype (Qwen3.5 keeps
    `A_log` and its gated norms in f32, and Runtime fills stored bytes as they are). Where the
    checkpoint keeps its multi-token-prediction head (`mtp.*`), that too, as the model's `mtp`:
    its decode drafts with the `lm.Draft` around it (`lm.HEADS`)."""
    with no_init_weights():
        model: PreTrainedModel = AutoModelForImageTextToText.from_config(  # type: ignore[no-untyped-call]  # unannotated in Transformers
            AutoConfig.for_model(**model_config),  # type: ignore[arg-type]  # config.json's keys
            dtype=torch.bfloat16,
        )
        model.tie_weights()
        text = model.config.get_text_config(decoder=True)
        head = None
        drafts = getattr(text, "mtp_num_hidden_layers", 0)
        if drafts and _held(config, f"{COMPONENT}.mtp.fc.weight"):
            head = lm.Draft(model.config, sys.modules[type(model).__module__])
            model.mtp = head.mtp.to(torch.bfloat16)
        moe.split(model)
    for name, parameter in model.named_parameters():
        stored = DTYPES.get(config.tensor_dtype(f"{COMPONENT}.{name}", default="bf16"))
        if stored is not None and parameter.dtype != stored:
            parameter.data = parameter.data.to(stored)
    if head is not None:
        lm.HEADS[model] = head.eval()
    evaluated: PreTrainedModel = model.eval()  # type: ignore[no-untyped-call]  # unannotated in Transformers
    return evaluated


def _held(config: Config, key: str) -> bool:
    """Whether the checkpoint stores `key`: a fine-tune may leave out the multi-token-prediction
    head its config still declares."""
    try:
        return bool(config.tensor_dtype(key, default=""))
    except ConformanceError:
        return False


def processor(
    assets: ModelAssets | Path,
    folder: str = "",
    config: Mapping[str, object] | None = None,
    image: Mapping[str, object] | None = None,
) -> ProcessorMixin:
    """The upstream processor from the checkpoint's files under `folder` (`processor/`), parsed in
    memory. The image and video processors' settings are their own files, one
    `processor_config.json`, or `config`: TensorFS keeps a component folder's config file as the
    component's config, not among its files. `image` replaces image settings (None drops one).
    The tokenizer is `tokenizer.json`, or a slow one's `vocab.json` and `merges.txt`; the chat
    template is `chat_template.jinja`, `chat_template.json` or the tokenizer config's. `assets` may
    be a directory a package bundles."""
    if isinstance(assets, Path):
        root = assets
        names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}

        def read(name: str) -> bytes:
            return (root / name).read_bytes()
    else:
        files = assets
        names = {name.removeprefix(folder) for name in files.names() if name.startswith(folder)}

        def read(name: str) -> bytes:
            return files.read(folder + name)

    def held(name: str) -> dict[str, Any]:  # type: ignore[explicit-any]  # JSON settings, Transformers' keywords
        return json.loads(read(name)) if name in names else {}

    given: dict[str, Any] = dict(config or {})  # type: ignore[explicit-any]  # as `held`'s
    extra = held("processor_config.json") or (given if "image_processor" in given else {})
    images = dict(extra.pop("image_processor", None) or held("preprocessor_config.json") or given)
    video = dict(extra.pop("video_processor", None) or held("video_preprocessor_config.json"))
    audio = dict(extra.pop("feature_extractor", None) or {})
    named = extra.pop("processor_class", None) or images.get("processor_class", "Qwen3VLProcessor")
    for settings in (images, video, audio):
        settings.pop("processor_class", None)
    for key, value in (image or {}).items():
        if value is None:
            images.pop(key, None)
        else:
            images[key] = value
    if video:
        kind = video.pop("video_processor_type", "Qwen3VLVideoProcessor")
        extra["video_processor"] = getattr(transformers, kind)(**video)
    if audio:  # Gemma 4 hears too: its feature extractor, as `processor_config.json` names it
        kind = audio.pop("feature_extractor_type")
        extra["feature_extractor"] = getattr(transformers, kind)(**audio)
    kind = images.pop("image_processor_type", "Qwen2VLImageProcessor")
    image_processor = getattr(transformers, kind)(**images)
    tokenizer_config: dict[str, Any] = json.loads(read("tokenizer_config.json"))  # type: ignore[explicit-any]  # as `held`'s
    added = tokenizer_config.get("added_tokens_decoder", {})
    chat = tokenizer_config.get("chat_template")
    for key in TOKENIZER_METADATA:
        tokenizer_config.pop(key, None)
    named_tokenizer = tokenizer_config.pop("tokenizer_class", "")
    tokenizer_class = getattr(transformers, named_tokenizer, None)
    if tokenizer_class is None or named_tokenizer == "PreTrainedTokenizer":
        tokenizer_class = PreTrainedTokenizerFast
    if "tokenizer.json" in names:
        tokenizer = tokenizer_class(
            tokenizer_object=Tokenizer.from_str(read("tokenizer.json").decode()), **tokenizer_config
        )
    else:  # a slow tokenizer: its vocabulary, merges and added tokens
        merges = [tuple(line.split(" ")) for line in read("merges.txt").decode().splitlines()]
        tokenizer = tokenizer_class(
            vocab=json.loads(read("vocab.json")),
            merges=[pair for pair in merges if pair[0] and not pair[0].startswith("#")],
            added_tokens_decoder={int(i): AddedToken(**token) for i, token in added.items()},
            **tokenizer_config,
        )
    if "chat_template.jinja" in names:
        chat = read("chat_template.jinja").decode()
    elif "chat_template.json" in names:
        chat = json.loads(read("chat_template.json"))["chat_template"]
    tokenizer.chat_template = chat
    made: ProcessorMixin = getattr(transformers, named)(
        image_processor=image_processor, tokenizer=tokenizer, chat_template=chat, **extra
    )
    return made


class Pipeline:
    def __init__(self, config: Config) -> None:
        model = construct(config.mapping(), config)
        self.components: dict[str, PreTrainedModel] = {COMPONENT: model}


class VisionLM(Model[Pipeline]):
    """A Qwen-VL model answering questions about text, images and videos, greedily or sampled
    from a seed. Runtime may serve encoded lanes' linears natively."""

    serving = Serving(min_bits=4, phase="decode")

    pipe: Pipeline
    processor: ProcessorMixin

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline)
        self.processor = processor(loader.assets)
        if "generation_config.json" in loader.assets.names():
            model = self.pipe.components[COMPONENT]
            model.generation_config = GenerationConfig(  # type: ignore[no-untyped-call]  # unannotated in Transformers
                **json.loads(loader.assets.read("generation_config.json"))
            )

    def execution_plan(self, plan: Plan) -> str:
        model = self.pipe.components[COMPONENT]
        return "+".join(filter(None, (lm.install(plan, model), moe.install(plan, model))))

    def unload(self, loader: Loader) -> None:
        del self.pipe

    @uses_components(COMPONENT)
    def ask(
        self,
        questions: list[Question],
        *,
        system: str,
        thinking: bool,
        stop: list[str],
        max_new_tokens: int,
        presence_penalty: float,
        temperature: float,
        top_p: float,
        seed: int,
        progress: Callable[[int], None] | None = None,
        no_repeat_ngram: int | None = None,
    ) -> list[str]:
        """One answer per question, all at once: one prefill over the rows, left-padded, and one
        decode of them together (the decoding arguments are `lm.generate`'s). Each video's
        blocks are stamped with their `<t seconds>` as the processor writes them. An answer ends
        before its first `stop`; with `thinking`, only what follows the thought is the answer,
        and a thought `max_new_tokens` cut short is returned as it stands, with a run warning."""
        texts = [self._chat(question, system, thinking) for question in questions]
        parts = [part for question in questions for part in question.read()]
        images = [part for part in parts if isinstance(part, Image)]
        videos = [part for part in parts if isinstance(part, Watched)]
        kwargs: dict[str, object] = {"images": images} if images else {}
        if videos:
            # The processor stamps a block with frame index / fps: millisecond indices keep seconds.
            kwargs |= {
                "videos": [video.frames for video in videos],
                "video_metadata": [
                    VideoMetadata(
                        total_num_frames=len(video.frames),
                        fps=1000.0,
                        frames_indices=[round(t * 1000) for t in video.seconds],
                    )
                    for video in videos
                ],
                "do_sample_frames": False,
            }
        decoding: _Decoding = {
            "max_new_tokens": max_new_tokens,
            "presence_penalty": presence_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "progress": progress,
            "no_repeat_ngram": no_repeat_ngram,
        }
        # A thought may say a stop string: then stops cut only the answer, after decoding.
        replies = self._generate(
            self._inputs(text=texts, **kwargs), [] if thinking else stop, decoding
        )
        if not thinking:
            return [_cut(reply, stop).strip() for reply in replies]
        answers = []
        for row, reply in enumerate(replies):
            thought, ended, answer = reply.partition(THOUGHT_END)
            if ended:
                answers.append(_cut(answer, stop).strip())
                continue
            WARNINGS.append(
                (
                    "a thought was cut short at max_tokens, before its answer; it is returned",
                    {"row": row, "max_tokens": max_new_tokens},
                )
            )
            answers.append(thought.strip())
        return answers

    def _chat(self, question: Question, system: str, thinking: bool) -> str:
        content = [
            {"type": "text", "text": part}
            if isinstance(part, str)
            else {"type": "video" if isinstance(part, Watched) else "image"}
            for part in question.read()
        ]
        messages = [{"role": "system", "content": system}] if system else []
        text: str = self.processor.apply_chat_template(
            # Transformers types a turn's content as text; a multimodal turn's is a list
            [*messages, {"role": "user", "content": content}],  # type: ignore[arg-type]  # see above
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=thinking,
        )
        return text

    def _inputs(self, **kwargs: object) -> BatchFeature:
        model = self.pipe.components[COMPONENT]
        self.processor.tokenizer.padding_side = "left"
        inputs: BatchFeature = self.processor(padding=True, return_tensors="pt", **kwargs)
        inputs = inputs.to(model.device)
        for name in ("pixel_values", "pixel_values_videos"):
            if name in inputs:
                inputs[name] = inputs[name].to(model.dtype)
        return inputs

    def _generate(self, inputs: BatchFeature, stop: list[str], decoding: _Decoding) -> list[str]:
        """Decoded answers. A stop string that is one token also ends its row's decode."""
        model, tokenizer = self.pipe.components[COMPONENT], self.processor.tokenizer
        configured = model.generation_config.eos_token_id
        single = [
            ids[0]
            for ids in (tokenizer.encode(s, add_special_tokens=False) for s in stop)
            if len(ids) == 1
        ]
        stops = [
            *(configured if isinstance(configured, list) else [configured]),
            tokenizer.eos_token_id,
            *single,
        ]
        tokens = [token for token in dict.fromkeys(stops) if token is not None]
        out = lm.generate(
            model,
            inputs,
            stop=tokens,
            pad=tokenizer.pad_token_id or tokens[0],
            **decoding,
        )
        replies = self.processor.batch_decode(out, skip_special_tokens=True)  # type: ignore[no-untyped-call]  # unannotated in Transformers
        return [str(reply) for reply in replies]


class _Decoding(TypedDict):
    """`ask`'s decoding arguments, as `lm.generate` takes them."""

    max_new_tokens: int
    presence_penalty: float
    temperature: float
    top_p: float
    seed: int
    progress: Callable[[int], None] | None
    no_repeat_ngram: int | None


def _cut(text: str, stop: list[str]) -> str:
    """`text` before the first of the `stop` strings in it."""
    return text[: min((at for s in stop if (at := text.find(s)) >= 0), default=len(text))]
