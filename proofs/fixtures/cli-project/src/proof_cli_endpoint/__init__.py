from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Literal

import msgspec

from cozy_runtime.author import (
    App,
    Artifact,
    AudioAsset,
    Config,
    ConformanceError,
    ImageAsset,
    Loader,
    Model,
    ModelRegistry,
    TensorSpec,
    VideoAsset,
    uses_components,
)

app = App()


class EchoRequest(msgspec.Struct, forbid_unknown_fields=True):
    text: str
    count: int = 1


class EchoResult(msgspec.Struct):
    text: str


@app.entrypoint
def echo(payload: EchoRequest) -> EchoResult:
    return EchoResult(payload.text * payload.count)


class ImageReference(msgspec.Struct, tag="image", tag_field="type"):
    image: ImageAsset


class VideoReference(msgspec.Struct, tag="video", tag_field="type"):
    video: VideoAsset


class AudioReference(msgspec.Struct, tag="audio", tag_field="type"):
    audio: AudioAsset


Reference = ImageReference | VideoReference | AudioReference


class InspectRequest(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    cover: ImageAsset = msgspec.field(name="hero")
    references: list[Reference]


class InspectResult(msgspec.Struct):
    prompt: str
    digests: list[str]


@app.job
def inspect(payload: InspectRequest) -> InspectResult:
    assets: list[ImageAsset | VideoAsset | AudioAsset] = [payload.cover]
    assets.extend(
        reference.image
        if isinstance(reference, ImageReference)
        else reference.video
        if isinstance(reference, VideoReference)
        else reference.audio
        for reference in payload.references
    )
    return InspectResult(
        prompt=payload.prompt,
        digests=[hashlib.sha256(asset.read_bytes()).hexdigest() for asset in assets],
    )


class _LogicalTensor:
    shape = (2, 2)
    dtype = "fp16"
    nbytes = 8


class _Layer:
    def state_dict(self) -> Mapping[str, _LogicalTensor]:
        return {"weight": _LogicalTensor()}


class _Pipeline:
    def __init__(self) -> None:
        self.components = {"core": _Layer()}


def _build(config: Config) -> _Pipeline:
    if config.keys():
        raise ValueError("the key-classification fixture takes an empty config")
    return _Pipeline()


class _KeyModel(Model[_Pipeline]):
    pipe: _Pipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(_Pipeline, factory=_build)

    def unload(self, loader: Loader) -> None:
        del loader
        del self.pipe

    @uses_components("core")
    def touch(self) -> None:
        return None


class KeyFacts(msgspec.Struct):
    canonical_refusal: str
    ignored_extras: list[str]
    strict_refusal: str
    transcribed_markers: list[str]
    marker_conflict: str
    clean: bool


def _artifact(
    *,
    extra: str = "",
    custody: Literal["canonical", "local"] = "canonical",
    strict: bool = False,
    stamps: Mapping[str, str] | None = None,
) -> Artifact:
    topology = {"core.weight": TensorSpec(shape=(2, 2), dtype="fp16", nbytes=8)}
    if extra:
        topology[f"core.{extra}"] = TensorSpec(shape=(1,), dtype="fp16", nbytes=2)
    return Artifact(
        snapshot="proof/key-contract@v1",
        topology=topology,
        config=Config({}),
        stamps=stamps or {},
        custody=custody,
        strict_keys=strict,
    )


def _refusal(artifact: Artifact) -> str:
    try:
        ModelRegistry(release="proof-cli").acquire(
            "key_contract.models.model", _KeyModel, artifact, mode="derive"
        )
    except ConformanceError as exc:
        return exc.code
    return ""


class KeyRequest(msgspec.Struct):
    run: bool = True


@app.job
def key_contract(payload: KeyRequest) -> KeyFacts:
    if not payload.run:
        raise ValueError("the proof arm must run")

    local = ModelRegistry(release="proof-cli")
    local.acquire(
        "key_contract.models.model",
        _KeyModel,
        _artifact(extra="left_over_weight", custody="local"),
        mode="derive",
    )
    local_record = local.generations()[0].record

    marked = ModelRegistry(release="proof-cli")
    marked.acquire(
        "key_contract.models.model",
        _KeyModel,
        _artifact(extra="v_pred", custody="local"),
        mode="derive",
    )
    marked_record = marked.generations()[0].record

    clean = ModelRegistry(release="proof-cli")
    clean.acquire("key_contract.models.model", _KeyModel, _artifact(custody="local"), mode="derive")
    clean_record = clean.generations()[0].record

    return KeyFacts(
        canonical_refusal=_refusal(_artifact(extra="left_over_weight")),
        ignored_extras=list(local_record.ignored_extras),
        strict_refusal=_refusal(_artifact(extra="left_over_weight", custody="local", strict=True)),
        transcribed_markers=list(marked_record.transcribed_markers),
        marker_conflict=_refusal(
            _artifact(extra="v_pred", custody="local", stamps={"objective": "epsilon"})
        ),
        clean=clean_record.ignored_extras == () and clean_record.transcribed_markers == (),
    )


@app.entrypoint(hidden=True)
def modeled(payload: EchoRequest, model: _KeyModel) -> EchoResult:
    model.touch()
    return EchoResult(payload.text)
