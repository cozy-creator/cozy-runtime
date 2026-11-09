"""The author-facing declaration and service surface.

Structural fence (cozy-runtime.md §7, boundaries.md): this package may import schema and
typing libraries and the pure canonical-JSON facade only. It cannot import
cozy_runtime.internal, cozy_runtime.cli, the
worker protocol, TensorFS, Varena or any RecordOwner; it cannot read environment or config;
it names no allocator or ingest surface, and a package's own kernels are declarations the
runtime compiles and hands back (`author.kernels`).

A handler's SIGNATURE is its capability declaration: `Context` says where it runs, and
everything it DOES rides a typed service parameter it named. A `Model` has two lifecycle
methods — `load(loader)` and `unload(loader)` — and the runtime owns when each runs
(model-lifecycle.md). A package writes its own loops in plain PyTorch (README.md). The codec
plane is separate (cr-017). Per-service fakes live in `cozy_runtime.author.fakes`.
"""

from cozy_runtime import __version__, canonical_json
from cozy_runtime.author._activity import ThreadPoolExecutor
from cozy_runtime.author._app import App, Registration, RegistrationError
from cozy_runtime.author._artifacts import Lora, ModelArtifact, ModelChoice, ObjectRef
from cozy_runtime.author._attention import (
    AttentionProducer,
    ProducerCall,
    ProducerSite,
    producer_sites,
    register_attention,
)
from cozy_runtime.author._attention_scope import (
    AttentionLayout,
    active_attention_layout,
    attention_scope,
    expect_attention,
)
from cozy_runtime.author._calls import ChildCallError, PendingCall, invocable, prefetch
from cozy_runtime.author._concurrent import concurrently
from cozy_runtime.author._context import (
    AdapterRef,
    Context,
    Device,
    RequestView,
    device,
)
from cozy_runtime.author._data import data_values
from cozy_runtime.author._decode import (
    DEFAULT_DECODE_LIMITS,
    AudioProbe,
    DecodedAudio,
    DecodedAudioChunk,
    DecodedAudioFormat,
    DecodedMediaEvent,
    DecodedMediaHeader,
    DecodedMediaStream,
    DecodedVideo,
    DecodedVideoFormat,
    DecodedVideoFrame,
    DecodeLimits,
    DecoderUnavailable,
    ImageProbe,
    MediaDecodeError,
    VideoProbe,
)
from cozy_runtime.author._defaults import Overlay, decode_request
from cozy_runtime.author._demand import RequestFeatures
from cozy_runtime.author._describe import assert_declaration_shape, describe
from cozy_runtime.author._errors import (
    NEUTRAL_ERROR_SET,
    AuthorError,
    Cancelled,
    CapabilityError,
    ConformanceError,
    DeadlineExceeded,
    DefaultResolutionError,
    DoesNotFit,
    InvalidRequest,
    Outcome,
    OutputError,
    PreflightViolation,
    UnsupportedInput,
    classify,
)
from cozy_runtime.author._files import (
    AudioFile,
    File,
    FileInfo,
    Files,
    ImageFile,
    Latents,
    References,
    Tree,
    VideoFile,
)
from cozy_runtime.author._image import (
    MEGAPIXEL,
    Box,
    Guidance,
    ImageRequest,
    ImageResult,
    NegativePrompt,
    Region,
    Sizes,
    Steps,
    Target,
    corners,
    held,
    ratio_of,
)
from cozy_runtime.author._invoke import (
    Invocation,
    InvocationResult,
    PreparedRequest,
    prepare,
)
from cozy_runtime.author._loader import (
    Artifact,
    Backend,
    CheckpointUnnormalized,
    Config,
    ConstructionRecord,
    DerivedCache,
    Fit,
    Loader,
    ModelAssets,
    ModelFitRefused,
    ModuleLike,
    Scheduler,
    TensorSpec,
    register_adapter,
    registered_adapters,
)
from cozy_runtime.author._markers import (
    Advanced,
    AudioPreparation,
    Bound,
    Count,
    FileBound,
    FileLimits,
    ImagePreparation,
    Label,
    ModelDefault,
    Placeholder,
    Preflight,
    Range,
    Ratios,
    Shape,
    Snap,
    Spans,
    Total,
    VideoPreparation,
)
from cozy_runtime.author._memo import MemoDistribution, Memoize, MemoResource
from cozy_runtime.author._model import (
    AdapterCompatibility,
    ConstructionIdentity,
    Model,
    ModelRegistry,
    ModelTestHarness,
    ScopeCall,
    graph_keys,
    hold_dtype,
    placeable,
    plain_forward,
    sequence_parallel,
    uncaptured,
    uses_components,
)
from cozy_runtime.author._recipes import (
    DerivationHook,
    EncodingRule,
    QuantizationPlan,
    Recipe,
    SelectionPlan,
)
from cozy_runtime.author._retained import retained
from cozy_runtime.author._services import (
    AdjustmentRow,
    AdjustmentRows,
    Adjustments,
    Egress,
    EgressResponse,
    ImageFrame,
    Outputs,
    SavedAudioFacts,
    SavedVideo,
    Secrets,
    Settings,
    Telemetry,
    VideoJoin,
)
from cozy_runtime.author._signature import ModelBinding, Surface
from cozy_runtime.author._split import Cut, Windows, split, windows
from cozy_runtime.author._spread import spread
from cozy_runtime.author._tool import (
    MENTION,
    AspectRatio,
    Audio,
    Image,
    Pickup,
    Prompt,
    Seed,
    Tool,
    Video,
    name_key,
    spoken,
)
from cozy_runtime.author._weights import (
    MAX_WEIGHTS_NEW_BYTES,
    Weights,
    WeightsOutput,
)
from cozy_runtime.author.publication import (
    AssessmentRef,
    CheckpointRef,
    ReleaseReceipt,
    attach_assessment,
    publish_release,
    upload_checkpoint,
)

__all__ = [
    "DEFAULT_DECODE_LIMITS",
    "MAX_WEIGHTS_NEW_BYTES",
    "MEGAPIXEL",
    "MENTION",
    "NEUTRAL_ERROR_SET",
    "AdapterCompatibility",
    "AdapterRef",
    "AdjustmentRow",
    "AdjustmentRows",
    "Adjustments",
    "Advanced",
    "App",
    "Artifact",
    "AspectRatio",
    "AssessmentRef",
    "AttentionLayout",
    "AttentionProducer",
    "Audio",
    "AudioFile",
    "AudioPreparation",
    "AudioProbe",
    "AuthorError",
    "Backend",
    "Bound",
    "Box",
    "Cancelled",
    "CapabilityError",
    "CheckpointRef",
    "CheckpointUnnormalized",
    "ChildCallError",
    "Config",
    "ConformanceError",
    "ConstructionIdentity",
    "ConstructionRecord",
    "Context",
    "Count",
    "Cut",
    "DeadlineExceeded",
    "DecodeLimits",
    "DecodedAudio",
    "DecodedAudioChunk",
    "DecodedAudioFormat",
    "DecodedMediaEvent",
    "DecodedMediaHeader",
    "DecodedMediaStream",
    "DecodedVideo",
    "DecodedVideoFormat",
    "DecodedVideoFrame",
    "DecoderUnavailable",
    "DefaultResolutionError",
    "DerivationHook",
    "DerivedCache",
    "Device",
    "DoesNotFit",
    "Egress",
    "EgressResponse",
    "EncodingRule",
    "File",
    "FileBound",
    "FileInfo",
    "FileLimits",
    "Files",
    "Fit",
    "Guidance",
    "Image",
    "ImageFile",
    "ImageFrame",
    "ImagePreparation",
    "ImageProbe",
    "ImageRequest",
    "ImageResult",
    "InvalidRequest",
    "Invocation",
    "InvocationResult",
    "Label",
    "Latents",
    "Loader",
    "Lora",
    "MediaDecodeError",
    "MemoDistribution",
    "MemoResource",
    "Memoize",
    "Model",
    "ModelArtifact",
    "ModelAssets",
    "ModelBinding",
    "ModelChoice",
    "ModelDefault",
    "ModelFitRefused",
    "ModelRegistry",
    "ModelTestHarness",
    "ModuleLike",
    "NegativePrompt",
    "ObjectRef",
    "Outcome",
    "OutputError",
    "Outputs",
    "Overlay",
    "PendingCall",
    "Pickup",
    "Placeholder",
    "Preflight",
    "PreflightViolation",
    "PreparedRequest",
    "ProducerCall",
    "ProducerSite",
    "Prompt",
    "QuantizationPlan",
    "Range",
    "Ratios",
    "Recipe",
    "References",
    "Region",
    "Registration",
    "RegistrationError",
    "ReleaseReceipt",
    "RequestFeatures",
    "RequestView",
    "SavedAudioFacts",
    "SavedVideo",
    "Scheduler",
    "ScopeCall",
    "Secrets",
    "Seed",
    "SelectionPlan",
    "Settings",
    "Shape",
    "Sizes",
    "Snap",
    "Spans",
    "Steps",
    "Surface",
    "Target",
    "Telemetry",
    "TensorSpec",
    "ThreadPoolExecutor",
    "Tool",
    "Total",
    "Tree",
    "UnsupportedInput",
    "Video",
    "VideoFile",
    "VideoJoin",
    "VideoPreparation",
    "VideoProbe",
    "Weights",
    "WeightsOutput",
    "Windows",
    "active_attention_layout",
    "assert_declaration_shape",
    "attach_assessment",
    "attention_scope",
    "canonical_json",
    "classify",
    "concurrently",
    "corners",
    "data_values",
    "decode_request",
    "describe",
    "device",
    "expect_attention",
    "graph_keys",
    "held",
    "hold_dtype",
    "invocable",
    "name_key",
    "placeable",
    "plain_forward",
    "prefetch",
    "prepare",
    "producer_sites",
    "publish_release",
    "ratio_of",
    "register_adapter",
    "register_attention",
    "registered_adapters",
    "retained",
    "sequence_parallel",
    "split",
    "spoken",
    "spread",
    "uncaptured",
    "upload_checkpoint",
    "uses_components",
    "windows",
]

_REFS = "Prompt[Image]: each prompt's references, an Image(file, description) per mention"
_SAID = "Prompt.references, keyed as the text spells each mention"
#: Names a Runtime removed, the release that removed them ("": this one), and what replaces each.
#: A package built for an older Runtime imports one of them and fails here, never decoding a
#: request with a name that now means something else.
_REMOVED = {
    "ImageAsset": ("0.36.0", "ImageFile"),
    "VideoAsset": ("0.36.0", "VideoFile"),
    "AudioAsset": ("0.36.0", "AudioFile"),
    "FileAsset": ("0.36.0", "File"),
    "Assets": ("0.36.0", "Files"),
    "AssetBound": ("0.36.0", "FileBound"),
    "AssetLimits": ("0.36.0", "FileLimits"),
    "AssetInfo": ("0.36.0", "FileInfo"),
    "Node": ("", "app.tool(label=, named=, reserved=, ui=) and the box's field types"),
    "Reference": ("", _REFS),
    "Picture": ("", _REFS),
    "PlacedPicture": ("", _REFS),
    "PictureRequest": ("", "ImageRequest with prompt: Prompt[Image]"),
    "Mention": ("", _SAID),
    "mentions": ("", _SAID),
    "render": ("", _SAID),
    "name_label": ("", _SAID),
    "Plate": ("", "a VideoFile field marked Spans(...)"),
    "ContinueFrom": ("", "Pickup"),
    "ShotResult": ("", "the package's own result struct"),
    "UpscaleResult": ("", "the package's own result struct"),
    "Follow": ("", "the package's own Literal"),
    "denoise_loop": ("", "a plain `for` loop calling the model; `ctx.progress(i, n)`"),
    "Preset": ("", "the package's own Literal"),
}


def removed(name: str) -> str:
    """Why `cozy_runtime.author.<name>` is gone, and what to use."""
    release, instead = _REMOVED[name]
    if not release:
        return (
            f"cozy_runtime.author.{name} is gone in this Runtime ({__version__}): use {instead}; "
            "this package was built for an older Runtime, so use its newest release"
        )
    return (
        f"cozy_runtime.author.{name} is gone in Runtime {release} ({instead} replaces it): "
        f"this package was built for Runtime < {release.rsplit('.', 1)[0]}; use its newest release"
    )


def __getattr__(name: str) -> object:
    if name in _REMOVED:
        raise ImportError(removed(name), name=name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
