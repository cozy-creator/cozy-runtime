"""The author-facing surface — the ONLY cozy-runtime package endpoint/job code imports.

Structural fence (cozy-runtime.md §7, boundaries.md): this package may import schema and
typing libraries only. It cannot import cozy_runtime.internal, cozy_runtime.cli, the
worker protocol, TensorFS, Varena or any coordinator; it cannot read environment or config;
it names no device, allocator, compile or ingest surface. `scripts/fence.py` proves it.

A handler's SIGNATURE is its capability declaration: `Context` says where it runs, and
everything it DOES rides a typed service parameter it named. The surface below is the
complete set of things an endpoint author touches, minus the codec plane (cr-017). Per-service
fakes live in `cozy_runtime.author.fakes`.
"""

from cozy_runtime.author._app import App, Registration, RegistrationError
from cozy_runtime.author._assets import (
    Asset,
    AudioAsset,
    FileAsset,
    ImageAsset,
    VideoAsset,
)
from cozy_runtime.author._context import (
    AdapterRef,
    Context,
    Device,
    RequestView,
)
from cozy_runtime.author._defaults import Clamp, Overlay, Recipe
from cozy_runtime.author._demand import RequestFeatures
from cozy_runtime.author._describe import DELETED_SURFACES, assert_declaration_shape, describe
from cozy_runtime.author._errors import (
    AuthorError,
    Cancelled,
    CapabilityError,
    ConformanceError,
    DeadlineExceeded,
    DefaultResolutionError,
    InvalidRequest,
    OutputError,
    PreflightViolation,
    UnsupportedInput,
)
from cozy_runtime.author._invoke import Invocation, ResultEnvelope, invoke
from cozy_runtime.author._loader import (
    Artifact,
    Backend,
    Config,
    ConstructionRecord,
    DerivedCache,
    Loader,
    ModuleLike,
    Scheduler,
    TensorSpec,
    register_adapter,
    registered_adapters,
)
from cozy_runtime.author._markers import Bound, ModelDefault, Preflight, Resources, Shape
from cozy_runtime.author._model import (
    ConstructionIdentity,
    Model,
    ModelRegistry,
    ModelTestHarness,
    ScopeCall,
    uses_components,
)
from cozy_runtime.author._services import (
    AdjustmentRow,
    Adjustments,
    Budget,
    Checkpoints,
    Egress,
    EgressResponse,
    ImageFrame,
    Outputs,
    Scratch,
    Secrets,
    Settings,
    Telemetry,
    TelemetryEvent,
)
from cozy_runtime.author._signature import ModelBinding, Surface

__all__ = [
    "DELETED_SURFACES",
    "AdapterRef",
    "AdjustmentRow",
    "Adjustments",
    "App",
    "Artifact",
    "Asset",
    "AudioAsset",
    "AuthorError",
    "Backend",
    "Bound",
    "Budget",
    "Cancelled",
    "CapabilityError",
    "Checkpoints",
    "Clamp",
    "Config",
    "ConformanceError",
    "ConstructionIdentity",
    "ConstructionRecord",
    "Context",
    "DeadlineExceeded",
    "DefaultResolutionError",
    "DerivedCache",
    "Device",
    "Egress",
    "EgressResponse",
    "FileAsset",
    "ImageAsset",
    "ImageFrame",
    "InvalidRequest",
    "Invocation",
    "Loader",
    "Model",
    "ModelBinding",
    "ModelDefault",
    "ModelRegistry",
    "ModelTestHarness",
    "ModuleLike",
    "OutputError",
    "Outputs",
    "Overlay",
    "Preflight",
    "PreflightViolation",
    "Recipe",
    "Registration",
    "RegistrationError",
    "RequestFeatures",
    "RequestView",
    "Resources",
    "ResultEnvelope",
    "Scheduler",
    "ScopeCall",
    "Scratch",
    "Secrets",
    "Settings",
    "Shape",
    "Surface",
    "Telemetry",
    "TelemetryEvent",
    "TensorSpec",
    "UnsupportedInput",
    "VideoAsset",
    "assert_declaration_shape",
    "describe",
    "invoke",
    "register_adapter",
    "registered_adapters",
    "uses_components",
]
