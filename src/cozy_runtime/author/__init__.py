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
    Tree,
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
    NEUTRAL_ERROR_SET,
    AuthorError,
    Cancelled,
    CapabilityError,
    ConformanceError,
    DeadlineExceeded,
    DefaultResolutionError,
    InvalidRequest,
    Outcome,
    OutputError,
    PreflightViolation,
    UnsupportedInput,
    classify,
)
from cozy_runtime.author._invoke import Invocation, ResultEnvelope, attempt, invoke
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
from cozy_runtime.author._markers import (
    AssetBound,
    Bound,
    ModelDefault,
    Preflight,
    Resources,
    Shape,
)
from cozy_runtime.author._model import (
    ConstructionIdentity,
    Model,
    ModelRegistry,
    ModelTestHarness,
    ScopeCall,
    uses_components,
)
from cozy_runtime.author._services import (
    JOB_ONLY,
    SERVING_ONLY,
    AdjustmentRow,
    Adjustments,
    Budget,
    BudgetExhausted,
    BudgetFacts,
    CheckpointConflict,
    CheckpointReceipt,
    Checkpoints,
    CheckpointSave,
    Egress,
    EgressResponse,
    ImageFrame,
    Outputs,
    Scratch,
    Secrets,
    Settings,
    Telemetry,
)
from cozy_runtime.author._signature import ModelBinding, Surface

__all__ = [
    "DELETED_SURFACES",
    "JOB_ONLY",
    "NEUTRAL_ERROR_SET",
    "SERVING_ONLY",
    "AdapterRef",
    "AdjustmentRow",
    "Adjustments",
    "App",
    "Artifact",
    "Asset",
    "AssetBound",
    "AudioAsset",
    "AuthorError",
    "Backend",
    "Bound",
    "Budget",
    "BudgetExhausted",
    "BudgetFacts",
    "Cancelled",
    "CapabilityError",
    "CheckpointConflict",
    "CheckpointReceipt",
    "CheckpointSave",
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
    "Outcome",
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
    "TensorSpec",
    "Tree",
    "UnsupportedInput",
    "VideoAsset",
    "assert_declaration_shape",
    "attempt",
    "classify",
    "describe",
    "invoke",
    "register_adapter",
    "registered_adapters",
    "uses_components",
]
