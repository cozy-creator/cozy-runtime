"""What a package needs to ship: an `App`, handlers whose parameters the Runtime injects by
type, and a `Model` whose `load(loader)` builds its components (README.md).

The core is `App`, `Context`, `Model`, `Loader`, `Outputs`, `Telemetry` and `Seed`. The rest is
optional and lives beside it: `cozy_runtime.image`, `.fields`, `.media`, `.accel`, `.rng`,
`.lm` and `.vlm`.

Structural fence: this package imports schema and typing libraries and the canonical-JSON
facade only. It never imports `cozy_runtime.internal`, the CLI, TensorFS or varena, and reads
no environment or config.
"""

from cozy_runtime import canonical_json
from cozy_runtime.author._activity import ThreadPoolExecutor
from cozy_runtime.author._app import App
from cozy_runtime.author._artifacts import Lora, ModelArtifact, ModelChoice, ObjectRef
from cozy_runtime.author._calls import ChildCallError, PendingCall, invocable, prefetch
from cozy_runtime.author._context import Context, Device, device
from cozy_runtime.author._errors import Cancelled, InvalidRequest, OutputError, UnsupportedInput
from cozy_runtime.author._loader import Config, DerivedCache, Loader, ModelAssets
from cozy_runtime.author._memo import Memoize
from cozy_runtime.author._model import Model, Serving, memoize, uses_components
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
    ImageFrame,
    Outputs,
    SavedAudioFacts,
    SavedVideo,
    Telemetry,
    VideoJoin,
)
from cozy_runtime.author._split import Cut, Windows, split, windows
from cozy_runtime.author._tool import Seed
from cozy_runtime.author._weights import MAX_WEIGHTS_NEW_BYTES, Weights, WeightsOutput

__all__ = [
    "MAX_WEIGHTS_NEW_BYTES",
    "AdjustmentRow",
    "AdjustmentRows",
    "Adjustments",
    "App",
    "Cancelled",
    "ChildCallError",
    "Config",
    "Context",
    "Cut",
    "DerivationHook",
    "DerivedCache",
    "Device",
    "EncodingRule",
    "ImageFrame",
    "InvalidRequest",
    "Loader",
    "Lora",
    "Memoize",
    "Model",
    "ModelArtifact",
    "ModelAssets",
    "ModelChoice",
    "ObjectRef",
    "OutputError",
    "Outputs",
    "PendingCall",
    "QuantizationPlan",
    "Recipe",
    "SavedAudioFacts",
    "SavedVideo",
    "Seed",
    "SelectionPlan",
    "Serving",
    "Telemetry",
    "ThreadPoolExecutor",
    "UnsupportedInput",
    "VideoJoin",
    "Weights",
    "WeightsOutput",
    "Windows",
    "canonical_json",
    "device",
    "invocable",
    "memoize",
    "prefetch",
    "retained",
    "split",
    "uses_components",
    "windows",
]
