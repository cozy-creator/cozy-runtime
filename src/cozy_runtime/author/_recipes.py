"""Quantize recipes: data a package declares at module scope, which the Runtime's `quantize` runs.

```python
INT8 = Recipe(lane="int8", model=Klein, plan=QuantizationPlan(encoding="int8.convrot", ...))
```

The static describe finds every `Recipe` bound at module scope of a module the App reaches and
records it in the package interface (`recipes`), named by its binding in lower case. A plan is
pure data: its canonical form (`canonical`) and that form's digest (`fingerprint`) identify it.
Work a plan cannot say is a `DerivationHook`: a plain module-level function, identified by its
module and name and fingerprinted with the package release that ships it. A plan fitted to the
model's own activations (`svdq.nvfp4`) names its `calibrate` function the same way: it runs real
inputs through the model, constructed at the source's 16 bits, while their recorder is open.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._model import Model
from cozy_runtime.author._weights import _OUTPUT_NAME, MAX_WEIGHTS_NEW_BYTES, Weights, WeightsOutput

if TYPE_CHECKING:
    from cozy_runtime.author._artifacts import ModelArtifact
    from cozy_runtime.author._context import Context
    from cozy_runtime.author._services import Telemetry


#: The entrypoint that runs a package's recipes (`cozy_runtime.derive.recipes`).
ENTRYPOINT = "quantize"
#: The Runtime function a package with a calibrated recipe registers as its `quantize`.
CALIBRATED_ENTRYPOINT = "quantize_calibrated"


class EncodingRule(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Selected weights whose `component.key` matches `pattern` take `encoding`."""

    pattern: str
    encoding: str


class QuantizationPlan(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    omit_defaults=True,
    tag_field="kind",
    tag="quantization",
):
    """A lane's weights and encodings. `encoding` is the lane's own; none writes the plain
    lane (every float normalized to bf16, nothing encoded).

    `keys` names exact weights in every component; `include` instead selects the weights
    whose `component.key` matches one of its fnmatch patterns, each of which must match one;
    neither selects every block-aligned rank-2 weight of `components`. Every other tensor
    keeps its source bytes. A selected weight matching an `encodings` rule (the first that
    does) takes that rule's encoding instead of the lane's, so one lane can carry, say, an
    nvfp4 DiT beside an int8 text encoder."""

    encoding: str | None = None
    components: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    encodings: tuple[EncodingRule, ...] = ()
    selected_schema_digest: str | None = None
    source_order_digests: tuple[str, ...] = ()
    output_precision: Literal["normalize-f32-to-bf16", "preserve"] = "normalize-f32-to-bf16"


class SelectionPlan(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    omit_defaults=True,
    tag_field="kind",
    tag="selection",
):
    """The weights a lane keeps: those whose `component.key` matches one of `include`'s fnmatch
    patterns, each of which must match one. Every other tensor is dropped; the kept ones keep the
    source's objects, and the pipeline, every component's config and its assets come along. A
    small lane of a big checkpoint's part, such as its multi-token-prediction head."""

    include: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DerivationHook:
    """A lane that needs code: `function(ctx, tel, source, output)` writes weights output
    `output` from `source` (`ctx.output`, `ctx.adopt_model`) and returns the model."""

    function: Callable[[Context, Telemetry, Weights[Any], str], ModelArtifact]


type Plan = QuantizationPlan | SelectionPlan | DerivationHook
#: `calibrate(ctx, tel, model, record)`: runs real inputs through `model` inside
#: `with record(components):`, `components` naming its modules (`{"transformer": module}`).
type Calibrate = Callable[[Context, Telemetry, Any, Callable[[Mapping[str, Any]], Any]], None]


@dataclass(frozen=True, slots=True)
class Recipe:
    """The recipe that writes `lane` from a checkpoint of `model`'s layout (`Model`: any)."""

    lane: str
    model: type[Model[Any]]
    plan: Plan
    #: A plan fitted to the model's activations: what runs real inputs through it (`Calibrate`).
    calibrate: Calibrate | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lane, str) or _OUTPUT_NAME.fullmatch(self.lane) is None:
            raise ConformanceError(
                f"recipe lane {self.lane!r} is not a portable name", code="recipe_declaration"
            )
        if not (isinstance(self.model, type) and issubclass(self.model, Model)):
            raise ConformanceError("a recipe's model is a Model class", code="recipe_declaration")
        if not isinstance(self.plan, (QuantizationPlan, SelectionPlan, DerivationHook)):
            raise ConformanceError(
                "a recipe's plan is a QuantizationPlan, SelectionPlan or DerivationHook",
                code="recipe_declaration",
            )
        if self.calibrate is not None and not (
            isinstance(self.plan, QuantizationPlan) and callable(self.calibrate)
        ):
            raise ConformanceError(
                "a recipe's calibrate is a function, beside a QuantizationPlan",
                code="recipe_declaration",
            )


class RecipeModel(msgspec.Struct, frozen=True, kw_only=True):
    """A recipe's model as the interface names it: its class and the components each stage
    reads, which a checkpoint must hold for the recipe to read it."""

    class_name: str = msgspec.field(name="class")
    component_use: dict[str, tuple[str, ...]] = msgspec.field(default_factory=dict)

    def components(self) -> set[str]:
        return {name for names in self.component_use.values() for name in names}


class DeclaredRecipe(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """One row of the package interface's `recipes`: a `Recipe` bound at `module`.`export`."""

    name: str
    lane: str
    module: str
    export: str
    model: RecipeModel
    plan: dict[str, Any]
    #: a declared plan's; a hook's or a calibrated plan's joins its release, so the run computes it
    fingerprint: str = ""
    #: `module:name` of a calibrated plan's `calibrate` function; "" for none
    calibrate: str = ""


def outputs(lanes: Iterable[str]) -> tuple[WeightsOutput, ...]:
    """`quantize`'s weights outputs: one per lane its recipes write."""
    return tuple(
        WeightsOutput(lane, max_new_bytes=MAX_WEIGHTS_NEW_BYTES) for lane in sorted(set(lanes))
    )


def canonical(plan: Plan) -> dict[str, Any]:
    """`plan` as JSON: its fields without defaults, tagged by `kind`; a hook by its function."""
    if isinstance(plan, DerivationHook):
        return {"kind": "derivation", "hook": identity(plan.function)}
    decoded: dict[str, Any] = msgspec.json.decode(msgspec.json.encode(plan))
    return decoded


def fingerprint(
    plan: dict[str, Any], package: str = "", release: str = "", calibrate: str = ""
) -> str:
    """The digest of a canonical plan. A hook, or a plan's `calibrate`, is code, so its package
    and release join it."""
    if plan.get("kind") == "derivation":
        return canonical_json.digest({**plan, "package": package, "release": release})
    if calibrate:
        coded = {"calibrate": calibrate, "package": package, "release": release}
        return canonical_json.digest({**plan, **coded})
    return canonical_json.digest(plan)


def identity(function: Callable[..., Any]) -> str:
    """A module-level function as a recipe names it: `module:name`."""
    return f"{function.__module__}:{function.__name__}"
