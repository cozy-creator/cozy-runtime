"""Author-surface arms over the REAL construction path — no mocks, no doubles of ours.

`ModelRegistry.acquire` -> `Loader.construct` -> `census` -> the scheduler scan is the
production serving path, driven here weightlessly: the package's own pipeline is an
ordinary object whose components answer `state_dict()`, which is all the census can see
(§1.1). Nothing is patched and nothing is stubbed; the only thing missing is torch, and
the construction plane never needed it.

Each arm below refuses on the commit before cr-060's corrective. What the arms cover:

* `test_a_component_survives_generic_introspection` — the recorded incident (#573). A
  censused component used to be replaced by a proxy that poisoned the whole generation on
  any attribute probe, so `hasattr` — every duck-type check in the ecosystem, including
  the one `typing` itself performs for a runtime-checkable Protocol — killed a live model
  and named a component no package had touched.
* `test_a_component_is_still_itself` — the same proxy made a real component INVISIBLE to
  `isinstance` and to the runtime's own `census`, which is a silent wrongness rather than
  a loud one.
* `test_a_declared_scope_still_leases_and_still_fences` — what was KEPT: the residency
  lease that admits before the method body runs, and the request-state-escape fence.
* `test_a_job_model_is_a_plain_manifest_record` — the derive-only Model is a record now,
  not an instance whose every attribute read ran an interception.
* `test_a_method_may_not_declare_a_runtime_decision` and
  `test_a_model_may_not_name_an_artifact_identifier` — the two §1.1 refusals a PACKAGE
  meets in band, driven through the real decorator and the real constructor. They are
  behaviour, so they are tested here rather than scanned; `checks/architecture.py` holds
  only the structural half, over the runtime's own published definitions.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import (
    CapabilityError,
    ConformanceError,
    Context,
    Device,
    FileAsset,
    InvalidRequest,
    Invocation,
    Model,
    ModelRegistry,
    decode_request,
    invocable,
    uses_components,
)
from cozy_runtime.author._loader import (
    Artifact,
    Config,
    Loader,
    ModuleLike,
    Scheduler,
    census,
)
from cozy_runtime.author._model import _derive_model, _public_methods
from cozy_runtime.author.fakes import (
    fake_attempt,
    fake_context,
    fake_input,
    fake_outputs,
    warm_with_fakes,
)

MANIFEST = "sha256:" + "ab" * 32


class Component:
    """A weightless component root: the census sees exactly `state_dict()` (§1.1)."""

    def __init__(self) -> None:
        self.calls = 0

    def state_dict(self) -> Mapping[str, object]:
        return {}

    def __call__(self) -> str:
        self.calls += 1
        return "denoised"


class Sampler:
    """A structural `Scheduler`: the construction fact the scheduler scan reads."""

    @property
    def config(self) -> Mapping[str, object]:
        return {"kind": "euler"}

    @property
    def compatibles(self) -> tuple[str, ...]:
        return ("euler", "heun")

    def clone(self, sampler: str | None = None) -> Sampler:
        return Sampler()


class Pipeline:
    def __init__(self) -> None:
        self.components: dict[str, object] = {
            "transformer": Component(),
            "scheduler": Sampler(),
        }
        self.seen: Context | None = None


class Denoiser(Model[Pipeline]):
    pipeline: Pipeline

    def load(self, loader: Loader) -> None:
        self.pipeline = loader.construct(Pipeline, factory=lambda config: Pipeline())

    @uses_components("transformer")
    def denoise(self) -> str:
        component = self.pipeline.components["transformer"]
        assert callable(component)
        return str(component())


class Lease:
    """A `Residency` that RECORDS. The lease is the real one — `_ComponentScope` calls it
    before the method body runs — and recording it is how the arm reads the ordering."""

    def __init__(self) -> None:
        self.admitted: list[tuple[str, tuple[str, ...]]] = []
        self.released: list[tuple[str, tuple[str, ...]]] = []

    def admit(self, method: str, components: tuple[str, ...]) -> None:
        self.admitted.append((method, components))

    def release(self, method: str, components: tuple[str, ...]) -> None:
        self.released.append((method, components))


def _ctx() -> Context:
    return Context(request_id="cr-060", deadline=time.monotonic() + 60.0)


def _generation(residency: Lease | None = None) -> Denoiser:
    registry = ModelRegistry(release="cr-060-arm")
    artifact = Artifact(snapshot=MANIFEST, tensor_schema={}, config=Config({}))
    model = registry.acquire("serve.models.model", Denoiser, artifact)
    if residency is not None:
        object.__setattr__(model, "_cozy_residency", residency)
    return model


def test_a_component_survives_generic_introspection() -> None:
    """#573: a probe on a constructed component must not kill the generation.

    `hasattr` is what `typing` runs for a runtime-checkable Protocol, what every library
    duck-type check runs, and what the runtime's own scheduler scan ran when this fired.
    """
    model = _generation()
    component = model.pipeline.components["transformer"]

    assert hasattr(component, "state_dict") is True
    assert hasattr(component, "config") is False
    assert isinstance(component, Scheduler) is False

    # The generation is still alive: the probe cost it nothing.
    assert model.denoise() == "denoised"


def test_a_component_is_still_itself() -> None:
    """A censused component answers `isinstance` and the runtime's own walk, as itself."""
    model = _generation()
    component = model.pipeline.components["transformer"]

    assert isinstance(component, ModuleLike)
    assert isinstance(component, Component)
    assert census(model.pipeline).components == ("transformer",)
    assert census(model.pipeline).schedulers == ("scheduler",)
    assert set(model._cozy_schedulers) == {"scheduler"}


def test_a_declared_scope_still_leases_and_still_fences() -> None:
    """What cr-060 KEEPS: admission before the body, release after, and no state escape."""
    lease = Lease()
    model = _generation(lease)

    assert model.denoise() == "denoised"
    assert lease.admitted == [("denoise", ("transformer",))]
    assert lease.released == [("denoise", ("transformer",))]

    view = model.for_request(_ctx())
    with pytest.raises(CapabilityError) as escape:
        model.leaked = view
    assert escape.value.code == "request_state_escape"

    with pytest.raises(CapabilityError) as frozen:
        model.extra = object()
    assert frozen.value.code == "persistent_allocation"


def test_a_job_model_is_a_plain_manifest_record() -> None:
    """A derive-only job Model carries one Manifest identity and constructs nothing."""
    job_model = _derive_model(Denoiser, MANIFEST)

    assert isinstance(job_model, Denoiser)
    assert job_model.checkpoint_ref == MANIFEST

    # No construction ran, so the component is ABSENT — the ordinary Python answer, named
    # by the ordinary Python error, with no interception installed to say it differently.
    with pytest.raises(AttributeError, match="pipeline"):
        _ = job_model.pipeline

    # And no generation is Ready, which is what refuses the serving surface.
    with pytest.raises(CapabilityError) as not_ready:
        job_model.for_request(_ctx())
    assert not_ready.value.code == "not_ready"


def test_a_saved_output_becomes_a_granted_input(tmp_path: Path) -> None:
    """`fake_input` is the round trip the executor performs: one attempt's OUTPUT is the
    next attempt's granted input, hydrated through the same `bind` with the same facts."""
    out = fake_outputs(fake_attempt("first", spool=tmp_path / "first"))
    saved = out.save_bytes(b"payload bytes", media_type="application/octet-stream")

    granted = fake_input(saved, attempt="second", input_id="document", max_decoded_bytes=1 << 20)

    assert type(granted) is type(saved)
    assert granted is not saved
    assert granted.ref == saved.ref
    assert granted.hydrated
    assert granted.digest == saved.digest
    assert granted.size_bytes == saved.size_bytes
    assert granted.read_bytes() == b"payload bytes"

    with pytest.raises(CapabilityError):
        fake_input(FileAsset("file:never-hydrated"), attempt="second")


def test_a_request_decodes_only_through_the_supported_path() -> None:
    """`decode_request` is the one wire path: a ref STRING hydrates nothing, and anything
    that does not validate arrives as `InvalidRequest` rather than a msgspec error."""

    class Request(msgspec.Struct):
        document: FileAsset
        count: int = 1

    decoded = decode_request(Request, {"document": "file:abc", "count": 3})
    assert decoded.document.ref == "file:abc"
    assert not decoded.document.hydrated
    assert decoded.count == 3

    with pytest.raises(InvalidRequest):
        decode_request(Request, {"document": {"ref": "file:abc", "local": "/etc/passwd"}})
    with pytest.raises(InvalidRequest):
        decode_request(Request, {"document": "file:abc", "count": "many"})


def test_a_method_may_not_declare_a_runtime_decision() -> None:
    """A model method that offers a placement knob is refused where it is WRITTEN (§1.1)."""
    with pytest.raises(ConformanceError) as refused:

        class Placed(Model[Pipeline]):
            @uses_components("transformer")
            def denoise(self, device: str = "cuda") -> str:
                return device

    assert refused.value.code == "excluded_keyword"
    assert refused.value.fields == ("device",)


def test_a_model_may_not_name_an_artifact_identifier() -> None:
    """Identity is deployment-side, so the one surface that takes a ref refuses one (§1.1)."""
    with pytest.raises(ConformanceError) as refused:
        Denoiser.for_test(checkpoint_ref="stabilityai/stable-diffusion-xl-base-1.0")
    assert refused.value.code == "identifier_in_code"

    with pytest.raises(ConformanceError) as scheme:
        Denoiser.for_test(checkpoint_ref="hf:org/model@v2")
    assert scheme.value.code == "identifier_in_code"

    # And the derive-only door takes a Manifest digest, never a name.
    with pytest.raises(ConformanceError) as derived:
        _derive_model(Denoiser, "stabilityai/stable-diffusion-xl-base-1.0")
    assert derived.value.code == "derive_model_manifest_invalid"


# --------------------------------------------------------------------------- warm (cr-110)


class Warmed(Model[Pipeline]):
    """A model whose `warm` takes ONE dry step through its declared scope (model-lifecycle.md)."""

    pipeline: Pipeline

    def load(self, loader: Loader) -> None:
        self.pipeline = loader.construct(Pipeline, factory=lambda config: Pipeline())

    def warm(self, ctx: Context) -> None:
        self.pipeline.seen = ctx  # on the pipeline: the model itself stays immutable
        self.denoise()

    @uses_components("transformer")
    def denoise(self) -> str:
        component = self.pipeline.components["transformer"]
        assert callable(component)
        return str(component())


@invocable
async def child(ctx: Context, *, seed: int) -> int:
    """A package call: reservable only under an admitted attempt's broker."""
    return seed


def test_warm_is_a_lifecycle_member_and_not_a_scope() -> None:
    """`warm` sits beside `load`/`unload`: `@uses_components` refuses it, and the base
    class's is a no-op so declaring one is the author's choice."""
    with pytest.raises(ConformanceError) as refused:

        class Scoped(Model[Pipeline]):
            @uses_components("transformer")
            def warm(self, ctx: Context) -> None:
                return None

    assert refused.value.code == "component_placement"
    Model().warm(fake_context())  # the base is a no-op: declaring one is the author's choice
    assert "warm" not in [name for name, _ in _public_methods(Warmed)]


def test_warm_runs_under_a_context_with_no_attempt_behind_it() -> None:
    """The Context `warm` receives has a device, a deadline and cancellation and nothing
    request-shaped: no id, no adapters, no `boot_warmup` flag anywhere on the surface —
    and an invocable called inside it refuses because no broker is bound, not because a
    flag says so."""
    model = Warmed.for_test(pipeline=Pipeline())

    ctx = warm_with_fakes(model, device=Device("cuda", 0))

    assert ctx.request_id == "" and str(ctx.device) == "cuda:0" and not ctx.cancelled
    assert ctx._adapters == ()
    assert not hasattr(ctx, "boot_warmup") and not hasattr(Invocation, "boot_warmup")
    # The dry step opened its declared scope exactly as a handler would, and the harness
    # recorded it — which is how a package test proves what its warm touches.
    model.harness.assert_scopes("denoise")
    assert model.pipeline.seen is ctx

    class Calling(Model[Pipeline]):
        pipeline: Pipeline

        def warm(self, ctx: Context) -> None:
            child(seed=1)

    with pytest.raises(CapabilityError) as refused:
        warm_with_fakes(Calling.for_test(pipeline=Pipeline()))
    assert refused.value.code == "child_broker_absent"


def test_warm_may_not_grow_the_model() -> None:
    """The generation stays immutable through `warm`: in-place work on the modules is the
    whole point, new state on the model is the fence `load` already answers (§1.3)."""

    class Growing(Model[Pipeline]):
        pipeline: Pipeline

        def warm(self, ctx: Context) -> None:
            self.tables = object()

    with pytest.raises(CapabilityError) as refused:
        warm_with_fakes(Growing.for_test(pipeline=Pipeline()))
    assert refused.value.code == "persistent_allocation"
