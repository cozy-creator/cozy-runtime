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
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import pytest

from cozy_runtime.author import (
    CapabilityError,
    Context,
    Model,
    ModelRegistry,
    uses_components,
)
from cozy_runtime.author._loader import Artifact, Config, ModuleLike, Scheduler, census
from cozy_runtime.author._model import _derive_model

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


class Denoiser(Model[Pipeline]):
    pipeline: Pipeline

    def load(self, loader: object) -> None:
        assert hasattr(loader, "construct")
        self.pipeline = loader.construct(  # type: ignore[attr-defined]
            Pipeline, factory=lambda config: Pipeline()
        )

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
    artifact = Artifact(snapshot=MANIFEST, topology={}, config=Config({}))
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
        model.leaked = view  # type: ignore[attr-defined]
    assert escape.value.code == "request_state_escape"

    with pytest.raises(CapabilityError) as frozen:
        model.extra = object()  # type: ignore[attr-defined]
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
