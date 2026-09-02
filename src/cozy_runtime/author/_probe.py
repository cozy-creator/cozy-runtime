"""The in-job activation-probe surface — tier 2 over pipelines a job already constructed.

A quantization gate holds BOTH lanes as real constructed pipelines (its slots construct,
cr-073) and needs the teacher-forced per-layer map without owning the mechanism.
`probe_activations` is that door over `cozy_runtime.probe.run_probe`: it resolves the named
component in each lane, swaps a recording stand-in into the reference pipeline for exactly
the duration of `drive` — so the caller drives its ORDINARY generation path and every
invocation of that component is recorded — and returns the raw ``cozy.probe.report/1``
document. Stats only: the verdict is the gate's (cozy-eval's ladder), never the runtime's.

No torch here: the author fence admits schema libraries only, and every tensor-touching
step lives in `cozy_runtime.probe`. The stand-in intercepts nothing but the CALL — every
attribute read reaches the real component — and it exists only between swap and restore;
unlike the component proxy #573 deleted, it never survives into a live generation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from cozy_runtime.author._loader import ModuleLike
from cozy_runtime.author._model import Model
from cozy_runtime.probe import Lane, ProbeConfig, ProbeRefusal, run_probe


def probe_activations(
    reference: object,
    candidate: object,
    drive: Callable[[], None],
    *,
    component: str,
    reference_label: str = "reference",
    candidate_label: str = "candidate",
    zoom: str | None = None,
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """One whole tier-2 probe over two constructed lanes; the ``cozy.probe.report/1`` doc.

    ``reference``/``candidate`` are constructed pipelines (what ``loader.construct``
    returned — a ``components`` mapping or public module attributes name their parts), or
    the ``Model`` instances holding them. ``drive`` runs ONE ordinary free-running
    generation over the REFERENCE lane — the caller's own loop, on the gate's pinned
    inputs — and while it runs, ``component`` in that pipeline is the recording stand-in.
    ``zoom`` re-resolves one named tap subdivided when the map needs detail.
    """
    reference_slot = _resolve(reference, component, label=reference_label)
    candidate_slot = _resolve(candidate, component, label=candidate_label)
    reference_component = reference_slot.get()
    candidate_component = candidate_slot.get()

    def recorded_drive(recording: Callable[..., Any]) -> None:
        reference_slot.swap(_RecordingStandIn(reference_component, recording))
        try:
            drive()
        finally:
            reference_slot.swap(reference_component)

    return run_probe(
        Lane(reference_label, lambda: reference_component),
        Lane(candidate_label, lambda: candidate_component),
        recorded_drive,
        component=component,
        zoom=zoom,
        config=config,
    )


class _RecordingStandIn:
    """The swap-scoped component stand-in: attributes are the real component's, the CALL
    is the probe's recorder (which itself invokes the real component)."""

    __slots__ = ("_call", "_component")

    def __init__(self, component: object, call: Callable[..., Any]) -> None:
        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "_call", call)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return object.__getattribute__(self, "_call")(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_component"), name)


class _Slot:
    """Where one component lives in one lane: a ``components`` mapping entry or a public
    attribute. `swap` is how the recording stand-in enters and how the original returns."""

    __slots__ = ("_holder", "_label", "_name", "_table")

    def __init__(
        self,
        holder: object,
        name: str,
        *,
        label: str,
        table: MutableMapping[str, object] | None = None,
    ) -> None:
        self._holder = holder
        self._name = name
        self._label = label
        self._table = table

    def get(self) -> object:
        if self._table is not None:
            return self._table[self._name]
        return getattr(self._holder, self._name)

    def swap(self, value: object) -> None:
        if self._table is not None:
            self._table[self._name] = value
            return
        try:
            setattr(self._holder, self._name, value)
        except (AttributeError, TypeError) as exc:
            raise ProbeRefusal(
                f"lane {self._label!r}: component {self._name!r} cannot be swapped on "
                f"{type(self._holder).__name__} ({exc}) — the recording pass needs a "
                "components mapping or a writable attribute",
                code="component_swap_unsupported",
            ) from exc


def _resolve(holder: object, component: str, *, label: str) -> _Slot:
    """The census rule, applied to one lane: the ``components`` mapping names parts
    explicitly; otherwise a public ModuleLike attribute is one. A ``Model`` holder is
    searched one level down — its declared state holds the constructed pipeline."""
    direct = _locate(holder, component, label=label)
    if direct is not None:
        return direct
    if isinstance(holder, Model):
        found = [
            slot
            for name in sorted(dir(holder))
            if not (name.startswith("_") or hasattr(Model, name))
            and (slot := _locate(getattr(holder, name, None), component, label=label)) is not None
        ]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ProbeRefusal(
                f"lane {label!r}: {type(holder).__name__} holds {len(found)} members that "
                f"supply a component named {component!r} — pass the pipeline itself",
                code="component_ambiguous",
            )
    raise ProbeRefusal(
        f"lane {label!r}: {type(holder).__name__} supplies no component named {component!r} "
        "— components are the constructed pipeline's `components` mapping entries or its "
        "public module attributes",
        code="component_unresolved",
    )


def _locate(holder: object, component: str, *, label: str) -> _Slot | None:
    if holder is None:
        return None
    table = getattr(holder, "components", None)
    if isinstance(table, Mapping) and component in table:
        if not isinstance(table, MutableMapping):
            raise ProbeRefusal(
                f"lane {label!r}: the components mapping of {type(holder).__name__} is "
                "read-only, so the recording stand-in cannot enter it",
                code="component_swap_unsupported",
            )
        return _Slot(holder, component, label=label, table=table)
    member = getattr(holder, component, None)
    if isinstance(member, ModuleLike):
        return _Slot(holder, component, label=label)
    return None
