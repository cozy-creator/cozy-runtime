"""Static model-production declarations.

A production is reviewed orchestration data in the package descriptor.  It executes
nothing and owns no source, destination, credential, release, or publication authority.
Every step names an ordinary installed job and joins model/component outputs by stable
names; Creator resolves the exact job releases and schedules the graph.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import Resources

MAX_PRODUCTIONS = 16
MAX_STEPS = 64
MAX_STEP_INPUTS = 16
MAX_STEP_OUTPUTS = 8
MAX_OUTPUTS = 16
MAX_CONTRACT_VALUES = 32

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_CALLABLE = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,63}/[a-z0-9][a-z0-9._-]{0,63}"
    r"@v(?:0|[1-9][0-9]*)/"
    r"[a-z][a-z0-9_-]{0,63}$"
)
_CONTRACT_VALUE = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,63}(?:/[a-z0-9][a-z0-9._+-]{0,63})*$")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _name(value: str, where: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ConformanceError(
            f"{where} {value!r} is not a portable production name",
            code="production_name",
            fields=[where],
        )


def _unique(values: Sequence[str], where: str, cap: int, *, nonempty: bool = True) -> None:
    if (nonempty and not values) or len(values) > cap:
        lower = 1 if nonempty else 0
        raise ConformanceError(
            f"{where} has {len(values)} entries; the bound is {lower}..={cap}",
            code="production_bound",
            fields=[where],
        )
    for value in values:
        _name(value, where)
    if len(set(values)) != len(values):
        raise ConformanceError(
            f"{where} contains a duplicate", code="production_duplicate", fields=[where]
        )


@dataclass(frozen=True, slots=True)
class ProductionContract:
    """Facts Tensorhub must independently derive from one produced Manifest.

    Both fields are exact TensorFS identities: the final header topology digest and its
    complete EncodingSpec object-ID set. Human task/structure/stamp labels cannot make
    produced bytes admissible and have no place in this contract.
    """

    topology_digest: str
    encodings: tuple[str, ...]

    def __post_init__(self) -> None:
        if _DIGEST.fullmatch(self.topology_digest) is None:
            raise ConformanceError(
                f"production contract topology digest {self.topology_digest!r} is malformed",
                code="production_contract",
                fields=["topology_digest"],
            )
        if not self.encodings or len(self.encodings) > MAX_CONTRACT_VALUES:
            raise ConformanceError(
                f"production contract encodings has {len(self.encodings)} entries; "
                f"the bound is 1..={MAX_CONTRACT_VALUES}",
                code="production_bound",
                fields=["encodings"],
            )
        if tuple(sorted(set(self.encodings))) != self.encodings or any(
            _DIGEST.fullmatch(value) is None for value in self.encodings
        ):
            raise ConformanceError(
                "production contract encodings must be sorted unique exact EncodingSpec object IDs",
                code="production_contract",
                fields=["encodings"],
            )


@dataclass(frozen=True, slots=True)
class ProductionStep:
    """One ordinary job invocation in a static production graph.

    ``models`` maps the referenced job's model input name to either one production source
    slot or ``<step>.<output>``. Static recipe assets are ordinary importlib resources owned
    internally by the installed job, not Creator inputs or graph edges.
    """

    name: str
    callable: str
    models: Mapping[str, str]
    outputs: tuple[str, ...]
    resources: Resources | None = None

    def __post_init__(self) -> None:
        _name(self.name, "production step")
        if _CALLABLE.fullmatch(self.callable) is None:
            raise ConformanceError(
                f"production step {self.name} callable {self.callable!r} is not "
                "org/package@vN/function",
                code="production_callable",
                fields=[self.name],
            )
        if not self.models or len(self.models) > MAX_STEP_INPUTS:
            raise ConformanceError(
                f"production step {self.name} has {len(self.models)} model inputs; "
                f"the bound is 1..={MAX_STEP_INPUTS}",
                code="production_bound",
                fields=[self.name],
            )
        for input_name, source in self.models.items():
            _name(input_name, f"production step {self.name} model input")
            if not isinstance(source, str) or not source:
                raise ConformanceError(
                    f"production step {self.name} input {input_name} has no source",
                    code="production_edge",
                    fields=[self.name, input_name],
                )
        _unique(self.outputs, f"production step {self.name} outputs", MAX_STEP_OUTPUTS)


@dataclass(frozen=True, slots=True)
class ProductionOutput:
    """One named checkpoint output and its required artifact contract."""

    name: str
    source: str
    required: ProductionContract

    def __post_init__(self) -> None:
        _name(self.name, "production output")
        if not isinstance(self.source, str) or "." not in self.source:
            raise ConformanceError(
                f"production output {self.name} source {self.source!r} is not <step>.<output>",
                code="production_edge",
                fields=[self.name],
            )


@dataclass(frozen=True, slots=True)
class Production:
    name: str
    sources: Mapping[str, str]
    steps: tuple[ProductionStep, ...]
    outputs: tuple[ProductionOutput, ...]

    def __post_init__(self) -> None:
        _name(self.name, "production")
        if not self.sources or len(self.sources) > MAX_STEP_INPUTS:
            raise ConformanceError(
                f"production {self.name} has {len(self.sources)} source slots; "
                f"the bound is 1..={MAX_STEP_INPUTS}",
                code="production_bound",
                fields=[self.name, "sources"],
            )
        for slot, profile in self.sources.items():
            _name(slot, f"production {self.name} source slot")
            if not isinstance(profile, str) or _CONTRACT_VALUE.fullmatch(profile) is None:
                raise ConformanceError(
                    f"production {self.name} source slot {slot!r} has malformed "
                    f"TensorFS profile {profile!r}",
                    code="production_source_profile",
                    fields=[self.name, slot],
                )
        if not self.steps or len(self.steps) > MAX_STEPS:
            raise ConformanceError(
                f"production {self.name} has {len(self.steps)} steps; the bound is 1..={MAX_STEPS}",
                code="production_bound",
                fields=[self.name],
            )
        names = [step.name for step in self.steps]
        if len(set(names)) != len(names):
            raise ConformanceError(
                f"production {self.name} has duplicate step names",
                code="production_duplicate",
                fields=[self.name],
            )
        if not self.outputs or len(self.outputs) > MAX_OUTPUTS:
            raise ConformanceError(
                f"production {self.name} has {len(self.outputs)} required outputs; "
                f"the bound is 1..={MAX_OUTPUTS}",
                code="production_bound",
                fields=[self.name],
            )
        output_names = [output.name for output in self.outputs]
        if len(set(output_names)) != len(output_names):
            raise ConformanceError(
                f"production {self.name} repeats an output name",
                code="production_duplicate",
                fields=[self.name],
            )
        self._validate_graph()

    def _validate_graph(self) -> None:
        by_name = {step.name: step for step in self.steps}
        dependencies: dict[str, set[str]] = {step.name: set() for step in self.steps}

        def resolve(reference: str, owner: str) -> str | None:
            if reference in self.sources:
                return None
            step_name, separator, output = reference.partition(".")
            step = by_name.get(step_name)
            if not separator or step is None or output not in step.outputs:
                raise ConformanceError(
                    f"production {self.name} {owner} references unknown edge {reference!r}",
                    code="production_edge",
                    fields=[self.name, owner],
                )
            return step_name

        for step in self.steps:
            for input_name, reference in step.models.items():
                dependency = resolve(reference, f"{step.name}.{input_name}")
                if dependency is not None:
                    dependencies[step.name].add(dependency)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ConformanceError(
                    f"production {self.name} contains a cycle through {name}",
                    code="production_cycle",
                    fields=[self.name, name],
                )
            if name in visited:
                return
            visiting.add(name)
            for dependency in dependencies[name]:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        required_steps: set[str] = set()

        def retain(name: str) -> None:
            if name in required_steps:
                return
            required_steps.add(name)
            for dependency in dependencies[name]:
                retain(dependency)

        for output in self.outputs:
            selected_step = resolve(output.source, f"output {output.name}")
            assert selected_step is not None
            visit(selected_step)
            retain(selected_step)
        disconnected = sorted(set(by_name) - required_steps)
        if disconnected:
            raise ConformanceError(
                f"production {self.name} has disconnected step(s): {', '.join(disconnected)}",
                code="production_disconnected",
                fields=[self.name],
            )
