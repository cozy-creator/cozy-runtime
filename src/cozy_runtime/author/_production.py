"""Static model-production declarations.

A production is reviewed orchestration data in the package descriptor.  It executes
nothing and owns no source, destination, credential, release, or publication authority.
Every node names an ordinary installed job and joins model/component outputs by stable
names; Creator resolves the exact job releases and schedules the graph.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import Resources

MAX_PRODUCTIONS = 16
MAX_NODES = 64
MAX_NODE_INPUTS = 16
MAX_NODE_OUTPUTS = 8
MAX_OUTPUTS = 16
MAX_CONTRACT_VALUES = 32

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_CALLABLE = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,63}/[a-z0-9][a-z0-9._-]{0,63}/"
    r"[a-z][a-z0-9_-]{0,63}$"
)
_MEMBER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_CONTRACT_VALUE = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,63}(?:/[a-z0-9][a-z0-9._+-]{0,63})*$")


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

    ``tasks`` is the complete task-capability set, ``structure`` is the model-family
    structure contract, and ``encodings`` is the complete stored encoding set.  ``stamps``
    carries the remaining open, package-owned compatibility vocabulary.  These are
    requirements to verify, never labels that can make incompatible bytes admissible.
    """

    tasks: tuple[str, ...]
    structure: str
    encodings: tuple[str, ...]
    stamps: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _unique(self.tasks, "production contract tasks", MAX_CONTRACT_VALUES)
        if _CONTRACT_VALUE.fullmatch(self.structure) is None:
            raise ConformanceError(
                f"production contract structure {self.structure!r} is malformed",
                code="production_contract",
                fields=["structure"],
            )
        if not self.encodings or len(self.encodings) > MAX_CONTRACT_VALUES:
            raise ConformanceError(
                f"production contract encodings has {len(self.encodings)} entries; "
                f"the bound is 1..={MAX_CONTRACT_VALUES}",
                code="production_bound",
                fields=["encodings"],
            )
        if len(set(self.encodings)) != len(self.encodings) or any(
            _CONTRACT_VALUE.fullmatch(value) is None for value in self.encodings
        ):
            raise ConformanceError(
                "production contract encodings must be unique versioned contract values",
                code="production_contract",
                fields=["encodings"],
            )
        if len(self.stamps) > MAX_CONTRACT_VALUES:
            raise ConformanceError(
                f"production contract has {len(self.stamps)} stamps; "
                f"the bound is {MAX_CONTRACT_VALUES}",
                code="production_bound",
                fields=["stamps"],
            )
        for key, value in self.stamps.items():
            _name(key, "production contract stamp")
            _name(value, f"production contract stamp {key}")


@dataclass(frozen=True, slots=True)
class ProductionNode:
    """One ordinary job invocation in a static production graph.

    ``models`` maps the referenced job's model input name to either one production source
    slot or ``<node>.<output>``. ``assets`` maps typed input names to package-relative
    immutable members; the package builder, not the job, resolves their bytes.
    """

    name: str
    callable: str
    models: Mapping[str, str]
    outputs: tuple[str, ...]
    assets: Mapping[str, str] = field(default_factory=dict)
    resources: Resources | None = None

    def __post_init__(self) -> None:
        _name(self.name, "production node")
        if _CALLABLE.fullmatch(self.callable) is None:
            raise ConformanceError(
                f"production node {self.name} callable {self.callable!r} is not "
                "org/package/function",
                code="production_callable",
                fields=[self.name],
            )
        if not self.models or len(self.models) > MAX_NODE_INPUTS:
            raise ConformanceError(
                f"production node {self.name} has {len(self.models)} model inputs; "
                f"the bound is 1..={MAX_NODE_INPUTS}",
                code="production_bound",
                fields=[self.name],
            )
        for input_name, source in self.models.items():
            _name(input_name, f"production node {self.name} model input")
            if not isinstance(source, str) or not source:
                raise ConformanceError(
                    f"production node {self.name} input {input_name} has no source",
                    code="production_edge",
                    fields=[self.name, input_name],
                )
        _unique(self.outputs, f"production node {self.name} outputs", MAX_NODE_OUTPUTS)
        if len(self.assets) > MAX_NODE_INPUTS:
            raise ConformanceError(
                f"production node {self.name} has {len(self.assets)} static assets; "
                f"the bound is {MAX_NODE_INPUTS}",
                code="production_bound",
                fields=[self.name],
            )
        for input_name, member in self.assets.items():
            _name(input_name, f"production node {self.name} asset input")
            if (
                not isinstance(member, str)
                or _MEMBER.fullmatch(member) is None
                or member.startswith(("/", "~"))
                or ".." in member.split("/")
            ):
                raise ConformanceError(
                    f"production node {self.name} asset {member!r} is not a safe "
                    "package-relative member",
                    code="production_asset",
                    fields=[self.name, input_name],
                )


@dataclass(frozen=True, slots=True)
class ProductionOutput:
    """One required graph output and its recipe-owned stable catalog lane key."""

    name: str
    source: str
    lane_key: str
    required: ProductionContract

    def __post_init__(self) -> None:
        _name(self.name, "production output")
        if not isinstance(self.source, str) or "." not in self.source:
            raise ConformanceError(
                f"production output {self.name} source {self.source!r} is not <node>.<output>",
                code="production_edge",
                fields=[self.name],
            )
        _name(self.lane_key, f"production output {self.name} lane key")


@dataclass(frozen=True, slots=True)
class Production:
    name: str
    sources: Mapping[str, str]
    nodes: tuple[ProductionNode, ...]
    outputs: tuple[ProductionOutput, ...]

    def __post_init__(self) -> None:
        _name(self.name, "production")
        if not self.sources or len(self.sources) > MAX_NODE_INPUTS:
            raise ConformanceError(
                f"production {self.name} has {len(self.sources)} source slots; "
                f"the bound is 1..={MAX_NODE_INPUTS}",
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
        if not self.nodes or len(self.nodes) > MAX_NODES:
            raise ConformanceError(
                f"production {self.name} has {len(self.nodes)} nodes; the bound is 1..={MAX_NODES}",
                code="production_bound",
                fields=[self.name],
            )
        names = [node.name for node in self.nodes]
        if len(set(names)) != len(names):
            raise ConformanceError(
                f"production {self.name} has duplicate node names",
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
        lane_keys = [output.lane_key for output in self.outputs]
        if len(set(output_names)) != len(output_names) or len(set(lane_keys)) != len(lane_keys):
            raise ConformanceError(
                f"production {self.name} repeats an output name or lane key",
                code="production_duplicate",
                fields=[self.name],
            )
        self._validate_graph()

    def _validate_graph(self) -> None:
        by_name = {node.name: node for node in self.nodes}
        dependencies: dict[str, set[str]] = {node.name: set() for node in self.nodes}

        def resolve(reference: str, owner: str) -> str | None:
            if reference in self.sources:
                return None
            node_name, separator, output = reference.partition(".")
            node = by_name.get(node_name)
            if not separator or node is None or output not in node.outputs:
                raise ConformanceError(
                    f"production {self.name} {owner} references unknown edge {reference!r}",
                    code="production_edge",
                    fields=[self.name, owner],
                )
            return node_name

        for node in self.nodes:
            for input_name, reference in node.models.items():
                dependency = resolve(reference, f"{node.name}.{input_name}")
                if dependency is not None:
                    dependencies[node.name].add(dependency)

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

        required_nodes: set[str] = set()

        def retain(name: str) -> None:
            if name in required_nodes:
                return
            required_nodes.add(name)
            for dependency in dependencies[name]:
                retain(dependency)

        for output in self.outputs:
            selected_node = resolve(output.source, f"output {output.name}")
            assert selected_node is not None
            visit(selected_node)
            retain(selected_node)
        disconnected = sorted(set(by_name) - required_nodes)
        if disconnected:
            raise ConformanceError(
                f"production {self.name} has disconnected node(s): {', '.join(disconnected)}",
                code="production_disconnected",
                fields=[self.name],
            )
