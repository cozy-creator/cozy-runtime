"""The worker's commands to its executor, decoded once where the executor reads them.

Each struct is the canonical-JSON frame `seam.py` carries, tagged by `cmd`. An omitted
field is its default, so an executor pinned to an older Runtime reads the same bytes, and
a field a newer worker adds is ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, get_args

import msgspec
from msgspec import UNSET, UnsetType

from cozy_runtime.author._assets import FileState, GrantedInput, InputMetadata
from cozy_runtime.author._services import MAX_OUTPUT_BYTES
from cozy_runtime.internal.canonical import Json

_MANIFEST = Annotated[str, msgspec.Meta(pattern="^sha256:[0-9a-f]{64}$")]

type Custody = Literal["canonical", "local"]


class Command(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True, tag_field="cmd"):
    """One worker command; `cmd` names it."""


class Hello(Command, frozen=True, kw_only=True, tag="hello"):
    pass


class Shutdown(Command, frozen=True, kw_only=True, tag="shutdown"):
    pass


class Probe(Command, frozen=True, kw_only=True, tag="probe"):
    #: collect cyclic garbage first: a cycle holding a device tensor is not a leak
    collect: bool = False


class Residency(Command, frozen=True, kw_only=True, tag="residency"):
    pass


class Warm(Command, frozen=True, kw_only=True, tag="warm"):
    """An older worker's startup warm. Never sent now; answered so that worker still runs."""


class Join(Command, frozen=True, kw_only=True, tag="join"):
    port: int
    backend: str = "nccl"


class DescribeInstalled(Command, frozen=True, kw_only=True, tag="describe_installed"):
    distribution: str
    output: str
    application: str = ""


class Vacate(Command, frozen=True, kw_only=True, tag="vacate"):
    residents: bool = True
    #: the group ranks that give their weights back; empty is every rank
    ranks: tuple[int, ...] = ()


class Restore(Command, frozen=True, kw_only=True, tag="restore"):
    names: tuple[str, ...] = ()


class Budget(Command, frozen=True, kw_only=True, tag="budget"):
    """This executor's weight-plane budgets (`weight_plane/1`). Lowering unmaps at once."""

    #: -1 is no grant: each stage derives its own from what the driver has free
    vram_bytes: int
    #: -1 leaves the pinned host tier's budget as it is
    pinned_bytes: int = -1


class Prefetch(Command, frozen=True, kw_only=True, tag="prefetch"):
    """Fill one loaded construction's host tier now, before its attempt holds the device."""

    construction: str


class Unload(Command, frozen=True, kw_only=True, tag="unload"):
    construction: str


class Activate(Command, frozen=True, kw_only=True, tag="activate"):
    construction: str
    authorized_device_limit_bytes: int | UnsetType = UNSET
    #: A follower's half: the constructions rank 0 parked to make room.
    park: tuple[str, ...] = ()


class Attention(Command, frozen=True, kw_only=True, tag="attention"):
    """A follower's attention: adopt rank 0's kernels, hold a `pin`, or restore (neither)."""

    construction: str = ""
    pin: str | UnsetType = UNSET
    model_key: str | None = None
    adopt: dict[str, str] | UnsetType = UNSET


class Release(msgspec.Struct, frozen=True, kw_only=True):
    """Where a package's code and interface live on this worker."""

    application: str = ""
    package_interface: str = ""


class MemoSettings(msgspec.Struct, frozen=True, kw_only=True):
    """The Worker's answer to an executor that lists `memo: stage/1` (tracker #298)."""

    #: The executor's in-memory tier; 0 turns it off, -1 is the executor's own default.
    process_bytes: int = -1
    #: The largest result the Worker's machine tier takes; 0 turns that tier off.
    entry_bytes: int = 0


class Start(Command, frozen=True, kw_only=True, tag="start"):
    devices: str = ""
    sequence_parallel_degree: int = 1
    application: str = ""
    package_interface: str = ""
    #: Import torch, the Runtime and the package and stop: no device is touched. Sent only to
    #: an executor whose hello lists `import_only`; an older one would ignore it and start.
    import_only: bool = False
    #: Absent from an older Worker: the executor keeps its in-memory tier at its default.
    memo: MemoSettings | None = None

    def release(self) -> Release:
        return Release(application=self.application, package_interface=self.package_interface)


class Adapter(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    model_id: str = ""
    ref: str = ""
    scale: float = 1.0
    kind: str = "lora"
    component: str = ""
    source_component: str = ""
    family: str = ""


class Binding(Release, frozen=True, kw_only=True):
    """One model's construction: its class, where its weights are, and how to fill them.

    A weightless binding names only its release and an empty `model_class`. Every field is
    written, defaults included: executors of older Runtimes index them directly.
    """

    model_class: str = ""
    model_binding_path: str = ""
    model_parameter_name: str = ""
    model_parameter_names: tuple[str, ...] = ()
    component: str = ""
    components: tuple[str, ...] = ()
    snapshots: dict[str, str] = {}
    store: str = ""
    snapshot: str = ""
    release: str = ""
    variant: str = ""
    development: bool = False
    custody: Custody = "canonical"
    objective: str = ""
    steps_basis: int = 0
    placement: str = ""
    package: str = ""
    model: str = ""
    adapters: tuple[Adapter, ...] = ()
    #: The read ring of executors before the weight plane; they index these directly.
    window_bytes: int = 4 << 20
    slots: int = 16
    readers: int = 16
    inflight: int = 8

    def parameter_names(self) -> tuple[str, ...]:
        """Every declared name for this shared construction, its primary first."""
        return tuple(
            dict.fromkeys(
                name for name in (self.model_parameter_name, *self.model_parameter_names) if name
            )
        )


class Budgets(msgspec.Struct, frozen=True, kw_only=True):
    # Older executors read the wire key `declared_weight_bytes`; 0.18.69 and earlier index it.
    logical_weight_bytes: int = msgspec.field(default=0, name="declared_weight_bytes")


class ModelLoad(msgspec.Struct, frozen=True, kw_only=True):
    binding: Binding
    budgets: Budgets = Budgets()


class GroupHardware(msgspec.Struct, frozen=True, kw_only=True):
    """Rank 0's hardware plan, broadcast with a group load."""

    degree: int
    devices: tuple[str, ...]


class Load(Command, frozen=True, kw_only=True, tag="load"):
    """Construct and fill one construction: one model (`binding`) or several (`models`)."""

    construction: str
    devices: str = ""
    authorized_device_limit_bytes: int | UnsetType = UNSET
    sequence_parallel_degree: int = 1
    attention_pin: str = ""
    #: The one-model form. Executors before the typed commands index both keys directly, so a
    #: worker always writes them for one model; a many-model load writes neither.
    binding: Binding | UnsetType = UNSET
    budgets: Budgets | UnsetType = UNSET
    models: tuple[ModelLoad, ...] | UnsetType = UNSET
    #: A follower's half: rank 0's hardware plan and, in a many-model load, which model.
    group: GroupHardware | UnsetType = UNSET
    model_key: str | UnsetType = UNSET
    #: The worker keeps pinned host tiers (`HostTier` exchanges); older workers do not.
    host_tier: bool = False
    #: The worker grants a turn per component-use scope (`stage/1`): `StageEnter` before each
    #: scope plans, `StageExit` after it. Older workers never set it.
    stages: bool = False

    def single(self) -> ModelLoad:
        return ModelLoad(
            binding=Binding() if isinstance(self.binding, UnsetType) else self.binding,
            budgets=Budgets() if isinstance(self.budgets, UnsetType) else self.budgets,
        )

    def rows(self) -> tuple[ModelLoad, ...]:
        return (self.single(),) if isinstance(self.models, UnsetType) else self.models


class InputFile(msgspec.Struct, frozen=True, kw_only=True):
    """One verified input file in the attempt's own spool."""

    local: str
    media_type: str
    digest: str
    length: int
    order: int = 0
    file_state: FileState | None = None

    def granted(self, input_id: str) -> GrantedInput:
        return GrantedInput(
            input_id=input_id,
            local=Path(self.local),
            file_state=self.file_state,
            media_type=self.media_type,
            digest=self.digest,
            length=self.length,
            order=self.order,
        )


class _Attempt(Command, frozen=True, kw_only=True):
    request_id: str
    #: The InvocationSpec's activation capture; the executor validates it against its models.
    capture: Json = None


class PrepareRequest(_Attempt, frozen=True, kw_only=True, tag="prepare_request"):
    entrypoint: str
    payload: Json
    construction: str = ""
    attention_kernel: str = ""
    input_metadata: dict[str, InputMetadata] = {}


class _Run(_Attempt, frozen=True, kw_only=True):
    spool: str
    deadline_s: float | None
    inputs: dict[str, InputFile] = {}
    #: Retained input trees by reference: (root, digest).
    trees: dict[str, tuple[str, str]] = {}
    max_output_bytes: int = MAX_OUTPUT_BYTES
    max_input_bytes: int | None = None


class Invoke(_Run, frozen=True, kw_only=True, tag="invoke"):
    entrypoint: str
    construction: str = ""
    attention_kernel: str = ""
    #: The weight plane's device budget for this attempt; -1: the executor derives it.
    plane_budget_bytes: int = -1
    #: Each component-use scope asks for its turn (`stage/1`, as `Load.stages`).
    stages: bool = False
    #: Read only by executors before the weight plane.
    placement: str = "all_resident"
    headroom_bytes: int = 0
    scope_headroom_bytes: dict[str, int] = {}
    measured_scopes: tuple[str, ...] = ()


class ModelManifest(msgspec.Struct, frozen=True, kw_only=True, rename={"class_key": "class"}):
    """One job Model parameter: a derive-only Manifest capability."""

    class_key: str
    manifest: _MANIFEST
    length: Annotated[int, msgspec.Meta(gt=0)]


class JobBudget(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    gpu_rate_micro_usd_per_hour: int = 0
    gpu_count: int = 0
    cap_micro_usd: int = 0


class CallInterface(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """One callable dependency the job may call, and how this executor binds it."""

    module: str
    export: str
    interface_digest: str = ""
    interface_document: Json = None
    interface_path: str = ""
    native_source: bool = False
    native_effect: bool = False
    request_fields: tuple[str, ...] | None = None
    unavailable: str = ""
    builtin: str = ""
    self_call: bool = msgspec.field(default=False, name="self")
    kind: Literal["entrypoint", "job"] = "job"


class RunJob(_Run, frozen=True, kw_only=True, tag="run_job"):
    job: str
    payload: Json
    application: str
    package_interface: str
    scratch: str = ""
    models: dict[str, ModelManifest] = {}
    call_interfaces: tuple[CallInterface, ...] = ()
    publish_to: str = ""
    budget: JobBudget = JobBudget()


type KnownCommand = (
    Hello
    | Shutdown
    | Probe
    | Residency
    | Warm
    | Join
    | DescribeInstalled
    | Vacate
    | Restore
    | Budget
    | Prefetch
    | Unload
    | Activate
    | Attention
    | Start
    | Load
    | PrepareRequest
    | Invoke
    | RunJob
)


#: Every command name this Runtime's executor answers.
NAMES = frozenset(str(kind.__struct_config__.tag) for kind in get_args(KnownCommand.__value__))


def decode(frame: object) -> KnownCommand:
    """The command a frame states; `msgspec.ValidationError` names what it lacks."""
    command: KnownCommand = msgspec.convert(frame, KnownCommand)
    return command


def encode(command: Command) -> dict[str, object]:
    """The frame for one command, with its `cmd` tag and without its defaults."""
    frame = msgspec.to_builtins(command)
    assert isinstance(frame, dict)
    return frame
