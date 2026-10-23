"""The machine's commands to its executor, decoded once where the executor reads them.

Each struct is the canonical-JSON frame `seam.py` carries, tagged by `cmd`; an omitted field
is its default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, get_args

import msgspec
from msgspec import UNSET, UnsetType

from cozy_runtime.author._assets import FileState, GrantedInput, InputMetadata
from cozy_runtime.author._services import MAX_OUTPUT_BYTES
from cozy_runtime.internal.attention import Choice
from cozy_runtime.internal.canonical import Json

_MANIFEST = Annotated[str, msgspec.Meta(pattern="^sha256:[0-9a-f]{64}$")]

type Custody = Literal["canonical", "local"]


class Command(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True, tag_field="cmd"):
    """One worker command; `cmd` names it."""


class TensorFSOptions(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Ordinary endpoint settings and an already supplied bearer credential."""

    endpoint: str
    token: str | None = None
    #: when `token` expires (0: unknown)
    expires_unix_ms: int = 0
    ca_der: str | None = None

    def __repr__(self) -> str:
        return f"TensorFSOptions(endpoint={self.endpoint!r}, token=<redacted>)"


class Supplied(msgspec.Struct, frozen=True, kw_only=True):
    """A signer's bearer for one endpoint, read off the seam in order (`Mux`): a `bearer` frame
    (TensorD's newer one, unanswered) or what a Load or Invoke carries."""

    actor: str = ""
    tensorfs: TensorFSOptions | None = None


class Hello(Command, frozen=True, kw_only=True, tag="hello"):
    pass


class Shutdown(Command, frozen=True, kw_only=True, tag="shutdown"):
    pass


class Probe(Command, frozen=True, kw_only=True, tag="probe"):
    #: collect cyclic garbage first: a cycle holding a device tensor is not a leak
    collect: bool = False


class Join(Command, frozen=True, kw_only=True, tag="join"):
    port: int
    backend: str = "nccl"
    #: A socket follows: a byte on it aborts this follower's collectives (`RankGroup.interrupt`).
    abort: bool = False
    #: The group forms again after a call ended short of memory (`mirror.mirror_component`).
    again: bool = False
    #: The NCCL route rank 0 chose from the group's measured links; set before joining.
    nccl: dict[str, str] = {}
    #: Every GPU then times NCCL against its copy engines (`copy_engine.calibrate`).
    calibrate: bool = False


class Fork(Command, frozen=True, kw_only=True, tag="fork"):
    """Fork an import-only executor into a new one on its own seam (`fork/1`)."""

    socket: str
    root: str
    #: The child's whole environment, as the machine seals a spawned executor's.
    environment: dict[str, str]


class Map(Command, frozen=True, kw_only=True, tag="map"):
    """TensorD's warming: one loaded construction's models at `level` (`Memory.load`): `gpu`
    is warm and fills free room now, `ram` leaves its resident pages idle."""

    construction: str
    level: Literal["gpu", "ram"] = "gpu"


class Unload(Command, frozen=True, kw_only=True, tag="unload"):
    construction: str


class RequestMemory(Command, frozen=True, kw_only=True, tag="request_memory"):
    construction: str
    request_id: str
    phase: Literal["begin", "plan", "end"]
    steps: tuple[tuple[str, ...], ...] = ()


class Activate(Command, frozen=True, kw_only=True, tag="activate"):
    construction: str
    authorized_device_limit_bytes: int | UnsetType = UNSET
    #: A follower's half: the constructions rank 0 parked to make room.
    park: tuple[str, ...] = ()


class Attention(Command, frozen=True, kw_only=True, tag="attention"):
    """A follower's attention: install rank 0's whole selection (`adopt`), hold a `pin`, or
    restore (neither)."""

    construction: str = ""
    pin: str | UnsetType = UNSET
    model_key: str | None = None
    adopt: Choice | UnsetType = UNSET
    #: rank 0 asks while it serves a request: this GPU opens no kernel on its own thread
    serving: bool = False


class Kernels(Command, frozen=True, kw_only=True, tag="kernels"):
    """A follower's deferred kernel installs (`Plan.when_ready`) at rank 0's boundary: which
    of `sets` this GPU has built, or, with `install`, install those rank 0 installed."""

    construction: str = ""
    model_key: str | None = None
    sets: tuple[str, ...] = ()
    install: bool = False


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


class CallInterface(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """One callable dependency an attempt may call, and how this executor binds it."""

    module: str
    export: str
    interface_document: Json = None
    interface_path: str = ""
    native_effect: bool = False
    builtin: str = ""
    self_call: bool = msgspec.field(default=False, name="self")


class Construction(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """One constructed model slot a session will load: its checkpoint and declared components."""

    #: `<fn>.models.<param>`
    path: str
    manifest: _MANIFEST
    #: the slot's `components=`; empty is every component
    components: tuple[str, ...] = ()


class Start(Command, frozen=True, kw_only=True, tag="start"):
    #: The session's GPUs; empty runs on the CPU and never opens CUDA (`accelerator=False`).
    devices: str = ""
    #: 2: leases beside two ledger descriptors per GPU; checked before reading either.
    memory_protocol: int = 0
    sequence_parallel_degree: int = 1
    application: str = ""
    package_interface: str = ""
    #: Import torch, the Runtime and the package and stop: no device is touched.
    import_only: bool = False
    #: Absent: the executor keeps its in-memory tier at its default.
    memo: MemoSettings | None = None
    #: Its GPUs' grants follow the frame, a table and a row description per GPU in `devices`
    #: order: rank 0 opens its memory on the first and hands follower r the r-th.
    ledgers: int = 0
    #: TensorD's namespace (its uid) each GPU's lease listener is named in: given only when it
    #: leases this executor's weight pages (memory protocol 2); none, its ledger rows alone.
    leases: int | None = None
    #: The callables its attempts may call: their proxies are installed before the package is
    #: imported, and every attempt gets the call broker over them.
    call_interfaces: tuple[CallInterface, ...] = ()
    #: What its loads will construct, known before anything heavy is imported.
    constructions: tuple[Construction, ...] = ()

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

    A weightless binding names only its release and an empty `model_class`.
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
    placement: str = ""
    package: str = ""
    model: str = ""
    adapters: tuple[Adapter, ...] = ()
    #: In a group, each placeable component's GPU (an index within it): 0 keeps it beside the
    #: serial scope, g >= 1 keeps it resident on GPU g, where its calls and remote scopes run.
    homes: dict[str, int] = {}

    def parameter_names(self) -> tuple[str, ...]:
        """Every declared name for this shared construction, its primary first."""
        return tuple(
            dict.fromkeys(
                name for name in (self.model_parameter_name, *self.model_parameter_names) if name
            )
        )


class ModelLoad(msgspec.Struct, frozen=True, kw_only=True):
    binding: Binding


class GroupHardware(msgspec.Struct, frozen=True, kw_only=True):
    """Rank 0's hardware plan, broadcast with a group load."""

    degree: int
    devices: tuple[str, ...]


#: An executor capability: `Load.attention_pin` is the request's pin, whose kernel build it
#: imports first; the construction itself still selects automatically.
LOAD_ATTENTION_PIN = "attention.load_pin/1"


class Load(Command, frozen=True, kw_only=True, tag="load"):
    """Ask Runtime to construct models and register their weights with its memory, read through
    TensorFS. The command names one model (`binding`) or several (`models`)."""

    construction: str
    tensorfs: TensorFSOptions | None = None
    #: the signer whose bearer `tensorfs` carries
    actor: str = ""
    devices: str = ""
    sequence_parallel_degree: int = 1
    #: the pin of the request this executor loads for (`LOAD_ATTENTION_PIN`)
    attention_pin: str = ""
    #: The one-model form; a many-model load names `models`.
    binding: Binding | UnsetType = UNSET
    models: tuple[ModelLoad, ...] | UnsetType = UNSET
    #: A follower's half: rank 0's hardware plan and, in a many-model load, which model.
    group: GroupHardware | UnsetType = UNSET
    model_key: str | UnsetType = UNSET

    def single(self) -> ModelLoad:
        return ModelLoad(binding=Binding() if isinstance(self.binding, UnsetType) else self.binding)

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
    #: The request's precision (16|8|4; 0: the rules choose): 16 holds every attention site at
    #: level 16 (`attention.EXACT`).
    bits: int = 0
    input_metadata: dict[str, InputMetadata] = {}


class Invoke(_Attempt, frozen=True, kw_only=True, tag="invoke"):
    entrypoint: str
    tensorfs: TensorFSOptions | None = None
    spool: str
    deadline_s: float | None
    inputs: dict[str, InputFile] = {}
    #: Retained input trees by reference: (root, digest).
    trees: dict[str, tuple[str, str]] = {}
    max_output_bytes: int = MAX_OUTPUT_BYTES
    #: None honors installed per-field bounds; zero retains the older default ceiling.
    max_input_bytes: int | None = None
    construction: str = ""
    #: The family whose GPUs its calls share: a wait on a child call yields its room to it.
    family: str = ""
    attention_kernel: str = ""
    bits: int = 0  # as `PrepareRequest.bits`
    #: Each `Weights[M]` parameter's checkpoint, opened directly through TensorFS.
    sources: dict[str, _MANIFEST] = {}
    #: The weights outputs this attempt may write, and each one's granted new bytes.
    weights: dict[str, int] = {}
    destination: str = ""
    actor: str = ""
    operation: str = ""


type KnownCommand = (
    Hello
    | Shutdown
    | Probe
    | Join
    | Fork
    | Map
    | Unload
    | RequestMemory
    | Activate
    | Attention
    | Kernels
    | Start
    | Load
    | PrepareRequest
    | Invoke
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
