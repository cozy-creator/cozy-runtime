"""The executor's durable requests to its worker, and the worker's answers.

The executor sends `{"event": "request", "seq": N, "kind": K, **body}` and blocks until
the worker answers `{"event": "answer", "seq": N, **answer}`; a `Handoff` answer is then
followed by one connected socket. Each side decodes what it receives once, here. Every
field is written, defaults included, and a field a newer peer adds is ignored: the
executor runs in its package's own environment, so either side may be the older Runtime.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, TypedDict, get_args

import msgspec
from msgspec import UNSET, UnsetType

from cozy_runtime.author._assets import FileState
from cozy_runtime.author._capture import ActivationCapture
from cozy_runtime.author._errors import CapabilityError, InvalidRequest

MAX_CALL_INDEX = (1 << 32) - 1


class _Request(msgspec.Struct, frozen=True, kw_only=True, tag_field="kind"):
    pass


class DeviceRoom(_Request, frozen=True, kw_only=True, tag="device_room"):
    """Idle co-tenants of the executor's devices give `free_bytes` back."""

    free_bytes: int


class ChildEvents(_Request, frozen=True, kw_only=True, tag="child_events"):
    """The socket the worker nudges when one of this attempt's calls moves."""


class Checkpoint(_Request, frozen=True, kw_only=True, tag="checkpoint"):
    operation_key: str
    logical_key: str
    content_digest: str
    length: int


class ChildCall(_Request, frozen=True, kw_only=True, tag="child_call"):
    call_index: Annotated[int, msgspec.Meta(ge=0, le=MAX_CALL_INDEX)]
    module: str
    export: str
    #: The callee's canonical request document.
    payload: str
    progress_label: str = ""
    capture: ActivationCapture | None = None


class _Indexed(_Request, frozen=True, kw_only=True):
    call_index: Annotated[int, msgspec.Meta(ge=0, le=MAX_CALL_INDEX)]


class ChildPoll(_Indexed, frozen=True, kw_only=True, tag="child_poll"):
    pass


class ChildCancel(_Indexed, frozen=True, kw_only=True, tag="child_cancel"):
    pass


class ChildForget(_Indexed, frozen=True, kw_only=True, tag="child_forget"):
    pass


class GpuRelease(_Request, frozen=True, kw_only=True, tag="gpu_release"):
    pass


class ModelPrefetch(_Request, frozen=True, kw_only=True, tag="model_prefetch"):
    module: str
    export: str
    #: The canonical `{slot: ModelArtifact | None}` document.
    payload: str


class PublishPart(msgspec.Struct, frozen=True, kw_only=True):
    """One part of a composite product: a spool file and the media time it adds."""

    local: str
    duration_us: Annotated[int, msgspec.Meta(ge=0)] = 0


class Publish(_Request, frozen=True, kw_only=True, tag="publish"):
    """Show one asset to the run's owner now, as a product of a declared output."""

    output: str
    label: str = ""
    asset_ref: str
    asset_kind: str
    media_type: str = ""
    size_bytes: int = 0
    digest: str = ""
    #: A composite product: its bytes are these parts concatenated in order.
    parts: tuple[PublishPart, ...] = ()


class TreeMember(_Request, frozen=True, kw_only=True, tag="tree_member"):
    tree: str
    path: str
    asset_kind: str
    max_bytes: int
    media_types: tuple[str, ...] = ()


class _Writer(msgspec.Struct, frozen=True, kw_only=True, tag_field="operation"):
    kind: Literal["weights_writer"] = "weights_writer"


class WriterSource(_Writer, frozen=True, kw_only=True, tag="source"):
    manifest: str


class WriterOutput(_Writer, frozen=True, kw_only=True, tag="output"):
    output_slot: str
    length: int


class WriterAdopt(_Writer, frozen=True, kw_only=True, tag="adopt"):
    output_slot: str
    transaction: str
    length: int


type _Tagged = (
    DeviceRoom
    | ChildEvents
    | Checkpoint
    | ChildCall
    | ChildPoll
    | ChildCancel
    | ChildForget
    | GpuRelease
    | ModelPrefetch
    | TreeMember
    | Publish
)
#: One request of a managed call to the worker's call lane.
type CallRequest = (
    ChildCall | ChildPoll | ChildCancel | ChildForget | ChildEvents | GpuRelease | ModelPrefetch
)
#: One request to the worker's native writer broker, tagged by `operation`.
type WriterRequest = WriterSource | WriterOutput | WriterAdopt
type Request = _Tagged | WriterRequest

_KINDS = frozenset(str(kind.__struct_config__.tag) for kind in get_args(_Tagged.__value__))


class Answer(msgspec.Struct, frozen=True, kw_only=True):
    """Every answer's verdict; a refusal names its bounded code and detail."""

    ok: bool = False
    code: str = ""
    detail: str = ""


def refuse(code: str, detail: str) -> Answer:
    return Answer(ok=False, code=code, detail=detail[:1024])


class Handoff(Answer, frozen=True, kw_only=True):
    """An answer the worker follows with one connected socket, whose received fd this is."""

    descriptor: int = -1


class Room(Answer, frozen=True, kw_only=True):
    authorized_device_limit_bytes: int | UnsetType = UNSET


class CheckpointReceipt(Answer, frozen=True, kw_only=True):
    receipt_id: str = ""
    replayed: bool = False


class CallProgress(msgspec.Struct, frozen=True, kw_only=True):
    sequence: int
    payload: dict[str, object]


class ByteMetadata(TypedDict):
    """Verified native byte metadata retained for forwarding and tree members."""

    kind: str
    digest: str
    length: int
    content_bytes: int
    media_type: str
    local: str


class ByteGrant(ByteMetadata):
    """A native result projected into this executor's spool by its worker.

    The wire keys stay the same across SDK versions. The consumer validates the
    declared result against these fields before constructing an asset handle.
    """

    output_id: str


class CallState(Answer, frozen=True, kw_only=True):
    """A managed call's state; `child_call`, `child_poll` and `child_cancel` answer it."""

    state: str = ""
    child_request_id: str = ""
    #: The child's canonical result document, once `state` is "succeeded".
    result: str = ""
    # A peer may omit fields unused by its asset kind. Hydration reads the
    # consumed fields into a typed record before constructing any handle.
    byte_grants: tuple[Mapping[str, object], ...] = ()
    progress: CallProgress | None = None
    #: The child's execution observation document.
    observation: object = None


class Published(Answer, frozen=True, kw_only=True):
    """The product's sha256 and length; `sequence` 0 means nothing is shown (a child call)."""

    digest: str = ""
    length: int = 0
    sequence: int = 0


class Member(Answer, frozen=True, kw_only=True):
    tree: str = ""
    path: str = ""
    digest: str = ""
    length: int = 0
    media_type: str = ""
    local: str = ""
    file_state: FileState | None = None


class Opened(Handoff, frozen=True, kw_only=True):
    """A native source or output channel."""

    length: int = 0
    transaction: str = ""


class Adopted(Answer, frozen=True, kw_only=True):
    request_id: str = ""
    manifest: str = ""
    manifest_length: int = 0
    receipt_digest: str = ""


class Exchange(Protocol):
    """One durable request; blocks until the worker's answer decodes as `into`."""

    def __call__[A: Answer](self, request: Request, into: type[A], /) -> A: ...


@dataclass(frozen=True, slots=True)
class DescriptorReply:
    """A trusted handler's answer and its one connected capability; never a JSON fd number."""

    answer: Answer
    descriptor: socket.socket


type Reply = Answer | DescriptorReply
#: The worker's answer to each durable request of one executor call.
type Handler = Callable[[Request], Reply]


def encode(value: Request | Answer) -> dict[str, object]:
    """A request or answer as the JSON object the seam carries."""
    document = msgspec.json.decode(msgspec.json.encode(value))
    assert isinstance(document, dict)
    return document


def decode(frame: Mapping[str, object]) -> Request:
    """The request a frame states; a `CapabilityError` names why it cannot be answered."""
    kind = frame.get("kind")
    try:
        if kind == "weights_writer":
            writer: WriterRequest = msgspec.convert(frame, WriterRequest, strict=True)
            return writer
        if kind not in _KINDS:
            raise CapabilityError(
                f"{kind!r} is not a durable exchange this worker answers",
                code="unknown_durable_request",
            )
        request: _Tagged = msgspec.convert(frame, _Tagged, strict=True)
        return request
    except (msgspec.ValidationError, InvalidRequest) as exc:
        raise CapabilityError(
            f"{kind!r} request is malformed: {exc}", code="durable_request_malformed"
        ) from exc


def respond(
    frame: Mapping[str, object], handler: Handler | None
) -> tuple[dict[str, object], socket.socket | None]:
    """The answer frame to one request frame, and the socket that follows it."""
    reply: Reply
    if handler is None:
        reply = refuse("no_durable_exchange", f"no handler for {frame.get('kind')!r}")
    else:
        try:
            request = decode(frame)
        except CapabilityError as exc:
            reply = refuse(exc.code, exc.message)
        else:
            reply = handler(request)
    handoff = reply.descriptor if isinstance(reply, DescriptorReply) else None
    body = encode(reply.answer if isinstance(reply, DescriptorReply) else reply)
    if handoff is not None:
        body["descriptor"] = True
    return {"event": "answer", "seq": frame.get("seq"), **body}, handoff
