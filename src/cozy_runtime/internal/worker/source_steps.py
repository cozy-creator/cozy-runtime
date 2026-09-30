"""The wire of a bounded native source child: one `Launch` in, progress and one answer out.

`source_worker` runs one accepted native source operation and `upload_child` one step of an
upload. Each side decodes the other's bytes once, into these types. Credentials and provider
URLs ride stdin only: never argv, environment, logs or persisted state.
"""

import faulthandler
import os
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Mapping
from typing import IO, Annotated, Literal, TypedDict

import msgspec

from cozy_runtime.author._artifacts import ObjectRef
from cozy_runtime.internal.source_interfaces import CommitFile

MAX_BYTES = 2 << 20
# Refusals a caller can act on; every other code is a native TensorFS code or opaque.
CODES = frozenset(
    {
        "model_ingestion_revision",
        "model_source_profiles_invalid",
        "model_source_components_overlap",
        "model_source_configs_overlap",
        "model_reference_unavailable",
    }
)

type Member = tuple[str, str, int, str]  # member, sha256:digest, length, url


class SourceRefusal(ValueError):
    """A reviewed refusal: its code crosses to the caller and its text names no secret."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


class Access(TypedDict):
    """How the child reaches a provider, as TensorFS's source calls take it."""

    credential: str
    huggingface: str | None
    civitai: str | None
    allow_local: bool


def access(endpoints: Mapping[str, str], credential: str) -> Access:
    """Endpoints stand in for the providers (tests) and admit local addresses."""
    return {
        "credential": credential,
        "huggingface": endpoints.get("huggingface"),
        "civitai": endpoints.get("civitai"),
        "allow_local": bool(endpoints),
    }


class Pin(msgspec.Struct, frozen=True):
    """The members a selection pins and the hosts that may deliver them."""

    members: list[Member]
    allowed_hosts: list[str]
    credential_hosts: list[str]


class Selection(Pin, frozen=True):
    canonical: str
    selection_sha256: str
    content_manifest_digest: str
    content_manifest_length: int


class Step(msgspec.Struct, frozen=True, kw_only=True, tag_field="step"):
    pass


class ResolveSource(Step, frozen=True, tag="resolve_source"):
    uri: str
    carriers: list[str]
    profiles: list[str]
    files: list[str]
    access: Access


class Download(Step, frozen=True, tag="download"):
    native_owner: str
    service_id: str
    pin: Pin
    access: Access


class Convert(Step, frozen=True, tag="convert"):
    native_owner: str
    service_id: str
    source_owner: str
    slots: list[tuple[str, str]]
    registry: bytes | None
    adopt: str | None
    writer_epoch: int
    computation_digest: str
    access: Access  # reaches the converters' pinned reference repositories


class View(Step, frozen=True, tag="view"):
    source_owner: str
    view_owner: str
    manifest: ObjectRef
    receipt_digest: str
    view_bytes: int


class Commit(Step, frozen=True, tag="commit"):
    request: CommitFile
    view_owner: str
    spool: str
    basename: str
    staging: str
    manifest: bytes


type SourceStep = ResolveSource | Download | Convert | View | Commit


class Resolve(Step, frozen=True, tag="resolve"):
    uri: str
    carriers: list[str] | None  # None: TensorFS selects a profile from the headers
    profiles: list[str]
    metadata: list[str]  # reviewed metadata pinned beside the carriers
    diffusers: bool  # a Diffusers index and its components' configs ride too
    registry: bytes | None
    access: Access


class Select(Step, frozen=True, tag="select"):
    uri: str
    members: list[Member]
    registry: bytes | None
    access: Access


class Heads(Step, frozen=True, tag="heads"):
    uri: str
    members: list[Member]
    path: str
    access: Access


class Plan(msgspec.Struct, frozen=True, kw_only=True):
    """The conversion every pass of one upload advances."""

    native_owner: str
    pin: Pin
    conversion: str
    source_selection_digest: str
    slots: list[tuple[str, str]]
    carriers: list[str]
    owners: dict[str, str]
    metadata: list[Member]
    metadata_owner: str
    heads_path: str
    registry: bytes | None
    recipe: str
    writer_epoch: int
    computation_digest: str
    access: Access

    def composition(self) -> "Composition":
        return Composition(
            native_owner=self.native_owner,
            writer_epoch=self.writer_epoch,
            computation_digest=self.computation_digest,
            metadata=self.metadata,
            recipe=self.recipe,
        )


class Composition(msgspec.Struct, frozen=True, kw_only=True):
    """What composes a converted model: its derived transaction and pinned metadata."""

    native_owner: str
    writer_epoch: int
    computation_digest: str
    metadata: list[Member]
    recipe: str


class Advance(Step, frozen=True, tag="advance"):
    plan: Plan
    window: list[str]
    budget: int | None
    adopt: dict[str, list[str]]


type UploadStep = Resolve | Select | Heads | Advance


# This module keeps eager annotations so msgspec can resolve ``step: S`` from the parameter.
class Launch[S](msgspec.Struct, frozen=True, kw_only=True):
    store: str
    parent_pid: Annotated[int, msgspec.Meta(gt=0)]
    step: S


class Answer(msgspec.Struct, frozen=True, kw_only=True, tag_field="answer"):
    pass


class Sample(Answer, frozen=True, tag="progress"):
    """One progress line ahead of the answer; the parent owns throttling."""

    stage: Literal["download", "convert"]
    position: int
    total: int


class Refused(Answer, frozen=True, tag="refused"):
    code: str


class Resolved(Answer, frozen=True, tag="resolved"):
    selection: Selection


class Produced(Answer, frozen=True, tag="produced"):
    result: dict[str, object]  # the operation's result document; the journal validates it
    native_receipt: bytes


class Selected(Answer, frozen=True, tag="selected"):
    profile: str


class Written(Answer, frozen=True, tag="written"):
    pass


class Prepared(msgspec.Struct, frozen=True):
    slot: str
    manifest_digest: str
    manifest_length: int


class Facts(msgspec.Struct, frozen=True):
    complete: bool
    landed: list[str]
    spent: list[str]
    converted_roles: int
    required_write_bytes: int
    sources: list[Prepared]


class Advanced(Answer, frozen=True, tag="advanced"):
    facts: Facts
    model: ObjectRef | None = None


class AdapterViewInput(msgspec.Struct, frozen=True):
    manifest: ObjectRef
    component: str
    source_component: str
    scale: str


class PrepareAdapterView(Step, frozen=True, tag="prepare_adapter_view"):
    identity: str
    base: ObjectRef
    adapters: tuple[AdapterViewInput, ...]


class AdapterViewReady(Answer, frozen=True, tag="adapter_view_ready"):
    manifest: ObjectRef


# ------------------------------------------------------------------ the child


_STDOUT = threading.Lock()
_SAMPLE = b'{"answer":"progress",'


def _line(body: bytes) -> None:
    with _STDOUT:
        sys.stdout.buffer.write(body + b"\n")
        sys.stdout.buffer.flush()


def report(stage: Literal["download", "convert"], position: int, total: int) -> None:
    _line(msgspec.json.encode(Sample(stage=stage, position=position, total=total)))


def serve[S](kind: type[Launch[S]], execute: Callable[[Launch[S]], Answer]) -> None:
    """Run the launch on stdin and answer it; the child dies with its parent."""
    # A native crash leaves every thread's stack in the stderr tail the parent keeps.
    faulthandler.enable()
    try:
        try:
            launch = msgspec.json.decode(sys.stdin.buffer.read(MAX_BYTES + 1), type=kind)
        except msgspec.DecodeError as exc:
            raise SourceRefusal("native_step_malformed", str(exc)) from exc
        parent = launch.parent_pid

        def parent_lifetime() -> None:
            tick = threading.Event()
            while os.getppid() == parent:
                tick.wait(0.25)
            os._exit(70)

        threading.Thread(target=parent_lifetime, daemon=True, name="source-parent").start()
        reply = execute(launch)
    except Exception as exc:
        # Only the fixed code is the answer. The parent redacts this stderr tail into the
        # failure's diagnosis, never into the result.
        traceback.print_exc()
        code = getattr(exc, "code", "native_source_failed")
        reply = Refused(code=code if isinstance(code, str) else "native_source_failed")
    body = msgspec.json.encode(reply)
    _line(
        body if len(body) <= MAX_BYTES else msgspec.json.encode(Refused(code="source_result_bound"))
    )


# ------------------------------------------------------------------ the parent


class _Tail:
    """The last stderr bytes of a child, drained continuously so it never blocks."""

    BYTES = 8 << 10

    def __init__(self, stream: IO[bytes]) -> None:
        self.data = bytearray()
        self.thread = threading.Thread(
            target=self._drain, args=(stream,), daemon=True, name="native-source-stderr"
        )
        self.thread.start()

    def _drain(self, stream: IO[bytes]) -> None:
        while chunk := stream.read(65536):
            self.data += chunk
            del self.data[: -self.BYTES]

    def text(self) -> str:
        self.thread.join()
        return self.data.decode(errors="replace")


def _exchange(
    process: subprocess.Popen[bytes], command: bytes, observe: Callable[[Sample], None]
) -> bytes:
    """Send the launch; stream progress lines; return the raw answer line."""
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.write(command)
        process.stdin.close()
    except BrokenPipeError:
        pass  # a killed child; its exit status says why
    answer = b""
    while line := process.stdout.readline(MAX_BYTES + 2):
        if line.startswith(_SAMPLE) and line.endswith(b"\n"):
            try:
                observe(msgspec.json.decode(line, type=Sample))
            except msgspec.DecodeError as exc:
                raise SourceRefusal("native_answer_malformed", str(exc)) from exc
            continue
        answer += line
        if len(answer) > MAX_BYTES + 1:
            process.kill()
            break
    return answer


def run[S: Step](
    module: str,
    launch: Launch[S],
    *,
    observe: Callable[[Sample], None] = lambda _sample: None,
    started: Callable[[subprocess.Popen[bytes]], None] = lambda _process: None,
) -> tuple[int, bytes, str]:
    """One child to its end: its exit status, its bounded answer bytes and its stderr tail.

    Its admission is the caller's to hold until this returns: the child writes no more.
    """
    command = msgspec.json.encode(launch)
    if len(command) > MAX_BYTES:
        raise SourceRefusal("native_step_bound", "a native step exceeds its bounded transfer")
    process = subprocess.Popen(
        [sys.executable, "-m", module],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    started(process)
    assert process.stderr is not None
    tail = _Tail(process.stderr)
    try:
        output = _exchange(process, command, observe)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
    return process.returncode, output if len(output) <= MAX_BYTES + 1 else b"", tail.text()


def answer[A: Answer](output: bytes, kind: type[A]) -> A | Refused:
    try:
        got: A | Refused = msgspec.json.decode(output, type=kind | Refused)
    except msgspec.DecodeError as exc:
        raise SourceRefusal("native_answer_malformed", str(exc)) from exc
    return got
