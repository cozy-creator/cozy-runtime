"""The executor's reply to one attempt (`invoke` or `run_job`), decoded once by the worker.

A refusal carries only `ok: false`, `code` and `detail`; a run carries the rest. Every field
has a default, so a reply from an executor pinned to an older Runtime decodes, and a field a
newer executor adds is ignored.
"""

from __future__ import annotations

import logging
from typing import Literal

import msgspec

from cozy_runtime.author._observations import Scalar
from cozy_runtime.internal import tolerant
from cozy_runtime.internal.canonical import Json

_LOG = logging.getLogger(__name__)


class Outcome(msgspec.Struct, frozen=True, kw_only=True):
    """The author kernel's verdict on the attempt."""

    terminal: Literal["succeeded", "failed", "refused", "canceled"]
    origin: str
    code: str = ""
    message: str = ""
    fields: tuple[str, ...] = ()
    traceback: str = ""


class Shortfall(msgspec.Struct, frozen=True, kw_only=True):
    """The capacity fact residency raised into the handler."""

    resource: str
    scope: str
    needed_bytes: int
    available_bytes: int
    evidence_class: str = "measured"
    request_shape: str = ""


class ObservationRow(msgspec.Struct, frozen=True, kw_only=True):
    kind: str = "log"
    name: str = ""
    value: Scalar = None
    fields: dict[str, Scalar] = {}
    at_unix_ms: int = 0
    seq: int = 0


class OutputRow(msgspec.Struct, frozen=True, kw_only=True):
    """One asset or tree the result names; size and digest once its bytes are encoded."""

    output_id: str
    asset_ref: str
    kind: str
    media_type: str = ""
    size_bytes: int | None = None
    digest: str = ""


class FrameRow(msgspec.Struct, frozen=True, kw_only=True):
    """One host frame the handler registered; the worker encodes it after release."""

    handle: str
    codec: str
    raw: str
    raw_bytes: int
    media_type: str = ""
    facts: dict[str, Json] = {}


class CaptureWritten(msgspec.Struct, frozen=True, kw_only=True):
    """The activation capture tree the executor wrote into the attempt spool."""

    output_id: str
    content_digest: str
    root: str
    length: int


class ResultRef(msgspec.Struct, frozen=True):
    digest: str
    length: int


class Adjustment(msgspec.Struct, frozen=True, kw_only=True):
    field: str
    requested: str
    applied: str
    reason: str = ""


class Streamed(msgspec.Struct, frozen=True, kw_only=True):
    """One component's streamed tail at its last stage: regions, resident prefix, window."""

    blocks: int = 0
    resident_blocks: int = 0
    window: int = 0


class PlaneFacts(msgspec.Struct, frozen=True, kw_only=True):
    """The executor's weight plane (`weight_plane/1`), as exact counters. Bytes are per device
    of this rank; `-1` is unreadable, never zero."""

    budget_bytes: int = -1
    committed_bytes: int = -1
    leased_bytes: int = -1
    pinned_budget_bytes: int = -1
    pinned_bytes: int = -1
    #: CUDA context and runtime workspaces outside torch and the plane, measured at load
    context_bytes: int = -1
    #: torch's peak above its bytes at the attempt's start: activations and derived weights
    activation_peak_bytes: int = -1
    resident: dict[str, int] = {}
    streamed: dict[str, Streamed] = {}
    h2d_bytes: int = 0
    h2d_gbps: float = 0.0
    #: of those, copied to the device straight from disk (the pinned tier did not hold them)
    #: and straight from the page cache (it did not either)
    disk_copy_bytes: int = 0
    mapped_copy_bytes: int = 0
    #: bytes this executor process read from storage since it started (`/proc/self/io`;
    #: TensorFS 0.3.92, always 0 before): a gate compares a delta across a return
    disk_read_bytes: int = 0
    late: int = 0
    stall_ns: int = 0
    misses: int = 0
    evictions: int = 0
    oom_retries: int = 0
    #: The decode mode and CFG layout each stage chose, e.g. {"decode": "untiled"}
    modes: dict[str, str] = {}


class Metrics(msgspec.Struct, frozen=True, kw_only=True):
    handler_ms: float = 0.0
    device_lease_ms: float = 0.0
    d2h_wait_ms: float = 0.0
    peak_vram_bytes: int = 0
    activation_peak_bytes: int = 0
    activation_peaks: dict[str, int] = {}
    allocated_at_start_bytes: int = -1
    working_peak_vram_bytes: int = 0
    shape_cell: str = ""
    rss_at_end_bytes: int = 0
    started_unix: float = 0.0
    gpu_count: int = 0


class AttemptReply(msgspec.Struct, frozen=True, kw_only=True):
    ok: bool
    code: str = ""
    detail: str = ""
    origin: str = ""
    poisoned: str = ""
    shortfall: Shortfall | None = None
    outcome: Outcome | None = None
    quiescent: bool = False
    #: Sealed environment names the package changed; the executor restored them.
    env_changed: tuple[str, ...] = ()
    residency: dict[str, Json] = {}
    #: Undecoded here: the worker reads each row alone, so an unreadable row costs itself,
    #: never the reply and its result.
    observations: tuple[object, ...] = ()
    observation_caps: dict[str, int] = {}
    attribution: dict[str, Json] = {}
    position: int = 0
    execution_observation: dict[str, Json] = {}
    execution: dict[str, Json] = {}
    #: A request pin some attention site of this call applied.
    attention_applied: bool = False
    capture: CaptureWritten | None = None
    outputs: tuple[OutputRow, ...] = ()
    frames: tuple[FrameRow, ...] = ()
    #: The working allowance the executor granted; absent, the admitted final outputs.
    max_output_bytes: int | None = None
    result_ref: ResultRef | None = None
    result_schema_digest: str = ""
    adjustments: tuple[Adjustment, ...] = ()
    ignored: tuple[str, ...] = ()
    """Request fields the package's types do not declare, dropped before decoding."""
    oversize_result_bytes: int = 0
    metrics: Metrics = Metrics()
    #: The card and ranks after the cache went back; an older executor is probed instead.
    after: dict[str, Json] | None = None
    plane: PlaneFacts | None = None


class PreparedReply(msgspec.Struct, frozen=True, kw_only=True):
    """`prepare_request`: the request resolved once, and only its digests."""

    ok: bool
    code: str = "request_unresolvable"
    detail: str = ""
    origin: str = ""
    features: dict[str, int] = {}
    features_digest: str = ""
    overlay_digest: str = ""
    facts_digest: str = ""
    adjustments: tuple[dict[str, str], ...] = ()


def budget_claim(reply: AttemptReply) -> dict[str, int]:
    """The allowance claim `output_budget.effective` checks against the admitted bound."""
    return {} if reply.max_output_bytes is None else {"max_output_bytes": reply.max_output_bytes}


#: Telemetry and narration: a member here that does not read takes its default. The verdict,
#: result, outputs, digests and `poisoned` stay exact.
_ADVISORY: dict[type, frozenset[str]] = {
    AttemptReply: frozenset(
        {
            "origin",
            "shortfall",
            "quiescent",
            "env_changed",
            "residency",
            "observations",
            "observation_caps",
            "attribution",
            "position",
            "execution_observation",
            "execution",
            "attention_applied",
            "adjustments",
            "ignored",
            "oversize_result_bytes",
            "metrics",
            "plane",
        }
    ),
    PreparedReply: frozenset({"origin", "detail", "adjustments"}),
}


def decode[R: (AttemptReply, PreparedReply)](frame: object, into: type[R]) -> R:
    """The reply a frame states. An unreadable advisory member costs itself, with one warning;
    a malformed load-bearing one is the executor's refusal, named."""
    try:
        reply, dropped = tolerant.read(frame, into, _ADVISORY[into])
    except msgspec.ValidationError as exc:
        return into(ok=False, code="executor_reply_malformed", detail=str(exc)[:1024])
    if dropped:
        _LOG.warning(
            "executor %s: dropped %d unreadable member(s); first %s",
            into.__name__,
            len(dropped),
            dropped[0][:512],
        )
    return reply
