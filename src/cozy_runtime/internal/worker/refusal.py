"""Refusal-text discipline: who is at fault, whose address missed, and whose remedy it is.

Three rules, each recovered from a v1 incident where the text was the whole cost:

**READER versus DATA.** "cannot read X" is two different bugs. A DATA fault means the stored
bytes are wrong and the artifact must be fixed or refetched; a READER fault means the bytes
are admissible and THIS runtime cannot handle them, which is a runtime gap and never the
user's problem. The refusal says which, in the sentence, because the two have opposite
remedies and an operator who guesses wrong wastes the outage on the wrong system.

**Fault ORIGIN decides severity (v1 REF_ORIGIN).** A CALLER-supplied address that misses is
the caller's typed error — the request named something that is not there. A
PLATFORM-produced address that misses is platform-fatal: the deployment handed the worker a
location it minted itself, so the miss is a defect in the deployment and reporting it as a
bad request would send the wrong person to look. The same `FileNotFoundError` is therefore
two different terminals, and only the origin of the ADDRESS separates them.

**Upstream remedy sentences travel VERBATIM.** When TensorFS or the fill plane says how to
fix it, that sentence is quoted, not paraphrased. A paraphrase drifts from the system that
owns the fix, and the reader ends up acting on this runtime's guess about someone else's
component.
"""

from __future__ import annotations

from dataclasses import dataclass

from cozy_runtime.protocol import worker_pb2 as pb

CAUSE = pb.CauseCode
ORIGIN = pb.CauseOrigin

#: Where an address came from. This is the ONLY input that decides a miss's severity.
CALLER = "caller"
PLATFORM = "platform"

#: Which side of a read is wrong. `data` = the bytes; `reader` = this runtime. `lease` is
#: the third answer the DATA/READER split cannot give: the bytes were admissible and the
#: reader had a path, and the HOLD over them went away underneath the read. Claiming either
#: of the other two for it sends the operator to rebuild an artifact that is fine or to
#: patch a runtime that is fine, which is the exact cost this module exists to avoid.
DATA = "data"
READER = "reader"
LEASE = "lease"

#: The fill/store refusal vocabulary split by side. A code absent here has no classification
#: and must be added deliberately — an unclassified refusal cannot claim either sentence.
SIDE: dict[str, str] = {
    "missing_object": DATA,
    "digest_mismatch": DATA,
    "length_mismatch": DATA,
    "object_corrupt": DATA,
    "object_absent": DATA,
    "input_digest_mismatch": DATA,
    "input_length_mismatch": DATA,
    "dtype_mismatch": READER,
    "derived_unmaterialized": READER,
    "encoder_unavailable": READER,
    "result_too_large": READER,
    "unsupported_encoding": READER,
    # The fill plane's store-refusal translations (`internal/fill.py::STORE_REFUSALS`).
    # `checkpoint_unreadable` is DATA for three of the four store codes behind it
    # (MALFORMED_JSON, NONCANONICAL_ENCODING, MISSING_FIELD are all "the stored header is
    # not what it claims"); UNKNOWN_FORMAT is the one reader-shaped member, and the fill
    # plane collapses it into the same code, so the sentence names the majority remedy.
    "checkpoint_unreadable": DATA,
    "lease_revoked": LEASE,
    # The stored carrier is not the encoding it cites — the header's role set or a role's
    # geometry disagrees with the spec object the tensor names (TensorFS `ROLE_SET_MISMATCH`
    # / `SHAPE_MISMATCH`, raised where the worker re-checks the header against its own
    # encoding closure). DATA: the bytes are wrong and the artifact must be rebuilt by its
    # producer. Every other side would send someone to fix a system that is fine.
    "carrier_geometry": DATA,
    # cr-006's three, which had no classification and therefore no fault sentence at all.
    # All READER, and each for the same reason: the header was VALIDATED against its own
    # cited specs before any of them can fire, so what is left in every case is that THIS
    # BUILD has no path for admissible bytes — no provider for a registered spec, no
    # capability record on this card, or a provider registered against a spec whose role set
    # it does not implement. None of the three is fixed by touching the artifact.
    "unknown_encoding": READER,
    "encoding_unqualified": READER,
    "role_mismatch": READER,
}

_SENTENCE = {
    DATA: (
        "the DATA is at fault: the stored bytes are not what they claim to be, so the "
        "artifact must be refetched or rebuilt - re-running this request against the same "
        "bytes reproduces it exactly"
    ),
    READER: (
        "the READER is at fault: the bytes are admissible and THIS runtime has no path for "
        "them, so the fix is in the runtime, not in the artifact and not in the request"
    ),
    LEASE: (
        "neither the data nor the reader is at fault: the store hold covering these bytes "
        "was released while the read was in flight, so the remedy is to re-acquire and "
        "retry - the artifact is intact and the runtime needs no change"
    ),
}


@dataclass(frozen=True, slots=True)
class Refusal:
    """One classified refusal: its code, its side, its origin, and the text to print."""

    code: str
    side: str
    origin: str
    detail: str
    remedy: str = ""

    def sentence(self) -> str:
        """The full refusal text. Ordered cause-first, then whose fault, then the remedy."""
        parts = [f"{self.code}: {self.detail}"]
        if self.side in _SENTENCE:
            parts.append(_SENTENCE[self.side])
        if self.remedy:
            parts.append(f"upstream remedy, verbatim: {self.remedy}")
        return " - ".join(parts)

    def cause(self) -> tuple[pb.CauseCode, pb.CauseOrigin]:
        """The typed terminal this refusal becomes. ORIGIN of the address decides it."""
        if self.origin == CALLER:
            return CAUSE.CAUSE_CODE_INVALID_REQUEST, ORIGIN.CAUSE_ORIGIN_CLIENT
        if self.side == READER:
            return CAUSE.CAUSE_CODE_CAPABILITY_UNAVAILABLE, ORIGIN.CAUSE_ORIGIN_RUNTIME
        return CAUSE.CAUSE_CODE_LOCAL_SAFETY, ORIGIN.CAUSE_ORIGIN_WORKER

    def severity(self) -> str:
        """`caller_error` or `platform_fatal` - the v1 REF_ORIGIN split, kept explicit."""
        return "caller_error" if self.origin == CALLER else "platform_fatal"


def classify(code: str, detail: str, *, origin: str, remedy: str = "") -> Refusal:
    """Classify one refusal. An unknown code keeps an empty side rather than guessing."""
    return Refusal(code, SIDE.get(code, ""), origin, detail, remedy)
