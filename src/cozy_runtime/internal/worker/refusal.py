"""Refusal-text discipline: whose address missed, and whose remedy it is.

Two rules, each recovered from a v1 incident where the text was the whole cost:

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


@dataclass(frozen=True, slots=True)
class Refusal:
    """One classified refusal: its code, its origin, and the text to print."""

    code: str
    origin: str
    detail: str
    remedy: str = ""

    def sentence(self) -> str:
        """The full refusal text. Ordered cause-first, then the remedy."""
        parts = [f"{self.code}: {self.detail}"]
        if self.remedy:
            parts.append(f"upstream remedy, verbatim: {self.remedy}")
        return " - ".join(parts)

    def cause(self) -> tuple[pb.CauseCode, pb.CauseOrigin]:
        """The typed terminal this refusal becomes. ORIGIN of the address decides it."""
        if self.origin == CALLER:
            return CAUSE.CAUSE_CODE_INVALID_REQUEST, ORIGIN.CAUSE_ORIGIN_CLIENT
        return CAUSE.CAUSE_CODE_LOCAL_SAFETY, ORIGIN.CAUSE_ORIGIN_WORKER

    def severity(self) -> str:
        """`caller_error` or `platform_fatal` - the v1 REF_ORIGIN split, kept explicit."""
        return "caller_error" if self.origin == CALLER else "platform_fatal"


def classify(code: str, detail: str, *, origin: str, remedy: str = "") -> Refusal:
    """Classify one refusal by the ORIGIN of the address that missed."""
    return Refusal(code, origin, detail, remedy)
