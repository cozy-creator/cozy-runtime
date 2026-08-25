"""The ONE shared exit-code matrix (cozy-runtime-cli.md).

cozy-creator uses this matrix unchanged; no local-only codes exist anywhere. Every verb
and every refusal path imports `EXITS` / `Exit` from here — a second matrix is a fence
violation (`single-config` scans for a competing table).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Exit(IntEnum):
    """Process exit codes. The name is the `error(<name>)` token in structured output."""

    ok = 0
    internal = 1
    usage = 2
    validation = 3
    not_found = 4
    credential = 5
    structural = 6
    confirm = 7
    offline_miss = 8
    unavailable = 9
    deadline = 10
    failed = 11
    canceled = 12
    conflict = 13
    capacity = 14


@dataclass(frozen=True, slots=True)
class ExitMeaning:
    code: Exit
    meaning: str


EXITS: tuple[ExitMeaning, ...] = (
    ExitMeaning(Exit.ok, "success, including idempotent no-ops"),
    ExitMeaning(Exit.internal, "unexpected fault — a bug, never a user condition"),
    ExitMeaning(Exit.usage, "bad invocation: unknown flag, malformed key=value, majorless target"),
    ExitMeaning(Exit.validation, "typed payload/schema/bounds refusal; verifier refusal at ingest"),
    ExitMeaning(Exit.not_found, "unknown ref/function/endpoint/attempt; hub 404 rendered verbatim"),
    ExitMeaning(Exit.credential, "private/gated source without a credential"),
    ExitMeaning(Exit.structural, "structural incompatibility — never a fit shortfall"),
    ExitMeaning(Exit.confirm, "destructive op without --yes"),
    ExitMeaning(Exit.offline_miss, "--offline and the bytes are not in the CAS"),
    ExitMeaning(Exit.unavailable, "socket / local server / hub unreachable"),
    ExitMeaning(Exit.deadline, "--timeout or request deadline exceeded"),
    ExitMeaning(Exit.failed, "attempt/job failure terminal (remedy verbatim)"),
    ExitMeaning(Exit.canceled, "canceled terminal"),
    ExitMeaning(Exit.conflict, "target exists / concurrent writer / failed replacement kept state"),
    ExitMeaning(Exit.capacity, "no proven plan fits below the floor — quantified shortfall"),
)
