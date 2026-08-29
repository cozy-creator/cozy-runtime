"""The ONE shared exit-code matrix (cozy-runtime-cli.md).

cozy-creator uses this matrix unchanged; no local-only codes exist anywhere. Every verb
and every refusal path imports `Exit` from here — a second matrix is a fence violation
(`single-config` scans for a competing table). Each code's meaning rides its member: a
parallel table of the same meanings is exactly the second matrix this file forbids.
"""

from __future__ import annotations

from enum import IntEnum


class Exit(IntEnum):
    """Process exit codes. The name is the `error(<name>)` token in structured output."""

    ok = 0  # success, including idempotent no-ops
    internal = 1  # unexpected fault — a bug, never a user condition
    usage = 2  # bad invocation: unknown flag, malformed key=value, majorless target
    validation = 3  # typed payload/schema/bounds refusal; verifier refusal at ingest
    not_found = 4  # unknown ref/function/package/attempt; hub 404 rendered verbatim
    credential = 5  # private/gated source without a credential
    structural = 6  # structural incompatibility — never a fit shortfall
    confirm = 7  # destructive op without --yes
    offline_miss = 8  # --offline and the bytes are not in the CAS
    unavailable = 9  # socket / local server / hub unreachable
    deadline = 10  # --timeout or request deadline exceeded
    failed = 11  # attempt/job failure terminal (remedy verbatim)
    canceled = 12  # canceled terminal
    conflict = 13  # target exists / concurrent writer / failed replacement kept state
    capacity = 14  # no proven plan fits below the floor — quantified shortfall
