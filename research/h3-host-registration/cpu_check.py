"""Check real Linux mapping/seal/custody behavior without loading CUDA."""

from __future__ import annotations

import json
import os
from pathlib import Path

from h3_host_registration import mapping

before = set(os.listdir("/proc/self/fd"))
cases = []
for size in (64 << 10, 8 << 20):
    for sealed in (False, True):
        with mapping(size, sealed) as (pointer, facts):
            assert pointer % 4096 == 0
            assert facts["mapping_permissions"] == ("r--s" if sealed else "rw-s")
            assert facts["fd_seals"] == (23 if sealed else 0)
            cases.append(facts)
assert set(os.listdir("/proc/self/fd")) == before
assert not any(
    "h3-host-registration-diagnostic" in row
    for row in Path("/proc/self/maps").read_text().splitlines()
)
print(
    json.dumps(
        {"local_cpu_only": True, "cases": cases, "fd_cleanup": True, "mapping_cleanup": True},
        indent=2,
    )
)
