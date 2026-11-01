# Categorical Runtime owner integration

This checkpoint adds startup and automatic donor transport around the explicit
native GPU peer; it does not complete Varena v2's memory policy.

TensorD obtains CPU-only executor Hello before cold `make_room` or CUDA Start.
One mutex binds the whole GPU lane to Legacy or Cooperative for the lifetime of
that Pool. Unknown devices allow capability discovery without running a byte
controller. A compatible Runtime can explicitly serve a Legacy device; an
incompatible executor refuses a Cooperative lane without falling back. A lane
cannot change policy under active or idle sessions. The first Cooperative
checkpoint supports one GPU; multi-GPU remains Legacy or refuses a mixed lane.

Start carries categorical sharing/priority, actual typed GPU/MIG UUID and a
shared tmpfs directory. Native identity/domain checks remain authoritative.
Cooperative lanes bypass every TensorD GPU byte fallback, including pressure
scheduling, room/open/lift, make_room, automatic Map(Ram), yield_to_call and the
inferred idle-weight credit in hardware free memory. Explicit warm intent,
host-RAM management, GPU permits and exact owned-child lifecycle remain.

The paired Runtime creates one participant for its Memory after CUDA startup,
wakes native publication at actual lifecycle boundaries, and periodically runs
native donor service with no byte formulas or alloc/free IPC. Typed contention,
capture and stale observations defer; sticky/device failures remain failures.
The owner stops its pump before guarded retirement outside Runtime/native/shared
locks. A failed attach cannot publish partial `host.memory`.

Qualification here comprises source/native checks, real CPU Hello/Shutdown,
actual mode-binding thread races, and the real stopped-writer/ledger matrix.
The ordinary GPU fixture `release-gate/suites/varena/owner-runtime` is prepared,
not executed at this checkpoint. It requires the actual automatically created
owner, uses an independent requester and never manually services the donor.
It temporarily configures its owner Shared when Start was Dedicated, records
and restores that category, and does not claim machine shared-mode propagation.

The preceding explicit native mechanism's run24919 stays bound to its original
1c6/e25/977 cohort. It is not automatic-owner proof for this change. Actual
allocator-shortage asks, cache/graph owner retirement, foreign pressure, game
coexistence, full container domains and the final deletion/default transition
remain required. Do not merge/release this draft as full P3 acceptance.

## GPU checkpoint after the actual receiver fix

Creatorafa6f9229 plus Runtimeb206015ba pass actual automatic-owner25106 and
same-PID/incarnation25107. Runtime88fd alone failed25090 because its separate
msgspec Start decoder dropped the native cooperation settings. The fix preserves
that typed field; initialization timing and the strict active acknowledgement
are unchanged. Full raw evidence and the scoped 48 MiB release / 32 MiB allocation
proof are in `evidence/varena-automatic-owner-2026-11-01/`. Complete P3
shortage/cache/foreign-pressure and shared-machine/game qualification remain open.
