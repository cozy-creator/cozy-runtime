# Over-cap stage admission

The first unchanged SDXL request on a fresh RTX3070 with a fixed 3 GiB physical
pool reached denoise step 1/20, then failed. Runtime0012896a sent a device-room
integer of -41943040; Rust72c5c88 refused that unsigned field and invalidated its
executor. Run4267/req-35771dd925da58892d4690a2, exact installed source hashes,
terminal output and fixed holder birth remain in
`outputs/codex-machine-audit-20261004/memory/fresh-pools-20261004`.

`Weights.room_before` priced measured activations against `cap - held`. Retained
torch reservations plus the context and mapped weights may already exceed the
cap, making that credit negative. CPU execution of the exact installed function
reproduced the observed -40 MiB request from authored over-cap observations.
Raw wire bytes were not retained; source and GPU failure agree on the field and
value, while the modeled accounting inputs are separately labeled.

Before a new stage, the Runtime now releases cached torch allocations and
remeasures its actual holdings. If they still exceed the existing cap, it
unparks prior idle stages and uses the existing plane shrink at the current
active-stage floor. Native leases remain the eviction boundary; the cap never
increases. If live holdings still exceed the cap, it reports an indivisible
capacity refusal locally. Only a positive grantable credit can ask other tenants
for room. A separate DeviceRoom boundary rejects any negative request before it
enters the durable channel. That guard alone is not the pressure fix.

CPU accounting tests cover cache release, idle-map trimming with a protected
floor, unreleasable holdings and the nonnegative durable boundary. They use
authored allocator observations and do not qualify native GPU eviction. The
required next gate is the same full1024-square20-step request, as a new attempt
on a reviewed immutable SDK, with the original failure and unchanged physical
holder preserved. No whole forward/decode replay, lower request or relaxed pool
is part of recovery. Grouped/alternating cells and all lower pools remain open.
