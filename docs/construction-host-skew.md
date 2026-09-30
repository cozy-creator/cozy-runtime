# Group construction across Worker host-capability versions

A current Worker advertising host_room receives the same aggregate construction HostRoom request and release as before. An older Worker omitting that additive capability receives no unsupported request: the leader's existing Planner instead observes available_host_bytes, subtracts its pending promises, and admits one coherent staging vector before any follower allocates. Only the leader selects this fallback; followers use the agreed vector and never issue local room RPCs.

This preserves the existing standalone/older-peer observed-capacity semantics. It is not a Worker-global reservation or a hard limit against other processes' unmaterialized promises. Current host/cgroup measurements and the existing allocator authority remain unchanged; no authentication, executor protocol revision or broad peer-version gate changes.

Planner owns geometry and pending-byte arithmetic. Executor only selects the advertised transport and carries the returned local charge through the existing all-rank drain/reap lifecycle. Success settles construction before warm; asymmetric abort settles only after follower reap. Failed local drain or unproved reap retains pending bytes until executor retirement. Local settlement does not imply physical allocator/cache pages were freed.

CPU tests exercise actual rank control processes on both capability arms, metadata-only/parked preparation, one-slot minimum capacity, refusal, leader abort, failed drain, unproved reap and prior pending promises. They do not qualify a published old Worker deployment or multi-GPU execution.
