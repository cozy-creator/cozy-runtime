# Device capacity accounting

Let `G` be the process's total device grant, `H` its observed high-water non-allocator
usage, `R` PyTorch allocator reserve, and `F` current driver-free memory. Additional
allocator room is `min(G - H - R, F)`; the allocator's total reserve cap is
`min(G - H, F + R)` (clamped to a legal fraction only when installing the cap).
Driver-free memory already excludes consumed context/module/communicator bytes.
`H` is not an independently measured promise for future growth.

Allocator reserve remains charged even when allocated tensor bytes are lower. Cached
holes may be fragmented or retained by live storage; they are not additional capacity.
A room invoice asks for physical free bytes separately from total ownership: after
prospective owned reclamation `V`, a target `N` asks for `max(N - V, 0)` free and
`max(R - V + N + H, 0)` total grant. Every reply and reclamation is followed by fresh
capacity measurement; the invoice itself proves no memory was recovered.

The R14 run1858 receipt had `F=458489856`, `R=415236096`, `H=195035136`, and
`G=1068761088`. Subtracting `H` from `F` again reduced reported room to263454720,
refusing a311167232-byte scope demand. Counting consumed overhead once gives458489856
bytes of room while still charging all reserve, including56275456 bytes above allocated.
This arithmetic removes that false admission constraint; it does not prove that the
complete1GiB request, later activations, or every operation fits.

This remains sampled, soft enforcement: other processes and non-allocator allocations
can grow after a reading, and PyTorch's allocator cap does not constrain those external
allocations. Existing process sampling, signed deficits, driver-budget telemetry, OOM
recovery, retry contracts and honest refusals remain necessary. No scope pins, tensor
aliases, numerical modes, component placement or residency ranking change here.
