# Verified loading on quota-limited machines

The descriptor verifier previously parsed only cgroup-v2 `0::` membership. On cgroup-v1
consumer pods it took the unreadable fallback and used one worker despite multi-core quotas.
The shared `proctree.available_cpus` helper now measures affinity plus the tightest visible
v1/v2 ancestor quota. It reuses the existing v1 mount-root/membership resolver rather than
assuming a host-root directory. Malformed/unreadable entitlement stays conservative.

Descriptor verification keeps the same full native SHA/readonly/inode checks, four-worker
maximum, half-entitlement selection, fd bound and live-memory byte window. Compiler and
quantization widths use the same helper, reducing three divergent detectors and benefiting
either legacy or descriptor loaders. No environment variable or source-SHA peer gate selects
this behavior. Legacy loading remains the default until matched descriptor parity or better.

CPU fixtures use real temporary cgroup files and the shared mount parser for v1/v2, fractional
and parent quotas, bad input and low memory. A v1 fixture drives actual native descriptor
verification and rejects corrupted bytes. Source fd/failure cleanup, older/current-bound
real inference, native host read plans, weight policy and compiler gates pass. Exact fd leak
checks collect unrelated previous fixture finalizers before the baseline, retaining their
strict equality assertion. Hardware performance is not inferred from these tests.

Comparison source base is fetched Runtime b3416e34 plus safe9eb29a1c (merge8e1acba). Both
hardware arms must share that stage-end timing change, TFS build, cached absolute COZY_HOME,
allocator and cgroup settings. The observed gap motivating this fix is matched4090 Load
6.283s descriptor versus4.993s legacy; native construct/fill is2559.9ms versus889.6ms while
qualification is nearly equal. Root owns actual hardware qualification. No GPU/NVML/rental
was used by this task.
