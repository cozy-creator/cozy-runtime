# Fresh scalar allocator observations

Planner needs four scalar facts: allocated and reserved bytes, current and peak.
Torch2.14's corresponding public scalar readers call `memory_stats`, which recursively
flattens and sorts the full native allocator dictionary before selecting one value.
The Planner now reads those same fresh native fields through the public
`memory_stats_as_nested_dict` capability, avoiding that formatting work.

The paths are `{allocated_bytes,reserved_bytes}.all.{current,peak}`. Missing paths
produce0, matching the public scalar readers. Native query/conversion failures are not
silently converted to0. Torch implementations without the nested reader keep using
the original scalar API. CPU execution never queries CUDA statistics.

There is no snapshot cache or sampling-cadence change. The reserved observations
before and after NVML remain distinct native queries; a mismatch still prevents a
fabricated overhead estimate. Peak resets, driver-free reads, grant formulas, reserved
cache charging and external-allocation limitations are unchanged.

Validation uses the installed Torch2.14 public scalar/nested implementations with the
native statistics seam replaced by CPU facts, including zero/negative/large values,
uninitialized/missing paths and fresh racing observations. The project optional Torch
extras require>=2.13; this change adds no version gate. Compatibility with a missing
nested capability is tested rather than inferred from an exact version string.

A guarded CPU benchmark over240synthetic returned fields made7000native-seam queries
per arm. Median scalar read was203.194µs through Torch's public flattened reader and
17.891µs through the Planner nested helper in that run. This includes a mocked native
call, excludes real driver/native dictionary-construction costs, and is not an
end-to-end inference speedup claim. GPU qualification remains separate.
