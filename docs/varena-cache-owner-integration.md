# Coherent Cache owner integration

This development-only branch combines the tested native retention/Plan/consumer baseline
with the weak native Cache owner bridge. It starts from Runtime0.36.23 and preserves
its host-tier admission change. Prior qualified artifacts and source branches remain
immutable. GPU qualification belongs to each exact native/Runtime/fixture cohort.

The Python retention window must remain absent. Native owner callbacks are not reclaimed
bytes: only physical release permits a bounded real allocator retry. Python owners claim
their own lifecycle nonblocking; an unavailable GIL or owner may leave demand exhausted.

The separate CUDA `Memory.reclaim` hard cut is an audit/preparation task here. No deletion
is enabled until real native-owner recovery and lifetime fixtures pass. MPS recovery and
explicit lifecycle disposal are separate contracts.
