# Foreign pressure: Runtime lifecycle boundary

This branch combines fresh Runtime640193706 with the qualified4d5844f0 native-owner
adapter. The sampler checkpoint now forwards actual request begin/end to the native
owner; it adds no Python memory formula or sampler.

Native Varena owns the byte policy. Runtime supplies the machine's categorical
sharing/priority configuration, actual activity/pass boundaries, owner retirement
work and fault reporting. It does not turn driver usage into a per-process budget.

Current seams:

- `Cooperation.Config` arrives through typed Start and binds at fresh CUDA Memory
  creation. CPU Hello must stay free of Torch/CUDA/NVML/owner-thread initialization.
- `Memory.cooperation_boundary` marks use and checks native faults. Non-idle requests
  additionally send native begin/end notifications outside Memory locks. End is in
  the unconditional cleanup path and uses unchecked fault reporting, preserving the
  original handler error. Failed begin never emits a matching end.
- `Memory._open`'s outer-forward boundary and request begin/end currently call
  `_take_room`. Native `working::phase_enter` now owns the actual regrowth latch.
  Removing `_room_pending`, byte conversion, `_room_floor` and legacy Room targets
  remains part of the later controller retirement.
- The additive native `PRESSURE_PROTOCOL` is separate from the existing owner service
  capability. An older owner cohort keeps its negotiated allocation/recovery service
  without inventing pressure support; a new cohort exposes sampled pressure facts.
  This does not change an active Memory into Legacy mode or install a Python fallback.
- Native/graph owners retain their own safety conditions. A queued shrink request
  never means bytes were released. The Cache owner lane is implementing the narrow
  async queue and actual physical-release completion facts separately.
- An unknown attribution remains unknown. Runtime cannot teach H from a device-wide
  used/free delta or TensorD's context-size PID inference. Native telemetry and its
  qualification determine whether a local opaque-growth observation is attributable.
- A pressure failure remains no-retry at the execution boundary. Do not replay a
  partially executed package or cancel an active request merely for foreign memory.

Actual machine mode requires the supported lifecycle route. The current CLI lacks
the `machine update --mode` setter tracked by cl-694; direct fixture configuration
does not prove that route. Coordinate with that owner before extending it on our
own branch. Startup machine configuration is available now and can support the
first native policy checkpoint without an invented hot-switch protocol.

Root's eventual integrated Runtime contains the qualified Plan consumer as well as
this owner adapter. Integrate there deliberately after seams are reviewed; do not
rebase this owned audit branch underneath parallel Cache/Plan work.

Required execution proof remains real Shared/Dedicated startup/lifecycle,
coexistence above the measured live/pinned floor, clean below-floor exhaustion,
actual release latency and foreign allocation failures, safe pass-boundary
regrowth, and unknown attribution that does not train H. No such new GPU proof is
claimed by this document.

The current paired CPU source checks preserve the baseline findings (mypy 13→13,
Ruff 12→12, no new findings), and the changed lifecycle module passes Ruff. Native
control and real-ledger CPU proofs live in the Creator release gate. Actual Runtime
request-window sampling, CPU Hello on the final built wheel, shared-machine startup,
and the existing deliberate error-cleanup GPU cell still require the final paired
artifact; source checks do not qualify those behaviors.
