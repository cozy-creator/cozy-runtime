# Foreign pressure: Runtime lifecycle boundary

Audit checkpoint only. This branch combines fresh Runtime640193706 with the
qualified4d5844f0 native-owner adapter; product behavior is unchanged.

Native Varena owns the byte policy. Runtime supplies the machine's categorical
sharing/priority configuration, actual activity/pass boundaries, owner retirement
work and fault reporting. It does not turn driver usage into a per-process budget.

Current seams:

- `Cooperation.Config` arrives through typed Start and binds at fresh CUDA Memory
  creation. CPU Hello must stay free of Torch/CUDA/NVML/owner-thread initialization.
- `Memory.cooperation_boundary` already marks use and checks native faults. Extend
  only its lifecycle facts; do not put a sampler or deficit computation back here.
- `Memory._open`'s outer-forward boundary and request begin/end currently call
  `_take_room`. Preserve these safe points for native regrowth notification while
  removing `_room_pending`, byte conversion, `_room_floor` and Room targets from the
  Cooperative path.
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
