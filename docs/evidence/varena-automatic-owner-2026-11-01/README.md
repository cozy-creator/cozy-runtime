# Automatic Runtime owner: exact GPU and decoder evidence

The successful cohort is Creator/TensorD/TensorFS **afa6f9229** with Runtime
**b206015ba**, not Runtime88fd alone. The ordinary frozen owner fixture is also
afa6f9229. Every saved raw file is unaltered and SHA256-listed in `manifest.json`.

| Run | Result and boundary |
| --- | --- |
| 25090 | Startup refused before the fixture. Native Start preserved cooperation, but Runtime's separate msgspec Start schema dropped it. Config became None; CUDA initialization occurred normally without the peer, and the worker's strict active acknowledgement correctly refused. This was not lazy CUDA or a missing native API. |
| 25106 | Actual automatically created Runtime Memory owner donated 48 MiB without any fixture call to donor publish/service. Actual driver used memory fell exactly 48 MiB; the independent requester then allocated/touched 32 MiB with an exact 32 MiB driver increase. |
| 25107 | Same executor PID191228 and native row(slot0,incarnation1) remained active; no replacement process is counted as owner-lifetime proof. |

The donor retained its 8 MiB live working value and 8 MiB pinned weights.
Duplicate ask returned the same completed round, adding no credit. Capture and
replay exclusion both made the automatic pump defer with zero native driver-call
delta. The settled parent returned to exactly17,039,360 requested live bytes /
20,971,520 mapped bytes (the existing two library allocations); pending bytes,
pools and event handles were zero. The requester retired all its managed state.
The observed driver-sample interval around ask/completion was502.980508ms,
including polling; this is not exact unmap latency or a general latency bound.

Startup was Dedicated(priority0). The fixture explicitly opted its own native
participant Shared(priority−10) and restored the actual Start categories. It did
not change TensorD's machine sharing configuration. This qualifies same-domain
automatic donor service, not shared-machine propagation, foreign/game pressure,
allocator-shortage asks or graph/cache owner retirement.

The collector verified the exact worker boot and SDK versions, one positive
`gpu_memory_policy=Cooperative` note for PID191228/device0, and a unique32-record
pre/post log anchor. No legacy Room/offload/ended-for-room/GPU-pressure event
appeared in the anchored policy-to-follow-up interval. Raw `memory --lines 0`
responses are saved; there is no global absence claim from a truncated tail or
from unlogged paths. The raw Start frame is not a `run show` field; actual active
owner state plus matching source/software and the strict worker check form that
acknowledgement boundary.

`codec/red.json` and `codec/green.json` replay the real native sender and actual
Runtime receiver without Torch. `codec/socket-green` runs the actual executor
socket path and confirms a forbidden import-only/cooperation Start is rejected
before Torch/CUDA. The earlier CPU wire proof covered TensorFS's native decoder
but missed Runtime's separate schema; the preserved25090 failure exposed it.

No product code changed when these receipts were committed. Source/build, this
finite automatic-owner execution, full P3 policy completion and release remain
separate gates.
