# Native-source provenance

The initial RMSNorm plus split-half RoPE kernel is derived under Apache-2.0 from
`Comfy-Org/comfy-kitchen` tag `v0.2.31`, commit
`7c6ca3a5b63857d42c2d49777d6afb69de23f13f`, tree
`2f916b2d207a35e2247b2d2f5a5235dd3966af42`.

These four files are retained byte-for-byte with their NVIDIA SPDX headers:

- `native/rms_rope.cu` — SHA-256 `eb44a105487459fd2a8fe1f69f7252c8c3840d5132ecc3598e0b3e591dc7d715`
- `native/rope_device.cuh` — SHA-256 `94f76011a78bf4e8406bdb79b90baee6d234b7ab47bdee02127d1a8c24e2f02c`
- `native/dtype_dispatch.cuh` — SHA-256 `5a155c3bf2123604a0689ad792016b703c45d9676a389381b482ac1f1afb23bc`
- `native/utils.cuh` — SHA-256 `a3482d9911029f017c64b39dae5198f76a4c9585b8ede5c734643bcc18d04a4d`

`native/binding.cpp`, the Python wrapper, build system, fleet architecture policy, and
package boundary are Cozy modifications. The upstream `LICENSE` and `NOTICE` are shipped
inside every wheel. No HIP, Triton, quantization, attention, or unrelated native operation
from the upstream distribution is included.
