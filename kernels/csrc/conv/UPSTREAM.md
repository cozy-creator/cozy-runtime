# Native-source provenance

`_kernels._conv` builds two comfy-kitchen kernels, Apache-2.0, from Comfy-Org/comfy-kitchen #167 (`62c5bb4`, the MiniMax
H3 VAE kernels), unmodified, out of the tree `CMakeLists.txt` pins as `KITCHEN_DIR`: tag `v0.2.40`, commit
`f8b7e97a1418d2e529bd5405fbf86312be198ae6`, tree `16a164828429cf8355d2cb0e241c17a54863a9a6` (byte-identical in `v0.2.41`,
which ComfyUI master requires):

- `comfy_kitchen/backends/cuda/ops/group_norm_pad3d.cu` — SHA-256 `8fd587782f76c180e4f1bd43415d70e5c3d4bfcc97a450d856ebf2319f4d583d`:
  per-frame GroupNorm statistics, then normalize, SiLU and the convolution's padding in one pass, NDHWC in and out.
- `comfy_kitchen/backends/cuda/ops/cutlass_conv3d_fp16.cu` — SHA-256 `73a73d901a15483c6d8948aad57e684958f03c0485ad6bfa1d7ee2c04782bf48`:
  CUTLASS's NDHWC implicit-GEMM conv3d accumulating in fp16 (fp32 past 16384 deep), bias and residual in the epilogue.
- their header `comfy_kitchen/backends/cuda/dtype_dispatch.cuh` — SHA-256 `5a155c3bf2123604a0689ad792016b703c45d9676a389381b482ac1f1afb23bc`.

`../conv.cpp` (the binding), the Python wrapper (`cozy_runtime.accel.fusion._conv`), the fusion layout that calls them, the
build flags (upstream's nvcc flags), the architecture targets and the package boundary are Cozy's. The upstream `LICENSE`
and `NOTICE` ride the cozy-runtime wheels.
