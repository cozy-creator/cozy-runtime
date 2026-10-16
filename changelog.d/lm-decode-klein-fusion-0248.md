Runtime 0.24.8 adds graphed LM decode and Klein's fused DiT blocks, routes Sol to B300, and bundles TensorD 0.5.38.

- LM decode (#1454): `cozy_runtime.lm` (graphed decode, `install()` for gated-delta kernels) and `cozy_runtime.graphs`; a faster nvfp4 one-row GEMV. Klein's upsampler on an 8 GiB pool: 4.6–6.9 s per caption (18.8–37.9 s before); 4090 nvfp4 decode 9.4 ms/token (67–69 before).
- Parallel fused DiT blocks (#1444): `fusion.DiTBlock` with one norm, `GatedMLP(up=None)`, `Rope(fp32=True)` (FLUX.3 Klein on a 4090 matches ComfyUI at 1 MP).
- Sol on B300 (#1459): SM103 runs SM100's kernel compiled for sm_103a.
- TensorD 0.5.38: an uninstallable `rental update --runtime-wheel` fails with uv's reason; runs on a development Runtime wheel name its build.
