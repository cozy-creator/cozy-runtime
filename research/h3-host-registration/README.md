H3 host registration diagnostic
================================

Run5006 records one host registration refusal and steady streamed weights moving through TensorFS pinned staging buffers. TensorFS0.4 maps sealed memfd read-only and correctly requests CUDA PORTABLE|READ_ONLY. Its pin_region returns only boolean status, so the original CUDA error is not retained.

The ordinary-CLI `package/verify` diagnostic queries device capabilities and reproduces readonly sealed memfd registration alongside writable controls at64KiB and8MiB. It records exact CUDA errors and memory snapshots and unregisters/unmaps everything it owns. It does not change model storage, allocator flags, benchmark defaults, or rental configuration. GPU execution is reserved for the root coordinator.

NVIDIA documents that READ_ONLY support is device attribute113 and unsupported devices return CUDA_ERROR_NOT_SUPPORTED: https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html . Attribute numbers99/100/113 are in https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/cuda_8h_source.html .

Residency source check
----------------------

The current H3 component-use scope covers the entire denoise loop. In the retained Runtime0.19 payload, `models/minimax_h3/model.py:188-235` admits `sample_fl2va` or `sample_fl2va_turbo` once; the Turbo adapter opens its own whole-call scope at `model.py:274-291`. `official.py:1658-1685` changes the attention context on each iteration, but does not release and readmit the Runtime weight scope. Consequently the general parked-stage fast path in `internal/weights.py:2908-2917` is not the immediate per-timestep explanation for this capture. There is no residency boundary at the dense-to-Sol transition in the current call path.

`weights.py:3168-3187` records a method's maximum allocated/reserved growth when the whole method finishes, and only raises the estimate. `driver_budget` at595-636 treats Torch cached slack as room available to future activations: growth is `max(activation_estimate - (reserved - allocated), 0)`. Releasing cached segments increases driver free bytes but decreases that slack by the same amount. If the historical activation estimate consumes the slack, the resulting weight budget does not increase merely because the cache was emptied. These are source facts, not a measured cause of the full denoise gap.

A Sol-eligible timestep still executes the refiner and first two main blocks densely, and its FFN dimensions are unchanged. No smaller safe activation requirement or fully resident32GB execution follows from the attention phase label. Any future Runtime experiment should first measure active working-set peaks separately from unused reserved cache, including those protected dense blocks and FFNs. If a smaller safe working set is demonstrated, an existing block/pass boundary can quiesce copies and replan with `_place(..., resume)` while preserving all open base/LoRA scopes, live tensors and the owner's cap. No package allocator calls or residency-policy edits were made for this investigation.

Source provenance: TensorFS `v0.4.0` at `c0947c347350e62b987b92c7106b0027434655d9`; retained Runtime `internal/weights.py` SHA256 `83bd379085433de9b979eb78767f092b3de1f52ae05f9ae00361200454489026`. Runtime wheel6df and the fusion1e2 overlay have identical weight-manager bytes.

Local diagnostic validation: the actual Linux64KiB and8MiB mapping cases preserve their bytes, use expected `rw-s`/`r--s` permissions and sealed flags23, and release every owned mapping/FD. Focused Ruff/mypy and ordinary CLI describe pass. CUDA execution remains pending; readonly registration support and its exact failure code are not inferred from the counter.

Installed distribution correction: run5010 stopped before execution. The0.0.1 fixture omitted its `cozy.application` distribution entrypoint even though local `package.toml` allowed source describe. A CPU negative control built and installed that original wheel in an isolated venv and Runtime0.19 installed-distribution describe refused with `static_distribution: installed wheel metadata is absent`. Fixture0.0.2 adds the entrypoint; the same installed-wheel describe now returns the expected accelerator `verify` contract. This fixes diagnostic packaging only; no Runtime or TensorFS implementation changed. The original source, wheel and refusal evidence remain retained.
