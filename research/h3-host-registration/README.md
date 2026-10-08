H3 host registration diagnostic
================================

Run5006 records one host registration refusal and steady streamed weights moving through TensorFS pinned staging buffers. TensorFS0.4 maps sealed memfd read-only and correctly requests CUDA PORTABLE|READ_ONLY. Its pin_region returns only boolean status, so the original CUDA error is not retained.

The ordinary-CLI `package/verify` diagnostic queries device capabilities and reproduces readonly sealed memfd registration alongside writable controls at64KiB and8MiB. It records exact CUDA errors and memory snapshots and unregisters/unmaps everything it owns. It does not change model storage, allocator flags, benchmark defaults, or rental configuration. GPU execution is reserved for the root coordinator.

NVIDIA documents that READ_ONLY support is device attribute113 and unsupported devices return CUDA_ERROR_NOT_SUPPORTED: https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html . Attribute numbers99/100/113 are in https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/cuda_8h_source.html .
