NVFP4 decoded-float providers use TensorFS's checked resident-buffer decoder on CPU and CUDA
architectures present in its installed kernel catalog. Existing final buffers retain the decoded
representation; packed native routes and unsupported-device fallback behavior are unchanged.
