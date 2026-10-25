MPS weights load through bounded TensorFS CPU reads followed by Torch-owned blocking copies,
because TensorFS's native destination ABI accepts CPU and CUDA, not Metal. Preparation accounts
for the staging window; residency uses the existing unified-memory budget and releases idle
MPS allocator cache when reclaiming. Varena remains CUDA-only.
