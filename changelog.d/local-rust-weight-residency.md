CUDA weights use the independently implemented process-local Rust manager shipped in
TensorFS. Runtime binds ordinary Torch tensor views whose storage retains its physical pin
and virtual-address owner, including escaped aliases. Idle bindings release their views;
native backing remains cached under a resident-prefix and bounded streaming policy.

Torch keeps its native activation cache. The supported native allocation hook reclaims
eligible weights and retries the requested allocation without re-running package code.
CUDA initialization requires the matching prebuilt TensorFS bridge and supported allocator;
graph capture/replay blocks native recovery device-wide. CPU and MPS retain Torch storage.

Root graphs retain their weight pins until retirement. Idle offload retires graphs before
releasing eligible backing, and changed fills receive new reservations while escaped aliases
keep the old reservation alive. No native code is compiled while serving.
