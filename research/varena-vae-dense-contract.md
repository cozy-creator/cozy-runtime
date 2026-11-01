# H3 VAE dense-linear allocation contract

Status: source research in progress; no GPU execution, compilation, or qualification.

Scope: Torch `2.14.0+cu130`, commit `08187d9e0fba026dc8217405802ab5381dc88d90`; Runtime integration `61fe32331968f89a1585d25263924ef15cc04a39`; the H3 VAE fused block's six plain dense projections. Kitchen INT8 attention already has a caller-buffer contract and is outside this note.

The first implementation constraint is concrete: `linear.out` has a different 3-D route from ordinary contiguous biased `linear`. Ordinary linear flattens leading dimensions and uses `addmm`; `linear.out` only uses `addmm_out` directly for 2-D input and otherwise uses `matmul_out` followed by bias addition. Preserve the actual bias-fused route by explicitly creating 2-D views and calling `addmm_out` with a pre-sized contiguous output. This addresses the result allocation only; workspace and casts still need a contract. [Pinned Linear.cpp](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/Linear.cpp#L48-L149)

Autocast cannot be treated as a persistent prepared-weight cache: inference mode excludes casts from the cache, and eligible uncached tensors use `.to(dtype)`. The exact stored dtype and execution mode must therefore be part of the preparation contract. [Pinned autocast_mode.cpp](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/autocast_mode.cpp#L112-L145)

Further sections will record exported workspace API feasibility, backend selection, exact buffer sizes, and shared ownership before this note is complete.
