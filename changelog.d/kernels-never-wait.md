A run never waits for a kernel compile: auto-selection and model preferences serve the best kernel ready (SDPA always is) while a preferred one compiles, and a warm construction selects again between requests once it is built. Explicit pins and model choices still wait.

The machine's compile ledger admits every build's compiler jobs (`PYTORCH_NVCC`) within the CPUs and the measured memory each job takes, replacing per-build shares.

FA3 builds only what the packages' sites run: head dimension 128 in BF16 (H3 and Qwen Image transformers) and 64 in FP16 (H3's video VAE), non-causal and fixed length; `flash-attn3-fp8` pins build `flash-attn3-full`. SageAttention2 builds only the family `sageattention` calls on the card; `sageattention-fp16pv` pins build `sageattention-full`. A worker image carries the kernel sources (`kernel_sources fetch`, only the paths recipes read), and a build fetches any it lacks side by side.
