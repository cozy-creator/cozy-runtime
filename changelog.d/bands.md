Ops the device cannot run whole run in bands of rows, for any model: convolutions read their reach,
pointwise ops, casts, copies, joins, paddings, nearest upsampling and norms over channels their own rows,
group norms gather float32 statistics first. An op runs in bands when it ran out of memory or an operand
lives in host memory (read over the bus once per band). Its result stays on the device when idle weights
and the least recently used tensors (now the spill order) can make room, else goes to host memory. A
convolution right after a zero padding skips the taps that meet only zeros (a lone frame through a causal
3D convolution runs 2D). This replaces the image VAE decode mirrored per Diffusers class and pinned to
Diffusers 0.40.0 source digests: SDXL, Anima and qwen-image-2 decode with Diffusers' own code, so their
packages must not switch on Diffusers tiling. Spills and host results reuse page-locked host blocks until
the stage closes. qwen-image-2 4 MP on an 8 GiB pool: decode 70 s -> 29 s.
