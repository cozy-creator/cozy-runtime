Qwen prefix KV caching is selected before sampling and remains fixed through every
step and operation retry. A capacity refusal no longer restarts the entire
denoising loop without its cache. The model accepts an explicit cache choice and
reports the selected mode to its caller; Qwen package 0.2.4 records it in the
result for reproducibility. Cached and recomputed reduced-precision attention are
not claimed to produce identical images.
