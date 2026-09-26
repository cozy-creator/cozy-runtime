# H3 Turbo direct activation

H3 Turbo is a supported direct activation path: the PDD-8 adapter's LoRA factors, pruned
AdaLN tables, and evaluation heads are loaded once with the prepared variant. Each denoise
step selects the prepared overlay through the request's attention selector. This preserves
the released BF16 factor rounding and leaves FP8-pruned base storage encoded; no
dequantize/merge/requantize operation is attempted.

`OfficialH3TurboLoRA.overlay()` returns the already prepared component for the requested
trunk. `H3TurboLoRA._sample()` installs that object only for the duration of the request
and restores the caller's `attention_kwargs` in `finally`. Separate requests therefore
cannot retain one another's overlay selection, while repeated steps in one request reuse
the same loaded adapter/tables/heads.

The focused CPU tests cover prepared-variant reuse and request-state restoration. GPU
quality and latency remain separate qualification work; this path intentionally keeps the
specialized per-forward activation semantics.
