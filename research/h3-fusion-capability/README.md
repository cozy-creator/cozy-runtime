MiniMax H3 fusion compatibility
==============================

Diffusers 0.41 adds a SwiGLU return annotation. The former source fingerprint rejected that harmless change and disabled every H3 glue substitution under `fusion="accept"`.

Fusion now checks the callable input contracts and validates all transformer topologies before mutating any module. Semantic hooks and instance forwards that fusion would bypass decline cleanly before any substitution. Parameter identity remains checked after installation. The constructed model exposes `_cozy_fusion_report`, and applied/fallback status is logged and retained in the existing executor receipt.

Validation: actual Diffusers 0.40 and the captured 0.41 wheel each pass 13 CPU tests, with 12 GPU tests skipped. These include exact BF16 agreement between the actual upstream block and the arithmetic composition used by fusion; changed dropout, activations, norms, hooks and input contracts are refused. Ruff and focused mypy pass. `admission/verify` exercises actual GPU kernels, FP8 rowwise leaves and permanent Turbo LoRA hooks in inactive, baked and unbaked states; its GPU execution is pending.

The private candidate overlays only the fusion installer onto the retained low-memory Runtime 0.19 benchmark wheel. Native worker, TensorFS cohort, attention implementations and precision defaults remain unchanged. No speed or video-quality claim is made before GPU qualification and full inference.
