MiniMax H3 fusion compatibility
==============================

Diffusers 0.41 adds a SwiGLU return annotation. The former source fingerprint rejected that harmless change and disabled every H3 glue substitution under `fusion="accept"`.

Fusion now checks the callable input contracts and validates all transformer topologies before mutating any module. Semantic hooks and instance forwards that fusion would bypass decline cleanly before any substitution. Parameter identity remains checked after installation. The constructed model exposes `_cozy_fusion_report`, and applied/fallback status is logged and retained in the existing executor receipt.

Validation: actual Diffusers 0.40 and the captured 0.41 wheel each pass 18 CPU tests, with 12 GPU tests skipped. These include exact BF16 agreement between the actual upstream block and the arithmetic composition used by fusion; changed dropout, activations, norms, hooks and input contracts are refused. Ruff and focused mypy pass. `admission/verify` exercises actual GPU kernels, FP8 rowwise leaves and permanent Turbo LoRA hooks in inactive, baked and unbaked states; its GPU execution is pending.

The private candidate overlays only the fusion installer onto the retained low-memory Runtime 0.19 benchmark wheel. Native worker, TensorFS cohort, attention implementations and precision defaults remain unchanged. No speed or video-quality claim is made before GPU qualification and full inference.

Admission fixture corrections: run5007 passed a device object to the integer-SM compile helper, and run5009 omitted the required rowwise provider encoding. Fixture0.0.4 uses the proven explicit `fp8-rowwise/1` leaf signature and shares a constructor helper between its GPU gate and CPU validation. All five cases were constructed at the actual5376/14336 widths on CPU against the exact Runtime0.19 fusion1e2 wheel and captured Diffusers0.41; real encoded leaves, permanent LoRA hooks, modulation tables, shapes and helper signatures passed. A six-case CPU contract regression covers architecture parsing and every constructor path. GPU arithmetic remains a separate gate; the Runtime wheel is unchanged.
