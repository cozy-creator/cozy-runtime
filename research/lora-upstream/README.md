# LoRA upstream reuse investigation

Owner: `/root/lora_upstream_reuse`. Base Runtime commit: `b30c56c2`.

This investigation uses current, unmodified upstream code to decide which parts
of Runtime's experimental LoRA path should be replaced. It makes no GPU
performance or video-quality claim from CPU results.

| Project | Examined commit |
|---|---|
| PEFT | `e99fdd275075068d093f8996a59bc0eb865f71a2` |
| Diffusers | `1f7be81f9bc84e7f2ab1ae91d37c7236c4c15691` |
| ComfyUI | `b00584967778871f329c2d31190a5e9b767b8366` |

The executable proof uses the real published spatial and Wushu LoRA factors at
their full 7168-by-5376 projection geometry. Its base weights and activations are
generated fixtures; retained factors are hash-checked. It exercises upstream
PEFT loading, dtype selection, residual execution, switching, and fusion. It
also attempts supported custom-module injection against Runtime's actual encoded
leaf. No fake `.weight` property or replacement FP8 computation is introduced.

The final comparison, reproducible commands and results will be recorded here
after running the proof. This branch deliberately does not install an unmeasured
backend into serving.
