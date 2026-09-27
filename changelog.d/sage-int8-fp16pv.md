New explicit attention kernel `sageattention-fp16pv`: SageAttention2 INT8 Q/K (per thread, smoothed
K) with FP16 P/V and FP32 accumulation (`sageattn_qk_int8_pv_fp16_cuda`), SM90 only, through the
same Ulysses wrapper and per-rank probe as `sageattention` (INT8 Q/K + FP8 P/V), with a tighter
3e-2 rel L2 budget. Pin it with `--attention-kernel=<component>=sageattention-fp16pv`.
