# Raw-QKV Kitchen producer collective correctness

Research-only CPU preparation for one bounded ordinary-CLI two-GPU diagnostic. No rental or GPU submission is authorized by this package. It reuses the existing tiny diagnostic checkpoint and public Runtime0.23.5/TensorFS0.6.1.

The intended comparison is identical raw pre-RMSNorm/pre-RoPE BF16 QKV, full56 heads versus actual Runtime Ulysses on two ranks with28 heads each. Both arms call existing Kitchen0.2.37 `sol_attn_chunked` with identical RMS weights, full rotary tables, prefix policy and4096-row chunks. Bootstrap/current statistics and one consecutive step with returned prior statistics are separate controls. No Kitchen fork, serving defaults, H3 videos or quality matrix.

Implementation and CPU/capture/peer checks are in progress; not GPU-ready.
