# Current-SDK LoRA byte-equivalence gate

This preserves the actual CUDA comparison from ordinary CLI run5012, now using
Runtime0.21.6+ and TensorFS0.6.1+ without selecting an old private Runtime wheel.
The immutable baseline helper SHA in `reference.json` also matches Runtime0.21.6's
unoptimized bake helper. It is a test oracle, never a serving fallback.

After the benchmark owner installs the reviewed integrated Runtime:

```sh
cozy run ./research/lora-bake-gate-021/verify --rental=kojirou \
  --tensorhub=http://127.0.0.1:8819 --json
```

The SM120 gate covers every E4M3 byte, five storage layouts, signed reserved codes,
plain BF16/FP32 and rowwise/blockscaled encoded bakes, statistics on/off, RNG state,
and unchanged LoRA factors. TF32/matmul settings restore even when a check raises.
It requires stored-byte equality and reports the installed candidate module hash.

Ordinary CLI `--describe --json` succeeds. Current-source CPU bake, decode,
lifetime, and weight-plane controls pass with TensorFS0.6.1. Actual CUDA execution
on the integrated0.21 wheel remains required; earlier0.19 evidence is not that proof.
