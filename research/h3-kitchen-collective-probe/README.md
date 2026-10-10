# Raw-QKV Kitchen producer collective correctness

Research-only ordinary-CLI diagnostic using the existing tiny checkpoint and public Runtime0.23.5/TensorFS0.6.1. No serving/default or third-party kernel changes; no H3 videos or quality matrix. Root owns any later rental and GPU invocation.

The fixture uses identical raw BF16 QKV[1,109225,56,128] on both ranks. It takes unequal local token slices[54613,54612], uses actual Runtime Ulysses exchange, and calls existing Kitchen0.2.37 sol_attn_chunked over global4096-row chunks on each rank's28heads. Both arms use the actual H3 global96-channel rotary presentation and shared RMS weights[128]. Prefix257 rounds to5 protected64-token query/key blocks, identically on full and partitioned paths; this is a synthetic prefix, not a claim about a particular H3 prompt.

Step0 bootstraps statistics. Step1 uses a second fixed input(seed+1) and each trajectory's OWN returned prior kmean/vscale[H,128]FP32. Full56 slices are comparison oracles only; they never replace the distributed trajectory's state. Repeats reuse the same input prior pair before advancing to first-returned statistics. Reports include exact QKV/inverse transport, full/repeated outputs, protected prefix, input custody, prior/returned statistics, all rank/PID/UUID identities and actual kernel/native source provenance. Failed transport returns diagnostics and skips attention.

Three warmed step1 CUDA/wall samples compare each rank's full reference with the two-rank exchange-plus-producer call. They are a kernel regression observation, not pure link time or a generation-speed benchmark. The full shape requires28GiB free/reusable memory; no automatic downsizing. Source memory accounting estimates full56 Kitchen workspace2.723GiB and half28workspace1.362GiB, with known peak liveness about13.11GiB plus rotary and allocator overhead; retain measured peak.

CPU checks cover real two-process Gloo/CpComms unequal and equal splits, packing/global row starts, actual H3 rotary presentation, checkpoint census and the real Diffusers CP tensor-output hook. They do not run CUDA fused RMS/RoPE/quantization or prove whole-pipeline determinism.

The component returns bounded CPUuint8 JSON so the actual Diffusers CP output hook accepts it; Model.verify decodes outside the mirrored component. Use the normal CLI captured carrier supplied in the handoff, not a manual environment edit. Example after explicit owner lease:

```
cozy run /absolute/package-final/verify --rental=OWNED --input=input.json --timeout=8m --idempotency-key=UNIQUE --out=outputs --await --json
```

Production integration would still need full pre-split RoPE retention and CP-aware raw projection. This fixture supplies global RoPE explicitly and does not implement those serving changes.
