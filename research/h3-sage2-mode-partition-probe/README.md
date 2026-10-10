# Sage2 mode and head-partition diagnostic

This separate research package extends the completed `h3-head-partition-probe`;
it does not modify that frozen package or any serving default. Submit `verify`
through ordinary `cozy run` only after the owner has finished the quality clips.
The package is intentionally not qualified to run until `cohort.json` names the
reviewed candidate version and its source/native hashes.

The four dense paths are the explicit H100 per-thread INT8 Q/K + FP8 P/V mode,
the per-warp variant of that mode, INT8 Q/K + FP16 P/V, and Kitchen INT8 control.
All Sage paths use the same `sageattention-full` artifact. The probe invokes the
actual Runtime adapters with their recorded implementation labels; it does not
substitute a kernel on refusal.

Every case uses identical synthetic BF16 Q/K/V, `[1,N,56,128]`, at N=129 and
N=109225. It compares a full call with fresh contiguous head groups for degrees
2, 4, 7 and 8. The result layout matches the contiguous BTHD tensors returned by
the pinned Ulysses exchange. Full-call and per-group repeats distinguish shape
effects from repeat instability. Comparisons use the same GPU scalar metrics as
the qualified original probe, with one final scalar download per metric.

For each Sage mode, the probe separately repeats the pinned upstream K-mean
expression and calls the upstream Q/K quantizer helpers with exactly the same
arguments. It compares means, INT8 codes and live FP32 scale slots across head
groups. It does not intercept a dense call or inspect V/P conversion or attention
accumulation. Unused scale slots past the live token count are excluded and their
counts are recorded. `upstream-call-contract.json` retains source-derived call
expressions and provenance; CPU controls verify those arguments and token-to-scale
ownership, not GPU arithmetic.

The FP16-PV entry also uses different Q/K block sizes from the SM90 FP8-PV entry.
These are upstream dense-mode comparisons, not a change to V precision alone.

Freeze the candidate without installing it:

```sh
python freeze_cohort.py --wheel=/path/to/reviewed.whl \
  --expected-sha256=REVIEWED_SHA256 --expected-version=EXACT_PRIVATE_VERSION
```

Use `{"backend":"all","size":"all","seed":581204}`. The eight fixed-size
cases are intended to finish within the existing eight-minute diagnostic
allocation, but CPU tests do not establish duration. The completed two-backend
H100 probe took 109 seconds including its cold effects. Require 24 GiB of
free/reusable memory for a full case; the probe refuses rather than shrinking
the workload. The new Q/K inspection adds work to that prior baseline. Retain
actual peak memory and elapsed time. Do not keep a rental
waiting for this package to be prepared.

This is one GPU with synthetic operands: no model, LoRA, collectives, VAE or
encoding. A mismatch localizes a source of head-geometry sensitivity, but does
not establish the first divergence in a full video or visible quality loss.
The diagnostic event timings include cold effects and exclude comparison/copy
work; they are not clean performance measurements.
