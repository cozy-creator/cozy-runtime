# H3 request profiling

This research package delegates to MiniMax H3's registered render handlers and retains the original video. It records CUDA activity, host phase ranges and weight-plane counters. It does not change production Runtime code, attention defaults or inference arithmetic.

## Reproduction

The package targets Python 3.12, H3 1.26.3, Runtime `>=0.21.6,<0.22`, TensorFS `>=0.6.1,<0.7`, Torch 2.14 and Triton 3.8. Its model defaults describe the benchmark's **one RTX 5090** setup. They require these model releases on the CLI's configured Tensorhub:

- `paul/minimax-h3@1.0.0-rc.2/fp8-pruned`
- `paul/minimax-h3-turbo-lora@1.0.0-audit.1/pdd8`

The package template also uses the named `tensorhub` index for `minimax-h3==1.26.3` and `qwen-image-2>=0.3.0`. Cozy supplies that index from its configuration; the template is not a standalone PyPI-only `uv sync` project. An explicit `--tensorhub` selects another Hub for that command. Missing packages or models must be provided on the selected Hub; this tool supplies no fallback and never changes Hub configuration.

For an explicit local capture, use verified wheels from the intended cohort:

```sh
python research/h3-request-profile/prepare_capture.py \
  --out ./h3-profile-capture \
  --h3-wheel /path/to/minimax_h3-1.26.3-py3-none-any.whl \
  --qwen-wheel /path/to/qwen_image_2-0.3.0-py3-none-any.whl \
  --diffusers-wheel /path/to/diffusers-0.41.0.dev0-py3-none-any.whl
cozy run ./h3-profile-capture/turbo --describe
```

The helper checks wheel names and versions, records hashes, and replaces only those three package sources with the supplied paths. Diffusers must come from the pinned commit `578c9b2c6636ab2424a0e56186268b83623656b2`; the qualified wheel SHA-256 was `103ccb9e6f5a085dda13f2277da113ea29b2785c8aa051d195526b398ee70692`. Verify source custody as well as version metadata. Normal Cozy local-package capture carries those dependencies to the worker; absolute client paths are not remote paths. Preserve the generated lock and capture receipt. The worker's actual serving SDK remains authoritative.

A Turbo input keeps all eight steps and the full 15-second request:

```json
{"prompt":"Your motion scene","duration_s":15,"seed":581204,"profile_enabled":false}
```

Run full unprofiled warmups through this **same entrypoint**, then change only `profile_enabled` to `true` for the trace input. The regular entrypoint additionally requires `steps: 30`. Keep `profile_detail` and `trace_cpu_ops` false for the qualified CUDA-only path; CPU operator/projection hooks can alter eligibility and timing.

```sh
cozy run ./h3-profile-capture/turbo --input=warmup.json --rental=YOUR_RENTAL --idempotency-key=h3-warmup-1 --await
cozy run ./h3-profile-capture/turbo --input=trace.json --rental=YOUR_RENTAL --idempotency-key=h3-trace-1 --await --out=./trace-output
```

Use distinct keys for distinct requests; observe an uncertain accepted request rather than submitting it under a new key. A full trace declares a 64 GiB output bound, which is **not a RAM cap**. Profiler buffers and export still require host memory and disk space. Kernel construction before the handler and artifact export after it are outside the authored capture.

## Analysis and qualification

`trace_summary.py` streams ChromeTrace JSON or gzip and unions device intervals across streams. The report's `research/h3-single5090/attribute_trace.py` uses this parser to correlate GPU activities with their launching CPU ranges. `phase_receipt.py` joins ordinary run-show data with either a traced or explicitly unprofiled summary. Missing trace steps in a profiled record remain an error.

```sh
python research/h3-request-profile/trace_summary.py trace.json.gz --out=trace-summary.json
python research/h3-request-profile/phase_receipt.py --run-show=run-show.json --summary=summary.json --out=phase-receipt.json
```

The exact 0.0.12 recorder and entrypoint bytes completed ordinary Cozy requests 5089 and 5090 with profiling disabled, followed by full CUDA capture 5091 on public Runtime 0.21.7/TensorFS 0.6.1. All eight steps, the original audiovisual output, and a 260,457,336-byte compressed trace were retained. Correlation analysis accounted for 1,483,599 GPU activities; 1,292 without a captured host API remain explicitly unassociated. The recorder and entrypoint code in this port are unchanged from that capture.

These are diagnostic timings, not clean inference benchmarks. The diagnostic and direct H3 captures had 33 lock-version differences; actual environments included AV 19 versus AV 18. Video streams matched, while audio differed, so audio differences cannot be assigned solely to attention. Their warmup histories and allocator states also differed: model reuse alone does not establish an equivalent warm state. Match dependency locks, request history, GPU identity, served kernels and memory state before making performance claims. Do not add nested phase, sink and device-duration sums; encoding overlaps video decode.
