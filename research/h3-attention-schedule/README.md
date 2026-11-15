# H3 attention scheduling experiment

Tracker: [cr-298](https://github.com/cozy-creator/tracker/blob/master/tracker/cozy-runtime/cr-298-h3-attention-kernel-scheduling.md).
Draft: [Runtime #2185](https://github.com/cozy-creator/cozy-runtime/pull/2185).

Owner: Codex `/root/optimize_h3_attention`. Purpose: screen an unchanged-arithmetic
load-pipeline candidate for H3's expensive selected-key exact attention tiles.
Branch: `codex/h3-attention-schedule-20261115`.
Base: Runtime `af2a4753cb70598b75405942901e4f8f3b24eda6` (`origin/master`, fetched 2026-11-15).
Worktree: `/home/fidika/cozy/.worktrees/cozy-runtime/h3-attention-schedule-20261115`.

The cr-297 session `43442514074` SM89 trace attributes 9.96 seconds of CUDA activity
to `sol_exact_kernel` in its sampled sparse H3 call. Activity is summed, not an
interval union. This identifies a candidate; it does not establish a speedup.

The baseline is Runtime's pinned Comfy Kitchen source
`f8b7e97a1418d2e529bd5405fbf86312be198ae6` (0.2.40), specifically
`comfy_kitchen/backends/cuda/sage_attention/sol_attn_exact.cu`.
It uses two asynchronous stages, four warps, 64 query/key rows and head dimension
128. The first candidate changes only the pipeline depth to three. Key tiles,
selected lists, protected-row policy, precision, and reduction order stay fixed.

Ranked, falsifiable possibilities:

1. More outstanding tile loads hide selected-key latency: three stages improve
   paired kernel time with equal output bytes.
2. Shared memory/occupancy limits dominate: three stages regress because each CTA
   needs 48 KiB instead of 32 KiB.
3. Per-tile scale metadata remains exposed: stage depth changes little, and a
   later metadata-prefetch experiment would be needed. It is not implemented here.

The local RTX 4070 Laptop (SM89, 8 GiB) is authorized for bounded kernel screening.
The parent coordinator serializes GPU access. No rental, production code,
default-selection table, installed package, or running worker is changed.

## Result: keep the existing kernel

Both candidates compiled and preserved output bytes, but neither supports a
serving change. The three-stage candidate regressed all seven shapes. A second
candidate kept two stages and used the existing `cp_async16_ca` helper for the
two K/V loads in `SOLX_STAGE`, testing whether L1 reuse across query CTAs helped.
It regressed six shapes. Its one apparent win is noisy and is not qualified.

| Tokens | Heads | Selected 64-token blocks | Three stages / baseline | L1 K/V / baseline |
|---:|---:|---:|---:|---:|
| 4,096 | 8 | 8 | 1.036 | 1.042 |
| 4,096 | 8 | 32 | 1.055 | 1.070 |
| 4,096 | 8 | 64 | 1.041 | 1.062 |
| 16,384 | 8 | 32 | 1.063 | 1.067 |
| 16,384 | 8 | 128 | 1.104 | 1.082 |
| 65,536 | 2 | 64 | 1.223 | 1.077 |
| 65,536 | 2 | 256 | 1.234 | 0.946 |

Values are medians of **paired candidate/baseline ratios**, so larger is slower.
Each comparison uses 13 shuffled pairs and three CUDA graph replays per arm per
sample, alternating two fixed input sets. These are warm-cache microbenchmarks,
not cold-input measurements. Cache residency differs with shape; input bytes
are recorded. Each candidate was compared with its own contemporaneous baseline.
No numbers from separate runs are divided to claim a gain.

| Variant | Registers/thread | Shared bytes/CTA | Maximum active CTAs/SM | Spills |
|---|---:|---:|---:|---:|
| Baseline | 168 | 32,768 | 3 | 0 |
| Three stages | 221 | 49,152 | 2 | 0 |
| L1 K/V | 168 | 32,768 | 3 | 0 |

The stage-depth result is consistent with lost occupancy. These resource counts
do not prove the exact stall mechanism; no hardware counter profile was captured.
The L1 candidate's last row spans baseline 14.8–63.4 ms and candidate 15.4–24.2 ms,
so the apparent 5.4% paired benefit is not a stable result. Clocks were not locked;
the GPU read 80 C after testing. Long-case timing spreads are retained, including
outliers, and limit any precise performance conclusion.

Each candidate passed 20 correctness cases: BF16 and FP16 outputs, two seeds,
zero/few/odd/full-capacity selected-block lists, ragged query tails, token
augmentation, and mixed scales. Each case checks finite outputs, raw output bytes
against the baseline, a repeated baseline, and all input bytes for mutation.
The seven timed shapes also compare output bytes. This proves the tested kernel
transformation on synthetic packed inputs. It does not qualify producer packing,
routing decisions, protected rows, actual H3 tensors, or a complete attention call.

Raw samples, arm orders, exact shapes, binary hashes, device/driver and resources:

- [Three-stage result](results/sm89-stage3.json)
- [L1 K/V result](results/sm89-l1kv.json)
- [Compiler evidence](results/compiler.txt)

The source pin, dispatcher and all installed packages are unchanged. No new
runtime tuning mechanism or user-facing knob is proposed. The local GPU was
released to the coordinator after both screens, with no compute process left.

## Reproduce

`build.py` downloads Runtime's exact source archive, verifies its recorded SHA256,
and builds only the exact-tile translation unit with an exported C launcher.
It preserves upstream's arithmetic-related flags (`-O3`, `--use_fast_math`). The
added resource query does not modify the kernel. Sources, generated units and
shared libraries remain in ignored `vendor/` and `build/` directories.

```sh
python3 research/h3-attention-schedule/build.py --nvcc /path/to/cuda/bin/nvcc
/path/to/torch-env/bin/python research/h3-attention-schedule/bench.py \
  --variants baseline stage3 --output /path/to/stage3.json
/path/to/torch-env/bin/python research/h3-attention-schedule/bench.py \
  --variants baseline l1kv --output /path/to/l1kv.json
```

The harness refuses other compute processes before correctness and measurement
rounds. It uses fixed synthetic selected lists; input generation, routing,
quantization, and output allocation are outside the timed region. D128, tile
geometry, arithmetic and routes stay identical between each pair. Both variants
operate on the same read-only carrier tensors and separate output buffers.

Environment: NVIDIA RTX 4070 Laptop, SM89, 36 SMs, 32 MiB L2, 8 GiB; Torch
2.15.0+cu132; coherent CUDA compiler/NVVM/assembler 13.0.88. The initial shared
toolchain failed because NVVM 13.4 emitted PTX 9.4 while its 13.0 assembler only
accepted PTX 9.0. A pre-existing coherent toolchain was used read-only instead;
no shared toolchain was repaired or changed. The pip-layout toolkit needed its
`lib/` directory passed explicitly to the linker, as recorded by `build.py`.

## Remaining qualification

There is no shortlisted improvement to send to a paid H3 benchmark. A future
candidate first needs stable kernel evidence and real packed H3 inputs. Regular
H3 qualification remains 15 seconds, 362 frames, 1344x768, 30 steps and audio;
Turbo uses its separate fixed eight-step schedule. Full video, multi-GPU and
cross-hardware evidence remain separate. This experiment makes no H3 speedup claim.
