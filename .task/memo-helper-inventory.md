# First-party memo dependencies remain a qualification gap

Inspected source heads on 2026-09-26: Runtime `7ad96f9c1da0bc89c35291443de108ea7e8ce896`, packages `a702697e32689dba7680d63a404ce1349abac788`, Eval `ca5fd6060c1432abb16ff0bada115c179f519dd4`. This is source inspection, not deployed-cohort or model-quality qualification. Machine-readable direct-call inventory: `outputs/memo-optional-version-20260926/inventory.json`.

The optional-version fix restores automatic identity for a Python callee. It hashes that callee and explicitly declared dependencies only. **An unchanged wrapper can still reuse stale results after an undeclared helper or external resource changes.** This PR does not make the following operations safe across those changes.

## Undeclared managed-operation dependencies

All sixteen functions below have `@invocable(memoize=True)` and declare neither `memo_dependencies` nor `memo_version` at the inspected heads. A manual version is not required for ordinary Python source changes; the missing helper declarations are the important gap.

| Source | Functions | Result-affecting behavior outside the wrapper AST |
| --- | --- | --- |
| Eval `src/cozy_eval/jobs/activation_measurements.py:13` | `measure_activation_pair` | `cozy_eval.activations.measure_activation_pair`, `seal_activation_pair`; capture-fact serialization. |
| Eval `src/cozy_eval/jobs/audio_measurements.py:49,62` | `measure_audio`, `compare_audio` | Local `audio`, `metrics.audio.signal_stats` / `paired_stats`, audio decoding, `measurement_facts` sealers. |
| Eval `src/cozy_eval/jobs/measurements.py:57,80` | `measure_media`, `compare_media` | Local `frames`; `metrics.signal.score`, `metrics.reference.compare`, temporal sampling/constants, integrity algorithms, fact sealers; decoder/native numerical behavior. |
| Eval `src/cozy_eval/jobs/normalize_instruments.py:239,250` | `cast_whisper`, `prepare_instrument` | Local `cast_source` / `prepare_source`, `instrument_config.plan` / `configuration`, conversion/buffer rules and native derived-writer behavior. |
| Eval `src/cozy_eval/jobs/weight_measurements.py:56` | `measure_weights` | Local `NativeWeightsView`, `cozy_eval.weights.measure_weights` / `seal_weights` / chunk constants, native reader and numerical behavior. |
| Packages `sdxl/sdxl/normalization.py:278,290` | `normalize_component`, `assemble_normalized` | `_normalize`, `_component_plan`, `_assemble_normalized`, and the loaded `normalization.json` plan. A resource edit can change transformations/order without changing either wrapper. |
| Packages `minimax-h3-tools/src/h3_tables/operations.py:152` | `assemble_full` | Source-selection, order and model-config helpers; `assets/model-config.json` and `assets/whole-order.json`. |
| Packages `minimax-h3-tools/src/h3_tables/adaln_operations.py:223,332` | `select_adaln_weights`, `compute_adaln_tables` | Local plan/topology/projection/validation/compute helpers; `kernel` functions; plan/config/order resources and Torch/native numerical behavior. `plan_digest` is an input but does not cover all helper implementation changes. |
| Packages `minimax-h3-tools/src/h3_tables/adaln_operations.py:514,525` | `apply_adaln`, `retable_adaln` | Both delegate to `_apply_adaln`; its projection, validation, order/config helpers and resources do not enter their wrapper-only identities. |
| Packages `minimax-h3-tools/src/h3_tables/turbo.py:336` | `prepare_turbo` | `build_turbo_adapter` and its table/adapter/kernel helpers, plan resources and Torch/native numerical behavior. |

Use deliberate helper-function/module declarations for Python behavior. For JSON tables, native kernels and codecs, bind immutable data identities or an explicit semantic/numerical contract; Python AST hashing does not read data files or native implementations. Do not replace this review with a recursive import graph or package/caller inventory hash.

## Additional serving boundary

Eval registers `measure_quality` with `app.entrypoint(..., memoize=True)` at `src/cozy_eval/jobs/__init__.py:27`. Its body in `quality_instrument.py:31` uses `frames`, `score_quality`, `ResidentJudge`, checklist/report/sealing helpers and `FRAME_MODELS`. The bound model/media inputs do not identify changes in these Python implementations. The current `App.entrypoint` API exposes `memoize` but no public `memo_dependencies`/`memo_version` parameters, so this requires an explicit serving-declaration design before complete helper invalidation can be claimed.

## Existing Runtime declarations

Runtime `derive/operations.py:173` declares `quantize/1` plus `_selection` and the facade, quantization, microscale and safetensors-I/O modules. `prepare_model` at line208 declares `prepare-model/1` plus the Qwen ingestion module. These two already declare their direct Python helper surface; no wrapper change is justified in this scoped PR. Native TensorFS and NumPy/numerical-contract coverage still needs qualification, and an explicit version is not proof that every external behavior was audited.

The three privileged native source facades use a separate implementation/native-extension identity path, not `@invocable` metadata. They are outside this default-version change.

Current inventory is eighteen decorated managed functions, one memoized serving entrypoint, and three native source facades; the older count of twenty-one predates Runtime `prepare_model`. Counts establish no reuse, interruption, retention, GPU, or quality result.

Follow-up scopes: **Eval memo helper and serving-instrument invalidation** and **SDXL/H3 memo helper/resource invalidation**. Both need actual edited-helper/resource negative controls, plus the still-open ordinary local/private CLI and partial-recovery matrix. No library declarations or live artifacts were changed by this audit.
