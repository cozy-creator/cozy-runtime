# Approved H3 attention defaults

Owner: `/root/linear_resume`. Branch: `feat/h3-approved-attention-defaults-20261009`.
Base: `4f5dd8e3` (`origin/master`, fetched 2026-10-09).

The user approved SageAttention3 global-mean NVFP4 dense attention followed by
Kitchen INT8 Sol for the single RTX 5090 H3 quality matrix. All-dense FP4 showed
visible artifacts and is not approved as a default. Preserve Turbo's four dense
then four sparse steps and regular generation's ten dense then twenty sparse
steps, including the model's protected paths and rows.

Runtime owns hardware, dtype, site and parallelism eligibility. H3 owns its
ordered model preference through the existing `choose_attention` contract;
explicit request pins continue to take precedence. SM120/SM121 Blackwell uses the
approved hybrid where supported. Other hardware retains its existing H3 preference;
source-supported Kitchen INT8 dense plus Sol is an explicit comparison candidate,
not a claim of better H100/4090/Ampere performance. Unsupported devices must retain
an explicit supported outcome, never pretend to serve an unavailable backend.

The executed Sage3 CUDA entrypoint (pinned upstream `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`)
accepts SM120 and SM121 and rejects SM100/B200 despite the setup build list. Dense
and sparse Kitchen kernels support SM80+ for this recipe. Actual reviewed video
proof is on RTX5090/SM120; source admission is not a new hardware speed measurement.
Degree1 uses the shared-QKV producer; higher degrees use the head-local materialized
hybrid through existing Ulysses exchange. That route needs its own CLI execution
proof. Full-dense FP4 remains explicit only.

The matrix cohort is Runtime wheel SHA256
`4460767713804fb09722222af816d8663862e6f51c6211d65d33472db8d2c03a`.
Its attention producer/shared-QKV/low-memory Sage implementation remains private
in PRs #1240, #1242, #1245 and #1247 at this branch's base. This integration ports
the required attention paths while preserving newer master compile behavior.
Unrelated FP8/BF16 comparison kernels and all-dense default promotion are outside
this change. The independent bake/GEMM/fusion optimization PRs remain separate.

## Validation and ordinary CLI qualification

CPU checks on this integration: 192 focused checks passed with ten GPU/dependency
skips, including actual Gloo exchanges at degrees2/4/7/8. The final policy/startup
subset passed51 checks, including atomic processor rollback. The full394-file
mypy check and CI bug-class lint pass. These are CPU/source results, not GPU speed
measurements. Diffusers0.40 and0.41 source cohorts are qualified separately.

The architecture scanner is already red on base4f5dd8e3: static describe import,
author/internal package import, environment authority and an Any-count ceiling.
The candidate has the same four failing fences (15/19green); its new tensor and
processor adapter boundaries add91 Any annotations (970to1061). This change does
not relax those fences or claim they are green.

Root owns installation, rental and GPU work. On an owned, idle rented machine,
install the reviewed current Runtime candidate through ordinary Cozy CLI, retain
its actual Runtime/TensorFS/agent and GPU identities, and start a fresh executor.
Do not mix these results with the original private0.19matrix.

1. Run the existing local H3 package's `fl2va_turbo` through `cozy run` with the
   retained15-second sparring input and no attention override. Preserve all eight
   authored steps. Run an untimed warmup, then a measured request. Confirm the
   observed shared-QKV FP4-dense/INT8-Sol implementation and192 sparse block calls
   for the retained50-block configuration; a fallback receipt does not qualify
   the approved default.
2. Run regular `fl2va` at the same15-second geometry, preserving all30 steps and
   the10-dense/20-sparse policy. Check actual dispatch and finished AV output.
3. Exercise an explicit supported override and then a default request on the
   same executor. The default must restore with original semantic/custody hooks.
4. Run the low-memory Sage gate through an ordinary local Cozy package on a
   non-default PyTorch CUDA stream, including quantization, attention, early
   release/reuse of owned intermediates and result comparison. Retain actual
   CUDA stream/kernel evidence; source comments alone are insufficient.
5. For degree2/4/8, qualify the head-local materialized hybrid with actual CLI
   runs and observed per-rank dispatch before claiming multi-GPU performance.
   CPU Gloo establishes exchange wiring only. H100/4090 Kitchen candidates are
   explicit comparison choices; their current defaults remain unchanged.

Merge/publication uses the normal Runtime release flow after integration review
and actual CLI proof. Any combined fusion/bake candidate is recorded separately;
the experimental alternate GEMM is not required or included here. There is no
new approval flow and no automatic installation by this task.
