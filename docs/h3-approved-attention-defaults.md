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
explicit request pins continue to take precedence. Consumer Blackwell uses the
approved hybrid where supported. Other hardware may use Kitchen INT8 dense plus
Kitchen INT8 Sol only where upstream device code and Runtime admission support
it. Unsupported devices must retain an explicit supported outcome, never pretend
to serve an unavailable backend.

The matrix cohort is Runtime wheel SHA256
`4460767713804fb09722222af816d8663862e6f51c6211d65d33472db8d2c03a`.
Its attention producer/shared-QKV/low-memory Sage implementation remains private
in PRs #1240, #1242, #1245 and #1247 at this branch's base. This integration ports
the required attention paths while preserving newer master compile behavior.
Unrelated FP8/BF16 comparison kernels and all-dense default promotion are outside
this change. The independent bake/GEMM/fusion optimization PRs remain separate.

## Qualification

Implementation and CPU checks are in progress. No current master integration has
been GPU qualified, installed, or published by this task. Ordinary CLI runs must
verify selected implementations, dense/sparse counts, all authored steps, and
output video before promotion. Existing 5090 matrix evidence is not an H100 or
4090 hardware execution claim. Root owns rental/GPU work and live installation.
