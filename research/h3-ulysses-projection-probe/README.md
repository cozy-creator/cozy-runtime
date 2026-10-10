# H3 projection partition check

This ordinary-CLI diagnostic uses two granted RTX 5090 ranks, the real native rowwise FP8
linear modules, and the real Turbo LoRA hooks. It compares full-document 4096-row QKV
projection chunks with uneven local shards (54613/54612 rows), materialized projection,
same-configuration repeats, and actual Ulysses transport. Three states cover inactive,
synthetically baked, and unbaked rank64 LoRA. H3's QKV projections map5376 inputs to7168 outputs.
The output projection also checks7168-to5376 full versus local calls and repeat equality.
LoRA modes are compared within each mode, never as a baking-equivalence claim.
Final GPU capture must use the combined candidate Runtime that supplies
`attention_producer._project_qkv`. The probe calls this actual CP assembly helper for its
exchange inputs and timings; it compares its values with the existing single-GPU chunk
projection. There is no fallback to a lookalike implementation. The source dependency pin
is replaced with the candidate wheel in the frozen portable carrier before GPU execution.

The existing small diagnostic checkpoint admits the Runtime group. Projection weights are
deterministic synthetic operands generated within that granted call; this does not validate
TensorFS weight or LoRA baking. No video is generated and no serving default changes.

Before renting: run CPU transport and actual Diffusers report-hook checks, freeze the
ordinary CLI capture with the verified portable Diffusers wheel, and review tensor liveness.
Submit only through `cozy run ./research/h3-ulysses-projection-probe/verify --rental=NAME`.
Use the existing private diagnostic checkpoint explicitly, preserve its report, and release
the rental promptly. Do not silently lower dimensions or resubmit after a failure.
