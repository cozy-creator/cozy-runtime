# Current Runtime wheel, host qualification

Runtime source `88b7f7619a8d1669bd8ae89e9e285a302ac3c061`; companion Creator
source `0a37d56a8ab4758e063f7470e6312922c177c730`. Owner
`/root/current_cohort_restart`, assigned by `/root`. Subsequent commits contain
only these receipts/documentation; the packaged source identity stays 88b7.

Artifact:
`outputs/varena-v2-p0-20261031/runtime-build-88b7f7619/dist/cozy_runtime-0.37.2+dev.g88b7f761-cp312-abi3-manylinux_2_28_x86_64.whl`

SHA256 `c35299925087fac0c300b768802761bd1f503b9897f3f6a67d47af1795a78e5a`.

The established `build-runtime-candidate-v4.py` repacked the source offline in
16.789s using the verified a09 prefix. No native compilation ran. All 65 native
source/build inputs remain unchanged; the real current import produced 49
set/architecture specs and 285 Triton keys, matching frozen source and every
manifest key. Launcher source matched. All 581 native members were rehashed in
both old wheels and the prefix before reuse, then in the completed new wheel.
The complete 832-row RECORD and 225 Python/type-stub members match; provenance
and CLI report the correct version and source. Prior artifacts were preserved.

Six sequential nice 19/one-thread host checks passed:

| Check | Wall seconds |
| --- | ---: |
| Import new wheel and changed modules; no CUDA init/driver mapping |2.918|
| Actual CLI version/provenance |0.419|
| Real Diffusers refiner/leaf/ATen contract |6.702|
| Exact production semantic expression and mutation/refusal checks |1.729|
| Strict framed startup: 9 cases, 7 rank selections |16.746|
| Updated actual-dispatch GPU fixture import only |4.553|

CPU imports used the isolated wheel extraction and existing d3e8 TensorFS wheel
`eadf34656d860e9e11b4f0b822d68495abd4759cad14cf20f2f5370d72b3ea2a`, read-only.
The d3e8 extension/core hashes were checked before import. This is not a new
TensorFS build from the current Creator dependency graph. Torch was 2.14.1+cpu;
Diffusers 0.41.0. CUDA remained uninitialized and libcuda unmapped. Host contracts
cover 12 activation dtype/row cases, exact wrapper output, actual leaf/provider
refusals and five semantic identity mutations. Metadata-only dispatch values are
not native GPU plan queries or execution evidence.

The strict startup script differs from the committed source script only in its
Runtime package root: it imports the **extracted wheel**. All protocol, error,
driver, source-hash and CUDA assertions remain. Sender frames are the prior real
Rust execution; current Creator's sender/protocol hashes are unchanged. Refiner
host/semantic scripts import packaged code; their source AST is exact by the
225-member closure check. No assertions were weakened.

No current worker/TensorFS build, cache borrowing, live install, GPU, rental,
merge or release was performed. Real refiner native dispatch, full H3 output,
independent two-GPU/per-rank startup and live updater qualification remain.
Earlier failure receipts retain their original artifact boundaries.
