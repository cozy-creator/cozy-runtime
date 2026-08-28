# Runtime proof scenario census

This census prevents a file move from being mistaken for evidence preservation. A public proof
must invoke an installed Runtime command. Direct private-module exercises are internal proofs.
Machine facts belong to qualification, and timing or memory measurements belong to benchmarks.

`migrated` means the replacement has run with equivalent assertions. `retired` means the
claimed interface was deliberately deleted and the current refusal was run instead. `pending`
means the old script remains the only local executor and must not be deleted yet.

| Existing owner | Modes / facts | Target owner | Status / equivalent evidence |
|---|---|---|---|
| `scripts/assets-live.py` | roundtrip, kill, refusals, egress, debts | internal asset proof; public debt receipt proof | pending; not rerun here |
| `scripts/bench.py` | CLI/import latency and RSS | `benchmarks/` | pending; not rerun here |
| deleted `scripts/cli-live.py surface` | process exit matrix, noninteractive AXI, exact/read-only descriptor, empty artifact UX | `proofs/cli/run.sh` | migrated 2026-08-27; fresh wheel, installed command, `/`, no `PYTHONPATH`, stdin closed |
| deleted `scripts/cli-live.py keys` | canonical/local/strict extra-key tiers and known-marker transcription | `proofs/cli/run.sh` fixture job | migrated 2026-08-27; old mode never invoked the CLI, replacement runs public author behavior through an installed CLI job |
| deleted `scripts/cli-live.py inputs` | typed local media, canonical invocation receipt and ordering; private transfer/RecordOwner fault plants | `proofs/cli/run.sh`; `proofs/internal/local_input_acceptance.py` | migrated 2026-08-27; public and explicitly private legs both run from the freshly installed wheel |
| deleted `scripts/cli-live.py identity` | real SDXL CLI-vs-attempt output, plan and ledger equivalence | current exit-6 local-binding refusal; `scripts/pipeline-live.py image` | retired 2026-08-27; model-bound `cozy-runtime run` was deleted with flat local binding records, so no public GPU result exists to compare and preserving the old path would revive removed authority |
| deleted `scripts/cli-live.py hubless` | network-cut SDXL inference and planted physical-floor refusal | `proofs/cli/run.sh`; `scripts/pipeline-live.py image/arms` | split 2026-08-27; installed weightless `--offline` runs with AF_INET/6 cut, while real pipeline output and quantified floor remain GPU-owned; the deleted local model-binding CLI path is not an acceptance contract |
| `checks/declared_facts.py` | descriptor facts have live writers | `checks/` | migrated 2026-08-27; hosted static check, path and CI owner now agree |
| `scripts/delivery-live.py` | variants, selection, refusals, calibration, serving | internal proof plus GPU qualification | pending; not rerun here |
| `scripts/derive-corpus.py` | adversarial derivation corpus | internal derivation proof | pending; not rerun here |
| `scripts/derive-verify.py` | H3/SDXL derivation, construction-owned fill order and refusals | internal derivation proof | pending migration; rerun 2026-08-27 with cr-041's 12-block order arm: construction `0,1,2,…,10,11`, lexical header `0,1,10,11,2`, serve enqueued the exact 52-row DestinationSet, and a same-set reordering refused; 111/111 green/red observations |
| deleted `scripts/environment-materializer-live.py` | historical full-venv reuse, wheel-store, RECORD rehash and external bytecode cache | `proofs/environment/run.sh`; `proofs/boot/run.sh` | retired 2026-08-28 (cr-027): the installed substrate deletes the full-venv serving path; public proof covers exact endpoint overlay intake/rebuild/refusals and the boot proof serves through the dynamic overlay |
| `scripts/environment-proof-live.py` | public environment cold/rebuild/refusals | `proofs/environment/run.sh` | migrated 2026-08-27; installed wheel from `/`, exact receipt bytes/digest/length, isolated imports, CUDA ABI, atomic race, closed-request and wheel refusals, no partial outputs |
| `checks/architecture.py` | architecture dependency scans | `checks/` | migrated 2026-08-27; hosted static check, path and CI owner now agree |
| `scripts/job-live.py` | job e2e, kill, refusals, identity, timing | internal job proof plus qualification/benchmark | pending; not rerun here |
| `scripts/leaf-live.py` | provider selection, carriers, initial fill, fresh-census restage, refusals | internal encoding proof plus qualification | pending migration; rerun 2026-08-27 against merged tfs-025/tfs-026 wheel: real CBOR ingest authored from the live construction census, initial native fill and decoded park→stage both returned exact `construction` sequence evidence, header-order red fired before copies, 22/22 green/red observations |
| `scripts/leaf-pod.py` | H3 fetch, ingest, probe, fill, forward, choose | `qualification/gpu/` | pending; dynamically imports pod script |
| `scripts/lifecycle-live.py` | process boot, cancellation, death, recovery | internal lifecycle proof plus platform qualification | pending; current hosted macOS owner |
| `proofs/internal/pull_stream_liveness.py` | one black-holed presigned stream is named, re-asked, and refused by name rather than held to the caller's deadline | itself | added 2026-08-28 (cr-055); real `pull.pull` against a loopback object store, three arms — clean control, one recoverable stall completing in 0.7 s, a permanent stall refusing `transfer_failed` after `FETCH_ATTEMPTS` asks and far inside a 120 s deadline |
| `scripts/materialize-launch-live.py` | adapter handoff, exact-document join, CAS admission | `proofs/boot/run.sh` plus this script | partially migrated 2026-08-28 (cr-048); the BOOT half is now a real containerized boot from a freshly built wheel with no provision document, reaching ready-and-empty and observed over both consumers (tensorhub's receipt, a renter's Claim). The remaining script keeps the closed-document join arms, which have no installed-command surface |
| `proofs/boot/run.sh` | ready-but-empty boot plus ready-and-loaded convergence; schema-rev-7 partial/terminal acquisition observations; cold endpoint/model intervals overlap at the origin barrier; the endpoint becomes an installed-base overlay and serves; remove/re-add proves warm endpoint download=0 and exact reuse; a mismatched following revision refuses while the converged placement remains serving | itself | extended 2026-08-28 (cr-027/cr-049) for cozy-creator's single `cozy-pod`; substrate stand-in, `--network none`, delegated executor cgroup, image Python plus endpoint-only overlay |
| `scripts/media-decode-live.py` | bounded PyAV inputs, clocks, transforms | internal media proof | pending; not rerun here |
| `scripts/media-live.py` | FLAC/MP4 codecs and scaled timing | internal media proof plus `benchmarks/` | pending; not rerun here |
| `scripts/media-stream-live.py` | 2/8-shot RSS and kill cleanup | internal media proof plus `benchmarks/` | pending; not rerun here |
| `scripts/model-contract-proof-live.py` | installed cold/rebuild identity, hidden omission, exact MCC, missing-binding refusal | `proofs/model-contract/run.sh` | migrated 2026-08-27; public commands from `/`, exact red exit 6, no output |
| `scripts/mxfp8-live.py` | CPU vectors, carrier, derivation, refusals | internal encoding proof | pending migration; rerun 2026-08-27 against merged tfs-025/tfs-026 wheel: exact construction-authored CBOR snapshot, verified lease, 29,568 stored vs 57,344 logical bytes, bit-exact independent dequant reference, 6/6 carrier observations |
| `scripts/mxfp8-native-probe.py` | native MXFP8 device facts, matrix, weights, speed, VRAM | `qualification/gpu/` plus `benchmarks/` | pending; not rerun here |
| `scripts/mxfp8-pod.py` | pod MXFP8 ingest, compare, derivation, block forward | `qualification/gpu/` plus `benchmarks/` | pending; not rerun here |
| `scripts/pipeline-live.py` | fixtures, poison/refusals, stage, image, bandwidth | internal pipeline proof plus qualification/benchmark | pending; not rerun here |
| `scripts/plan-conform.py` | objective/cache/spec/geometry/identity/envelope/consent | internal plan proof | pending; intended fact owner after duplicate deletion |
| `scripts/process-containment-live.py` | OS containment and restart recovery | internal lifecycle proof plus platform qualification | pending; not rerun here |
| `scripts/pull-live.py` | remote pull, streams, repair/resume, refusals | `proofs/pull/` plus network qualification | pending; not rerun here |
| `scripts/quant-live.py` | encoding production/ingest/dequant/refusals/image/timing | internal encoding proof plus qualification/benchmark | pending; not rerun here |
| `scripts/redarm.py` | planted architecture and live refusals | `checks/` plus owning internal proofs | pending; not rerun here |
| `scripts/remote-live.py` | TLS, credentials, grants, remote worker control | internal remote proof plus network qualification | pending; dynamically imports worker/lifecycle scripts |
| `scripts/slice-live.py` | slice, ledger, refusals, triage, witness, clone, timing | internal execution proof plus qualification/benchmark | pending; not rerun here |
| `scripts/tiered-cache-live.py` | real TensorFS worker-local CAS plus optional network cache; non-empty exact object sets, cloud/cache source receipts, corruption fallback, concurrent same-object pulls, and schema-rev-7 model downloaded/reused counter projection | itself | rerun 2026-08-28 against TensorFS `0f49f3b`: PASS with the built abi3 wheel; cold non-empty model bytes counted downloaded, network-cache bytes counted reused, and a throwing telemetry callback could not gate hydration |
| `scripts/verify.py` | author surface and examples end to end | internal author proof; installed CLI arm to `proofs/cli/` | pending; not rerun here |
| `checks/wire_schema.py` | generated wire-schema identity | `checks/` | migrated 2026-08-27; hosted static check, path and CI owner now agree |
| `scripts/worker-live.py` | worker attempts, death/recovery, journals, readiness, timing | internal worker proof plus qualification/benchmark | pending; not rerun here |

Five old entrypoints dynamically load other entrypoints (`assets-live`, `delivery-live`,
`leaf-pod`, `lifecycle-live`, and `remote-live`). Replacements must use ordinary
support modules or process contracts; moving that coupling under a new directory does not count
as migration.
