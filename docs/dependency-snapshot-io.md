# Local snapshot verification I/O

Owner: `/root/dependency_snapshot_fix`  
Branch: `perf/dependency-snapshot-io-20260913`  
Base: `9db4334de706cd9cfbe3ca0e9fb993c10cdad978`

The local installed dependency snapshot copied each file, verified its installed
RECORD hash, rehashed it to rewrite RECORD, and rehashed it again for the snapshot
manifest. Package preparation then verified the full manifest twice. The copies
are needed when the source is a mutable local venv; the repeated digest calculations
are not independent custody boundaries.

Retain each verified digest and byte length in memory during snapshot creation.
Use that result for unchanged RECORD rows and the snapshot manifest. Newly written
launchers and rewritten RECORD files receive new hashes. After package wheels and
optional numerical supplements are installed, verify every final dependency file
once before publication. No digest/version cache, new database, shared mutable
inode, source hook execution, or change to content identity is introduced.

When reflink is unavailable, the existing Python copy loop now calculates the
RECORD hash and canonical SHA256 while copying. A successful reflink still gets its
target verification read. Both paths retain source inode/size/mtime/mode checks,
flush/fsync, and the final post-install dependency verification. SHA384/SHA512 RECORD
hashes and canonical SHA256 are calculated together from the same chunks.

## Measurement

`scripts/profile-dependency-snapshot.py` drives the real snapshot writer and final
verification over 64 MiB of dependency payload, plus metadata and a rewritten
console launcher (69 retained files). The original source is pinned to commit
`9db4334de706cd9cfbe3ca0e9fb993c10cdad978` in the evidence directory.

| memory-backed fixture | before | after |
| --- | ---: | ---: |
| process bytes read (`rchar`) | 335,573,013 | 134,245,538 |
| process bytes written (`wchar`) | 67,120,031 | 67,120,031 |
| CPU time | 0.4330 s | 0.2876 s |
| elapsed | 0.4330 s | 0.2876 s |
| physical disk reads / writes | 0 / 0 | 0 / 0 |
| peak RSS | 49,272 KiB | 47,452 KiB |

This removes three payload read passes: 60% fewer process reads and 33.6% less CPU
time in this fixture. All variants produce exactly the same manifest SHA256:
`8d7f9d74781bed3b3b1ef630b95e238ee3886728f4fafba31fe6e4fe47723916`.

Disk-backed runs also measured the same read reduction and unchanged physical
writes, but elapsed varied with concurrent work: original 0.814 / 0.893 s versus
streaming 1.689 / 1.916 s. Those samples do not establish a disk or full-job speedup.
`rchar` includes cached reads and must not be described as physical disk traffic.
The benchmark covers snapshot plus final verification, not the entire Creator
capture or a remote model workload. File copies and per-file durability remain;
the separately removed duplicate post-install verification is not included in the
benchmark's before/after interval.

## Verification and scope

All 29 dependency/carrier/core-capture tests pass, including SHA256/SHA384/SHA512
and unhashed RECORD entries, source bytes changed with restored mtime, concurrent
uv hardlink churn, two distributions claiming the same file, changed rewritten
launchers, and final package overwrite detection. The real core SDK and authored
root-extra captures still survive changes to the original source code. Ruff and
focused mypy checks pass.

Evidence: `outputs/dependency-snapshot-io-20260913/` contains the before, intermediate,
streaming, warmed-source, memory-backed, test and mypy outputs. Runtime PRs 464 and
465 are merged source only; no published Runtime release or shared SDK was changed
by this work. The selected authored-root-extra contract and shared immutable
dependency generation reuse remain separate follow-ups, tracked with proto-052.
