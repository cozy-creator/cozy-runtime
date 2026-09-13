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

Measurement and focused mutation/collision verification are pending.
