# Private package dependency environments

cr-186 / proto057 implementation for Runtime 0.18.6. Worker protocol wire57
adds exact dependency requirements to private package preparation and revision
identity. The coordinated hard cut sets the protocol compatibility floor to57.

Private environments install the complete wheel closure, including executor Runtime,
Torch and ordinary transitive dependencies. Image inventory cannot supply missing
requirements. Wheel tags, Requires-Python, selected extras and transitive metadata
remain checked. Content and local installation identities use isolation version2.
Existing active generations are never rewritten.

Private dependency requirements contain sorted public wheel URL rows with exactly
one SHA256 each. No credentials, URL queries, fragments, markers, indexes, floating
pins or overlap with uploaded wheels. Remote wheel bytes use supervisor download
and disk admission. Local bytes use the existing Runtime egress boundary with disk
admission and hash verification. Partial remote preparations release their wheel
operation on failure as well as success. Local wheel staging is reusable derived
cache. Both paths use the same private uv installation and closure validator.

uv shares its cache under the configured machine dependency cache / uv-cache;
legacy callers without that explicit root use install_root / uv-cache. Hardlinks
share installed dependency payloads across separate package roots. Read-only sealing
changes only private files and directories; shared payload inode modes remain owned
by uv. Cleanup changes directory modes only. The retained venv is a trusted-owner
storage boundary; hardlinks are not an adversarial same-uid sandbox.

Baked artifact cache seeding is a separate image optimization. A baked version alone
is not evidence of exact reusable bytes. Alternate Torch/Triton preparation records
warnings on the worker durable activity lane as well as installer logs. It may need
additional download time/disk or lack image-specific optimizations. Actual native
loading still establishes driver/ABI compatibility.

Verification: focused tests cover conflicting dependency versions, ambient Python/uv
settings, missing transitive requirements, private direct URL fetch, offline reuse,
revision identity, warm reopening without uploaded carriers, and physical shared
inodes for an 8 MiB dependency installed into two separate package roots. Deleting
one generation leaves sibling bytes and inode permissions unchanged. Canonical wire
fixtures and generated bindings match the pinned worker-protocol source. Ruff,
focused mypy and architecture fences are checked. Ordinary installed Creator CLI
qualification and alternate real Torch/native-stack execution remain root integration
work; toy Torch metadata admission is not GPU execution evidence.
