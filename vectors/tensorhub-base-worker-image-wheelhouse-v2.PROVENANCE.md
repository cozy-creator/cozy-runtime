# Tensorhub base-worker-image ownership vector

`tensorhub-base-worker-image-wheelhouse-v2.json` is copied byte-for-byte from Tensorhub PR #84's
Go-authored `vectors/base-worker-image-wheelhouse-cu126.json`. It is the
canonical no-newline payload the image embeds, not the repository-text spelling reparsed or
reserialized by Runtime.

- Producing Tensorhub commit: `b5216c4109cd4b28d4eb9ecb7a7742182b410d8b`

- Length: 23,930 bytes
- SHA-256: `b37eb3300a08f2edf29bb0bd963c898e1ece9fdb08b2998cf9193be65d4dc74d`
- Census: 82 normalized distributions, 359 import roots, 5 control wheels

TensorFS is the source-reproducible manylinux2014 wheel built from pinned commit `4f3d16d` in
Tensorhub's digest-pinned maturin container: `sha256:7aa6d6388b1b2bcc0556d71ccfe58a043c9ba3ff4c8a113e541ac7bbff307a16`
/ 1,027,463 bytes. It replaces the developer-path-bearing manylinux_2_34 wheel; distribution and
import-root ownership are unchanged.

The control Runtime wheel is the exact source-first base input recorded by Tensorhub:
`sha256:a1b4815666f6cffe0a0ce45701c770d4bc62608d94d4e5ad225f1af4880c6f4c`
/ 681,258 bytes. This vector copy does not reinterpret those packaged bytes.

`checks/architecture.py --only base-ownership-vectors` runs Runtime's real strict parser over
these exact foreign-authored bytes and refuses any byte, schema, or census drift.
