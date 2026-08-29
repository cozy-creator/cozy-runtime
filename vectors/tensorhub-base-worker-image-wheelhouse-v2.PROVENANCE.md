# Tensorhub base-worker-image ownership vector

`tensorhub-base-worker-image-wheelhouse-v2.json` is copied byte-for-byte from Tensorhub PR #84's
Go-authored `vectors/base-worker-image-wheelhouse-cu126.json`. It is the
canonical no-newline payload the image embeds, not the repository-text spelling reparsed or
reserialized by Runtime.

- Producing Tensorhub commit: `390938935dde0f62e446c28f5fcfac0ef95bf385`

- Length: 23,930 bytes
- SHA-256: `7e30e22a6c951c40909e911cfa042d1b649a47dc8eef2a305cfdd390f60c7de5`
- Census: 82 normalized distributions, 359 import roots, 5 control wheels

TensorFS is the source-reproducible manylinux2014 wheel built from pinned commit `4f3d16d` in
Tensorhub's digest-pinned maturin container: `sha256:7aa6d6388b1b2bcc0556d71ccfe58a043c9ba3ff4c8a113e541ac7bbff307a16`
/ 1,027,463 bytes. It replaces the developer-path-bearing manylinux_2_34 wheel; distribution and
import-root ownership are unchanged.

The control Runtime wheel is the exact source-first base input recorded by Tensorhub:
`sha256:4ed12c4af9fa2d1ac6a704e2f335c030f98a9bade2c724b8e62868822beb3e24`
/ 697,437 bytes. This vector copy does not reinterpret those packaged bytes.

`checks/architecture.py --only base-ownership-vectors` runs Runtime's real strict parser over
these exact foreign-authored bytes and refuses any byte, schema, or census drift.
