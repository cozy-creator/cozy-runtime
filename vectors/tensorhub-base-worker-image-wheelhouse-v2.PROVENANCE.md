# Tensorhub base-worker-image ownership vector

`tensorhub-base-worker-image-wheelhouse-v2.json` is copied byte-for-byte from Tensorhub PR #74's
Go-authored `vectors/base-worker-image-wheelhouse.json` after its stored-byte hardcut. It is the
canonical no-newline payload the image embeds, not the repository-text spelling reparsed or
reserialized by Runtime.

- Producing Tensorhub commit: `bce0a399e3c9c95d51ee80aed7ad2fdcb3c94776`

- Length: 23,930 bytes
- SHA-256: `c01451e80dd43a6b8dae8ecaf265003ffdb93e4dfee8b7da51381f356af34fd8`
- Census: 82 normalized distributions, 359 import roots, 5 control wheels

TensorFS is the source-reproducible manylinux2014 wheel built from pinned commit `4f3d16d` in
Tensorhub's digest-pinned maturin container: `sha256:7aa6d6388b1b2bcc0556d71ccfe58a043c9ba3ff4c8a113e541ac7bbff307a16`
/ 1,027,463 bytes. It replaces the developer-path-bearing manylinux_2_34 wheel; distribution and
import-root ownership are unchanged.

The control Runtime wheel is merged Runtime `a0205994d4f77fd429842ba2438017243490ab1a`,
`sha256:f831680b755771dbec178d61ce213e165dade5c59c22f62a83081c01430567f5`
/ 678,991 bytes. This vector-only follow-up does not enter that wheel's packaged `src/` tree.

`checks/architecture.py --only base-ownership-vectors` runs Runtime's real strict parser over
these exact foreign-authored bytes and refuses any byte, schema, or census drift.
