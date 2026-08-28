# Tensorhub base-worker-image ownership vector

`tensorhub-base-worker-image-wheelhouse-v2.json` is copied byte-for-byte from Tensorhub PR #74's
Go-authored `vectors/base-worker-image-wheelhouse.json` after its stored-byte hardcut. It is the
canonical no-newline payload the image embeds, not the repository-text spelling reparsed or
reserialized by Runtime.

- Producing Tensorhub commit: `9c4ae40a45b59e7f93793ca02885988e41bce87a`

- Length: 23,742 bytes
- SHA-256: `cdb6d7753c57049e8801673da26fa67c3c39daf39ab18a4aae1afff0ab99bf1f`
- Census: 82 normalized distributions, 359 import roots, 5 control wheels

TensorFS is the source-reproducible manylinux2014 wheel built from pinned commit `4f3d16d` in
Tensorhub's digest-pinned maturin container: `sha256:7aa6d6388b1b2bcc0556d71ccfe58a043c9ba3ff4c8a113e541ac7bbff307a16`
/ 1,027,463 bytes. It replaces the developer-path-bearing manylinux_2_34 wheel; distribution and
import-root ownership are unchanged.

`checks/architecture.py --only base-ownership-vectors` runs Runtime's real strict parser over
these exact foreign-authored bytes and refuses any byte, schema, or census drift.
