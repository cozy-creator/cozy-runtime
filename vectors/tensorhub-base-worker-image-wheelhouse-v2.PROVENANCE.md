# Tensorhub base-worker-image ownership vector

`tensorhub-base-worker-image-wheelhouse-v2.json` is copied byte-for-byte from Tensorhub PR #74's
Go-authored `vectors/base-worker-image-wheelhouse.json` after its stored-byte hardcut. It is the
canonical no-newline payload the image embeds, not the repository-text spelling reparsed or
reserialized by Runtime.

- Producing Tensorhub commit: `36f5eb95961802fd5c90090e23cbf44938341da5`

- Length: 23,687 bytes
- SHA-256: `d27017b3b0173a373ce70df3dabaea285f4b85646a730daec3b3c0765b7430c8`
- Census: 82 normalized distributions, 359 import roots, 5 control wheels

`checks/architecture.py --only base-ownership-vectors` runs Runtime's real strict parser over
these exact foreign-authored bytes and refuses any byte, schema, or census drift.
