# Tensorhub base-worker-image ownership vector

`tensorhub-base-worker-image-wheelhouse-v2.json` is copied byte-for-byte from Tensorhub PR #74's
Go-authored `vectors/base-worker-image-wheelhouse.json` after its stored-byte hardcut. It is the
canonical no-newline payload the image embeds, not the repository-text spelling reparsed or
reserialized by Runtime.

- Producing Tensorhub commit: `3d40b38a19d8e3ede6c44c11b6208ba686afd390`

- Length: 23,687 bytes
- SHA-256: `4cbfafc9afcfe3bbc148070bf394a1d585ded4d5f9a6a33795c6e6a924ce1d0a`
- Census: 82 normalized distributions, 359 import roots, 5 control wheels

`checks/architecture.py --only base-ownership-vectors` runs Runtime's real strict parser over
these exact foreign-authored bytes and refuses any byte, schema, or census drift.
