# Tensorhub PackageRelease contract fixture

`package-release-v1.json` is copied byte-for-byte from Tensorhub's
`vectors/package-release-v1.json` at Tensorhub commit
`47bb83f2b0c712d88f0ae7184d6c7b6c26c18e35`. Its stored file includes one trailing LF and hashes to
`5f868fe549a22151e3a925d5638a13a315ae3508b16cbbad07ca0382d552313e`; the canonical document is
the preceding 968 bytes and has identity
`sha256:f894cff72dce53a2f85943bc10d1a1226f3fcacc983faddad6afbd6d61c425cf`.

Runtime's `tensorhub-package-release` architecture fence consumes those exact canonical bytes,
requires nested `{ref, distribution, filename, import_roots, tags, version}` project and dependency
wheel facts, joins them with an inline PlacementSet/1 Environment, and proves the retired flat
WheelFact shape refuses. The two small wheel files are retained from the earlier Tensorhub vector
solely to exercise the complete Runtime join.

`base-worker-image-wheelhouse-cu130.json` is copied byte-for-byte from Tensorhub commit
`8b8d50cf41db5dbf05f0563012fd354b765f5b27`, its single approved CPython 3.12 / official PyTorch
CUDA 13.0 producer. It contains the Runtime `0.0.11` wheel
(`sha256:9ad5642698d4879d4982152c5f88d0599a2b2c2f418f20c38bbb7d7fc5760334`) and TensorFS
`0.0.3` wheel (`sha256:55db3ee75eda51e77ef9c821d5a7b62294036b67c893bccb1bab910b783f9269`).
Runtime CI checks these local producer bytes only: 25,305 bytes with SHA-256
`73738b08458a6c83cae1b97a9675f775d27b616e08a7d034b85a481e63b1b2da`.
