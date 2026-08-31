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

The CPU and CUDA 13.0 wheelhouse fixtures are the exact candidate manifests for Runtime `0.0.19`.
They carry its twice-reproduced `cozy_runtime-0.0.19-py3-none-any.whl`
(`sha256:fc513bd48db8b915c695f32689606feb31d67d211ca8fc65f06c002d70586a53`,
740,940 bytes) and TensorFS `0.0.5`
(`sha256:a131c4688188dacd1518fbe93c417e64d6af575f65b52edac48bcb60f88d43a5`,
2,337,278 bytes). They also record the exact Click `8.5.0` control wheel
(`sha256:255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360`).
Runtime CI checks the canonical producer bytes directly: the CPU fixture is 24,968 bytes with
SHA-256 `54e7a11ca862d40ec65c279bf29b4a15d076714463e6f2c7d95d481f5c78e7b2`; the CUDA fixture is
25,538 bytes with SHA-256 `92aeecce5257a5050a8769fc380c37ddefd6346e6eb097f65dcd49ac5f57ed3c`.
