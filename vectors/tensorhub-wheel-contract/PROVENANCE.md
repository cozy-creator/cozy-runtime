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
(`sha256:ad48274299c80ed2013cfa4259672517fc8b6718637a173196434c83320fafbb`,
744,067 bytes) and TensorFS `0.0.5`
(`sha256:a131c4688188dacd1518fbe93c417e64d6af575f65b52edac48bcb60f88d43a5`,
2,337,278 bytes). They also record the exact Click `8.5.0` control wheel
(`sha256:255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360`).
Runtime CI checks the canonical producer bytes directly: the CPU fixture is 24,968 bytes with
SHA-256 `6dd580aec64cc13fc1ac51150d8d7d9dadeb1ac43dac09c4802117189912556f`; the CUDA fixture is
25,538 bytes with SHA-256 `c9c000daf9e83f9a97ad4119d3309cf19c3c6c96aa5a50fdaf88f4a0523974d0`.
