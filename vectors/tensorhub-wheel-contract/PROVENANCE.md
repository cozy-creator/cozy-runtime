# Tensorhub PackageRelease contract fixture

`package-release-v1.json` is copied byte-for-byte from Tensorhub's
`vectors/package-release-v1.json`. Its stored file includes one trailing LF and hashes to
`5f868fe549a22151e3a925d5638a13a315ae3508b16cbbad07ca0382d552313e`; the canonical document is
the preceding 968 bytes and has identity
`sha256:f894cff72dce53a2f85943bc10d1a1226f3fcacc983faddad6afbd6d61c425cf`.

Runtime's `tensorhub-package-release` architecture fence consumes those exact canonical bytes,
requires nested `{ref, distribution, filename, import_roots, tags, version}` project and dependency
wheel facts, joins them with an inline PlacementSet/1 Environment, and proves the retired flat
WheelFact shape refuses. The two small wheel files are retained from the earlier Tensorhub vector
solely to exercise the complete Runtime join.

`base-worker-image-wheelhouse-cpu.json` and `base-worker-image-wheelhouse-cu130.json` are copied
byte-for-byte from Tensorhub commit `ee0ff10d75da84e5341c9f2cd90f25e6ee1ec4e6`. They are the
released CPU and CUDA 13.0 base worker image manifests containing the same Runtime `0.0.10` wheel
(`sha256:94460296a7c6ad745d25693f8c67faa6543579ffb0779ee1f9b8056ef9a30ba8`) and TensorFS `0.0.2`
wheel (`sha256:5d5420f9571778ae884daf6fe31bc649318014cd644d94b17a7c465929be5d93`).
Runtime CI checks these local producer bytes only: CPU is 24,977 bytes with SHA-256
`2ea86f5131b1f95efd1f2bd3870efd52b94cdd25c623e438614e6ff3afb31cb8`; CUDA is 32,558 bytes
with SHA-256 `62be81f24f970df1db8dce059a06c208b6d35c1b277450c5e98ea495474b15f0`.
