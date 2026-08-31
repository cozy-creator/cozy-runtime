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

`base-worker-image-wheelhouse-cpu.json` and `base-worker-image-wheelhouse-cu130.json` are copied
byte-for-byte from the same Tensorhub commit's CPython 3.12 producer. They are the CPU and official PyTorch
CUDA 13.0 base worker image manifests containing the same Runtime `0.0.11` wheel
(`sha256:032f372a77290160ca118adfae0a6928540d0e837975366032fa2d4e37161ce1`) and TensorFS
`0.0.3.dev0` wheel (`sha256:894e14456673adfe569a8f5c368ba96df878f2d75799c7058fd45d509068efa3`).
Runtime CI checks these local producer bytes only: CPU is 24,668 bytes with SHA-256
`f1027c384b2ab34ecada82bf494bfaab92d019907005128aa9c78b2d4d3c392a`; CUDA is 25,636 bytes
with SHA-256 `5437560d941af71770c9625f8b15a9b2d4618432215432652e6120e164995f6c`.
