# Tensorhub PackageRelease contract fixture

`package-release-v1.json` is copied byte-for-byte from Tensorhub's
`vectors/package-release-v1.json` at Tensorhub commit
`a8339345d061f03b243dd7a295819f12a2852de6`. Its stored file includes one trailing LF and hashes to
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
CUDA 13.0 base worker image manifests containing the same Runtime `0.0.11.dev0` wheel
(`sha256:c633b3e5d1b1aa81249cc61c63d1aadb6347a534935191a52cc553cabba34448`) and TensorFS
`0.0.3.dev0` wheel (`sha256:894e14456673adfe569a8f5c368ba96df878f2d75799c7058fd45d509068efa3`).
Runtime CI checks these local producer bytes only: CPU is 24,683 bytes with SHA-256
`0332d1afd11b222df7ef411ef6567cbe12af7ec8edf0309c5ccd846bcbf35406`; CUDA is 25,651 bytes
with SHA-256 `6614fddec7ab1ecf634293c4ff6c254f4fb2ed143d96022fbe9f6169deeede50`.
