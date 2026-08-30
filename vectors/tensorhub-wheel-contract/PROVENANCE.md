# Tensorhub PackageRelease contract fixture

`package-release-v1.json` is copied byte-for-byte from Tensorhub's
`vectors/package-release-v1.json`. Its stored file includes one trailing LF and hashes to
`add41cdc1e6f19e43f9f7f411fd07dd038ad62126c2299c2c234950b71682725`; the canonical document is
the preceding 773 bytes and has identity
`sha256:26d75c61b144538a6954c276c6d903816623325f7f75fa845652aaedbfeb8630`.

Runtime's `tensorhub-package-release` architecture fence consumes those exact canonical bytes,
requires nested `{ref, distribution, filename, import_roots, tags, version}` project and dependency
wheel facts, joins them with an inline PlacementSet/3 Environment, and proves the retired flat
WheelFact shape refuses. The WheelhouseManifest and two small wheel files are retained from the
earlier Tensorhub vector solely to exercise the complete Runtime join.
