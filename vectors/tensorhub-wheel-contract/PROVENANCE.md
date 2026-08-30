# Tensorhub PackageRelease contract fixture

`package-release-v1.json` is copied byte-for-byte from Tensorhub's
`vectors/package-release-v1.json`. Its stored file includes one trailing LF and hashes to
`9fc018fcda64fee307ee7db28b632b7abe15dd77c290451525c632ab4a0c16fe`; the canonical document is
the preceding 962 bytes and has identity
`sha256:a7169ae741990c565ad3f6cd27ff2ce28c294aba53815156d2ad70422c39dbbd`.

Runtime's `tensorhub-package-release` architecture fence consumes those exact canonical bytes,
requires nested `{ref, distribution, filename, import_roots, tags, version}` project and dependency
wheel facts, joins them with an inline PlacementSet/1 Environment, and proves the retired flat
WheelFact shape refuses. The WheelhouseManifest and two small wheel files are retained from the
earlier Tensorhub vector solely to exercise the complete Runtime join.
