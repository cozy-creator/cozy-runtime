# Installed environment proof

`run.sh` builds and installs the current Runtime wheel, then drives its public
`cozy-environment-proof` executable from `/` with source imports and network access disabled.
Python support prepares exact protocol documents and inspects stored bytes; shell owns every
Runtime process boundary.

The public proof seat is deliberately one-shot and requires a fresh root. The separate private
materializer proof covers content-keyed reuse, wheel-pool sharing and post-publication tamper
refusals.

The driver measures its complete installed distribution/import-root inventory into
`WheelhouseManifest/3`. A real fake NumPy is base-installed. One arriving `numpy` wheel refuses
with the image-owned version, and a differently named wheel exporting `numpy` refuses with the
base provider before any overlay root or receipt exists. A lying `import_roots` fact also refuses
against measured ZIP members. The ordinary resolved fixture exports a genuinely new root and
still imports from the published overlay. Legacy WheelhouseManifest/1 and ResolvedWheelSet/1
documents are refusal vectors, never compatibility inputs. Two differently named custom wheels
exporting one new root refuse as `overlay_import_root_duplicate` before extraction.
