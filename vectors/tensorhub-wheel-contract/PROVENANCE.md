# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub's Go-authored
`vectors/wheel-contract/` output for the uv 0.12.7 base worker image update. Runtime neither
recreates its JSON shape nor reserializes it: the `base-ownership-vectors` architecture fence
verifies the fixed SHA-256 identities, sends the exact PackageEnvironmentSpec/2,
PackageBundle/3, and ResolvedWheelSet/3 bytes through `package_overlay_input` with the exact
foreign wheel files, then materializes those wheel grants against Tensorhub's exact 24,246-byte
WheelhouseManifest/3 payload.

- Producing Tensorhub commit: `a7d1391`

| file | SHA-256 |
|---|---|
| `base-worker-image-wheelhouse.json` | `330ea6ec84c2012a3e73772ffaa0b69ad7f0381e244a709dd70a91a21fef795a` |
| `source-tree.json` | `6d4a297f2e1faae52a71f9260ff1507ca67823322ac075bbbf37c7c122f99c05` |
| `package-bundle.json` | `b150c4f2642c5b3f78663a371779bc4ee6950ab29b24c1b20e1252aa1ae5cf4d` |
| `package-environment-spec.json` | `35ae314fa0333930bb59b13fe82cb3830fa9d723baba432334aa7e24762a787c` |
| `fixture-index.json` | `179f3f1569bdc369e9b119bcb7de94c0a9fabb24a2d29d14e87f954d72d4632e` |
| `project-wheel.json` | `fb196cac5916f5890a468075cfdc2dfa8134674b4abb9fda7a2a14ad8426f858` |
| `resolution-lock.json` | `57d1ec1c637d6ebbff889bc3f59631452d45cbffd5616807f1f9627c8da4f3ca` |
| `resolved-wheel-set.json` | `e83ad3b049134707bdb093167c83f6fb74f196be723b77931a55cdb0b70859f7` |
| `wheels/proof_package-1.0.0-py3-none-any.whl` | `8dbcdcafce5f9b3a8871eb6e930f22cceb94a2348c183a61645f595d3b90fca1` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
