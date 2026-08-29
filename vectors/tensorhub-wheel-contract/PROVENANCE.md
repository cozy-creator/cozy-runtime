# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #103's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact PackageEnvironmentSpec/2, PackageBundle/3, and ResolvedWheelSet/3 bytes through
`package_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,930-byte WheelhouseManifest/3 payload.

- Producing Tensorhub commit: `6dc35df`

| file | SHA-256 |
|---|---|
| `base-worker-image-wheelhouse.json` | `650f9e4010062d0895582a3e2d3e3c2ea21130d1b9ca30ed0de0f43eb1d2d655` |
| `source-tree.json` | `6d4a297f2e1faae52a71f9260ff1507ca67823322ac075bbbf37c7c122f99c05` |
| `package-bundle.json` | `110538d9a02da2b09843ee2c1b58713f69f77708cdbbf467c015d04790b368bb` |
| `package-environment-spec.json` | `12a81bdb7549dbcb706644468ce13407c158e6f66c212017449b7f80f4296c03` |
| `fixture-index.json` | `f7beb3d9d4c4c7f865fd4af8078d6223a64f18ac1e75dedb092157ef4d8caaf7` |
| `project-wheel.json` | `fb196cac5916f5890a468075cfdc2dfa8134674b4abb9fda7a2a14ad8426f858` |
| `resolution-lock.json` | `1a1f34b8ada8cfd08d311208c24932a80fb52a147e5af374289257367e784b55` |
| `resolved-wheel-set.json` | `f0dc0011192e18b55352ca91748cb268db0d3d5d992bfa6f2b6612fb1c9ca112` |
| `wheels/proof_package-1.0.0-py3-none-any.whl` | `8dbcdcafce5f9b3a8871eb6e930f22cceb94a2348c183a61645f595d3b90fca1` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
