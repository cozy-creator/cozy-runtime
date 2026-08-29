# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #102's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact PackageEnvironmentSpec/2, PackageBundle/3, and ResolvedWheelSet/3 bytes through
`package_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,930-byte WheelhouseManifest/3 payload.

- Producing Tensorhub commit: `5277ada2da28d8e68732b2cd5c1a6d1323f288d1`

| file | SHA-256 |
|---|---|
| `base-worker-image-wheelhouse.json` | `650f9e4010062d0895582a3e2d3e3c2ea21130d1b9ca30ed0de0f43eb1d2d655` |
| `package-bundle.json` | `9a02bb4f507224140f1aa2c5281dfc13d4b2c0e200af2617c1593388fcd63105` |
| `package-environment-spec.json` | `d17897b9c7048f359660e368c668be33a9c913ab9406eb0262e467f891f22819` |
| `fixture-index.json` | `735da16e57ecaf62d3169b9d2c23483d6f2046cff516883d41fb289070a422b6` |
| `project-wheel.json` | `fb196cac5916f5890a468075cfdc2dfa8134674b4abb9fda7a2a14ad8426f858` |
| `resolution-lock.json` | `1a1f34b8ada8cfd08d311208c24932a80fb52a147e5af374289257367e784b55` |
| `resolved-wheel-set.json` | `f0dc0011192e18b55352ca91748cb268db0d3d5d992bfa6f2b6612fb1c9ca112` |
| `wheels/proof_package-1.0.0-py3-none-any.whl` | `8dbcdcafce5f9b3a8871eb6e930f22cceb94a2348c183a61645f595d3b90fca1` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
