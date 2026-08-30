# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub's Go-authored
`vectors/wheel-contract/` output for the final proto-013 base worker image. Runtime neither
recreates its JSON shape nor reserializes it: the `base-ownership-vectors` architecture fence
verifies the fixed SHA-256 identities, sends the exact PackageEnvironmentSpec/2,
PackageBundle/3, and ResolvedWheelSet/3 bytes through `package_overlay_input` with the exact
foreign wheel files, then materializes those wheel grants against Tensorhub's exact 25,503-byte
WheelhouseManifest/3 payload.

- Producing Tensorhub commit: `00de9e6c6f482054b483ebaca4e103e46d3520ae`

| file | SHA-256 |
|---|---|
| `base-worker-image-wheelhouse.json` | `f5fc85a3dee5a9e92318356323f51f7c7ad5ca32d3e76a5c8839e0e0b6477cad` |
| `source-tree.json` | `6d4a297f2e1faae52a71f9260ff1507ca67823322ac075bbbf37c7c122f99c05` |
| `package-bundle.json` | `f12a0a7170b7fa22ee572b5668c81db14501bb9a6746e11bb5b8fc1e266e5fa6` |
| `package-environment-spec.json` | `6144ab858c10e1b85ba1bc0e606e70c13441b23a447889f4233f90c11daeac5b` |
| `fixture-index.json` | `e91c55e4ce31fd8b7d027882e0a9dae4dce658e07d53a074f6dcc8a5e095b46a` |
| `project-wheel.json` | `fb196cac5916f5890a468075cfdc2dfa8134674b4abb9fda7a2a14ad8426f858` |
| `resolution-lock.json` | `57d1ec1c637d6ebbff889bc3f59631452d45cbffd5616807f1f9627c8da4f3ca` |
| `resolved-wheel-set.json` | `e5c3664d48456b61249fd369d75d68f3d1ee4b02c243adeff236bacd436e2b94` |
| `wheels/proof_package-1.0.0-py3-none-any.whl` | `8dbcdcafce5f9b3a8871eb6e930f22cceb94a2348c183a61645f595d3b90fca1` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
