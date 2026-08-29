# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #103's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact PackageEnvironmentSpec/2, PackageBundle/3, and ResolvedWheelSet/3 bytes through
`package_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,930-byte WheelhouseManifest/3 payload.

- Producing Tensorhub commit: `a282119`

| file | SHA-256 |
|---|---|
| `base-worker-image-wheelhouse.json` | `33e0e0f17ee0e1a9d41e8acca0fcf14362280597a50dbe0229c76f65fbfdaafc` |
| `source-tree.json` | `6d4a297f2e1faae52a71f9260ff1507ca67823322ac075bbbf37c7c122f99c05` |
| `package-bundle.json` | `0ec72a58775c618bf4d5980856e2458692f8582f60dfe5a4477f7bf6a5c3c6ae` |
| `package-environment-spec.json` | `df17c9c92134b0bb7bb226bcee01ad5f6f6d244baeb61418af61295144277f` |
| `fixture-index.json` | `75324dfae56425bbd26fb6d54ec2fc726a01c81dcb9ff02f4d9d51e7cd028ece` |
| `project-wheel.json` | `fb196cac5916f5890a468075cfdc2dfa8134674b4abb9fda7a2a14ad8426f858` |
| `resolution-lock.json` | `57d1ec1c637d6ebbff889bc3f59631452d45cbffd5616807f1f9627c8da4f3ca` |
| `resolved-wheel-set.json` | `93e668e46ae4d89bf8dfbe85b3f79d68e94ae1ddae3e8938f8bddea82fc27b36` |
| `wheels/proof_package-1.0.0-py3-none-any.whl` | `8dbcdcafce5f9b3a8871eb6e930f22cceb94a2348c183a61645f595d3b90fca1` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
