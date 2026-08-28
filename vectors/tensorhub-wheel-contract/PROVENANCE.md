# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #74's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact EndpointEnvironmentSpec/1, EndpointBundle/2, and ResolvedWheelSet/2 bytes through
`endpoint_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,742-byte WheelhouseManifest/2 payload.

- Producing Tensorhub commit: `9c4ae40a45b59e7f93793ca02885988e41bce87a`

| file | SHA-256 |
|---|---|
| `endpoint-bundle.json` | `e1ecebdd65b794cfe8eeec9906b6531a4d172dfb88f6a6184854bec1c811477f` |
| `endpoint-environment-spec.json` | `1ade3c64be8da9a20d3723377386d270fc10807499f92fc25ea1d24880d96eee` |
| `fixture-index.json` | `b3fd2ea9e6973c5e028dc67ff1ed973fa681c515ef35a8a0047074b983898cdf` |
| `project-wheel.json` | `e374821cce113ced053fe870c067fb2252ad32b649b5539ebb4847bf639aed22` |
| `resolved-wheel-set.json` | `82307cc2d4d3580e0a9b617e93c18102e5a9d45e404e8d03543e3092f3068bd7` |
| `wheels/proof_endpoint-1.0.0-py3-none-any.whl` | `a1616ae8fb7c8bd28972f586b67f23f7445c15fd6ba54dfd7f67621b206f9cde` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
