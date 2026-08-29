# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #84's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact EndpointEnvironmentSpec/2, EndpointBundle/2, and ResolvedWheelSet/3 bytes through
`endpoint_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,930-byte WheelhouseManifest/3 payload.

- Producing Tensorhub commit: `93c5c79a715092e245ac3db364475d6ea8d27ba1`

| file | SHA-256 |
|---|---|
| `base-worker-image-wheelhouse.json` | `cb6c46c667ba43897c7d3cc5ea26e44484b83d54a47344dfffdd5e2ddb24ffe1` |
| `endpoint-bundle.json` | `76ddb2ed3c7de4357f5d7c1898fcd328820da67bf6f2dc7ae8758fce64230800` |
| `endpoint-environment-spec.json` | `1c51b6f924bddee706c196e4b419a8c256bca6dba1b0df19e689a228e4e8b695` |
| `fixture-index.json` | `097a2dddb8e4f6908a2c8f65eef67637dccefb1885cf547d18e87571f6efa6e8` |
| `project-wheel.json` | `e374821cce113ced053fe870c067fb2252ad32b649b5539ebb4847bf639aed22` |
| `resolution-lock.json` | `436a122eb921d2e9b3196c95390df7e633f198816a36ed33cbd962fdff0a7d63` |
| `resolved-wheel-set.json` | `c402ba85dbf8e943455d2415f6d59e491b07f1ab3b3d885ef073c06eabdbbcb7` |
| `wheels/proof_endpoint-1.0.0-py3-none-any.whl` | `a1616ae8fb7c8bd28972f586b67f23f7445c15fd6ba54dfd7f67621b206f9cde` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
