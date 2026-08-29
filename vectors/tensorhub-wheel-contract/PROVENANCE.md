# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #84's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact EndpointEnvironmentSpec/2, EndpointBundle/2, and ResolvedWheelSet/3 bytes through
`endpoint_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,930-byte WheelhouseManifest/3 payload.

- Producing Tensorhub commit: `a34f5b45905fefbdc89636c315d6f16aaca1f48b`

| file | SHA-256 |
|---|---|
| `base-worker-image-wheelhouse.json` | `650f9e4010062d0895582a3e2d3e3c2ea21130d1b9ca30ed0de0f43eb1d2d655` |
| `endpoint-bundle.json` | `ac3b9b3482fbc673ec62d4b2143b64254454c8e45f06987eedf4f39fb2701fb9` |
| `endpoint-environment-spec.json` | `a34714c0610936206e1744a565da41301f58c8e1003b15ed55491e84077bca75` |
| `fixture-index.json` | `2b6c46e40b0408c449d1a3d68751dc27c916a87a033fc516edfd233b621a26c7` |
| `project-wheel.json` | `e374821cce113ced053fe870c067fb2252ad32b649b5539ebb4847bf639aed22` |
| `resolution-lock.json` | `436a122eb921d2e9b3196c95390df7e633f198816a36ed33cbd962fdff0a7d63` |
| `resolved-wheel-set.json` | `432c0052384b3c54ba0ca5d2336fd57b14633bbd6c8a7dcfce8ff5d4be7caa2d` |
| `wheels/proof_endpoint-1.0.0-py3-none-any.whl` | `a1616ae8fb7c8bd28972f586b67f23f7445c15fd6ba54dfd7f67621b206f9cde` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
