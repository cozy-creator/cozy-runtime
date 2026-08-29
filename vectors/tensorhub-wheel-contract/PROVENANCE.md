# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #84's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact EndpointEnvironmentSpec/2, EndpointBundle/2, and ResolvedWheelSet/3 bytes through
`endpoint_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,930-byte WheelhouseManifest/3 payload.

- Producing Tensorhub commit: `390938935dde0f62e446c28f5fcfac0ef95bf385`

| file | SHA-256 |
|---|---|
| `endpoint-bundle.json` | `f95aa077347a5f3fc84c7f9d80b801fe38d121605b8d03fe3c61945afa7e4b8d` |
| `endpoint-environment-spec.json` | `154d4d77d513596237bde209077bc5761c278f61251d985b3a7470f3eb5c1afa` |
| `fixture-index.json` | `8809d5843a57b99e4bdd1abc4e4accf631c16ba73c694c4b1c53b518eaffc013` |
| `project-wheel.json` | `e374821cce113ced053fe870c067fb2252ad32b649b5539ebb4847bf639aed22` |
| `resolution-lock.json` | `436a122eb921d2e9b3196c95390df7e633f198816a36ed33cbd962fdff0a7d63` |
| `resolved-wheel-set.json` | `203b86882159262c94f6d72daf38623c11bef576ed6d824118da786403e6aeac` |
| `wheels/proof_endpoint-1.0.0-py3-none-any.whl` | `a1616ae8fb7c8bd28972f586b67f23f7445c15fd6ba54dfd7f67621b206f9cde` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
