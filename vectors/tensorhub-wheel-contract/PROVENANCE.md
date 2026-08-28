# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #74's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact EndpointEnvironmentSpec/2, EndpointBundle/2, and ResolvedWheelSet/3 bytes through
`endpoint_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,930-byte WheelhouseManifest/3 payload.

- Producing Tensorhub commit: `bce0a399e3c9c95d51ee80aed7ad2fdcb3c94776`

| file | SHA-256 |
|---|---|
| `endpoint-bundle.json` | `7ebe5a7100bf3aaaab0c9245fe8684ce14b62875ed67a7e3023f854f747f7771` |
| `endpoint-environment-spec.json` | `fadd98f988d65e667d4a70191460748f5412897cddc18d3516417953449345bc` |
| `fixture-index.json` | `54d3fd2b1986520616e1accd284e8056c1ebc616c3f7b7298793bb0eab0f8b98` |
| `project-wheel.json` | `e374821cce113ced053fe870c067fb2252ad32b649b5539ebb4847bf639aed22` |
| `resolved-wheel-set.json` | `0ffa18cd2514c0f0fffd753ce5bbb785cb5417f1b73b02df4b90088289c1d162` |
| `wheels/proof_endpoint-1.0.0-py3-none-any.whl` | `a1616ae8fb7c8bd28972f586b67f23f7445c15fd6ba54dfd7f67621b206f9cde` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
