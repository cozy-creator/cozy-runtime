# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #74's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact EndpointEnvironmentSpec/1, EndpointBundle/2, and ResolvedWheelSet/2 bytes through
`endpoint_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,742-byte WheelhouseManifest/2 payload.

- Producing Tensorhub commit: `bce0a399e3c9c95d51ee80aed7ad2fdcb3c94776`

| file | SHA-256 |
|---|---|
| `endpoint-bundle.json` | `277d12352b294c64516d3b903b148a037976a849ea06178d021308e725ab9569` |
| `endpoint-environment-spec.json` | `bff844fc438f21327649216018494efe7cc0ccc44e586deb2fb34c9759675b0d` |
| `fixture-index.json` | `7ff32278510377c8d579706d3dc2580791e3ea15620af9f5db87af3cfa5127bb` |
| `project-wheel.json` | `e374821cce113ced053fe870c067fb2252ad32b649b5539ebb4847bf639aed22` |
| `resolved-wheel-set.json` | `4d0be76e57338749a2f2403cecf452081bd09da50a7024549e919baf21580e18` |
| `wheels/proof_endpoint-1.0.0-py3-none-any.whl` | `a1616ae8fb7c8bd28972f586b67f23f7445c15fd6ba54dfd7f67621b206f9cde` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
