# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #74's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact EndpointEnvironmentSpec/1, EndpointBundle/2, and ResolvedWheelSet/2 bytes through
`endpoint_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,687-byte WheelhouseManifest/2 payload.

- Producing Tensorhub commit: `3d40b38a19d8e3ede6c44c11b6208ba686afd390`

| file | SHA-256 |
|---|---|
| `endpoint-bundle.json` | `f8844101d181e4dee7e483c87e37f349b140121f2d9d3b0d43c4c36d7d0f1f5f` |
| `endpoint-environment-spec.json` | `f3e0c5394b030dbda2645813252f5063f1f6023a170574576f5a798e7c9d2988` |
| `fixture-index.json` | `13a543eb8530559d4e7aa539e48cb8cf86feed4698c473362d12808b1a50906e` |
| `project-wheel.json` | `e374821cce113ced053fe870c067fb2252ad32b649b5539ebb4847bf639aed22` |
| `resolved-wheel-set.json` | `b9c9c9e31ed22390e9ed15df2c206bea86184e205aaf22ef23d9cdc3d78240ac` |
| `wheels/proof_endpoint-1.0.0-py3-none-any.whl` | `a1616ae8fb7c8bd28972f586b67f23f7445c15fd6ba54dfd7f67621b206f9cde` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
