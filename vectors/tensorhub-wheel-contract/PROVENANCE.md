# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #74's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact EndpointEnvironmentSpec/1, EndpointBundle/2, and ResolvedWheelSet/2 bytes through
`endpoint_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,687-byte WheelhouseManifest/2 payload.

- Producing Tensorhub commit: `36f5eb95961802fd5c90090e23cbf44938341da5`

| file | SHA-256 |
|---|---|
| `endpoint-bundle.json` | `e343ba95b5a078ef1703cf20fc452d1524f04310174d9e076834775a15f3e5cd` |
| `endpoint-environment-spec.json` | `4508f6a4505a592517beefefaf38b174abfa8f54c16070b8a2bd4b77903d7135` |
| `fixture-index.json` | `2ae701c00bd988f8c7e052cd16cd4b287b5ef58efdbb1c7487d98465b78cf666` |
| `project-wheel.json` | `e374821cce113ced053fe870c067fb2252ad32b649b5539ebb4847bf639aed22` |
| `resolved-wheel-set.json` | `789a7c76d304423f658effa0dd46ff1a83109b1ae2fab6b8b870302720665881` |
| `wheels/proof_endpoint-1.0.0-py3-none-any.whl` | `a1616ae8fb7c8bd28972f586b67f23f7445c15fd6ba54dfd7f67621b206f9cde` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
