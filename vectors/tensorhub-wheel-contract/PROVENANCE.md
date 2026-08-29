# Tensorhub WheelFact contract vectors

This complete fixture is copied byte-for-byte from Tensorhub PR #84's Go-authored
`vectors/wheel-contract/` output. Runtime neither recreates its JSON shape nor reserializes it:
the `base-ownership-vectors` architecture fence verifies the fixed SHA-256 identities, sends the
exact EndpointEnvironmentSpec/2, EndpointBundle/2, and ResolvedWheelSet/3 bytes through
`endpoint_overlay_input` with the exact foreign wheel files, then materializes those wheel grants
against Tensorhub's exact 23,930-byte WheelhouseManifest/3 payload.

- Producing Tensorhub commit: `b5216c4109cd4b28d4eb9ecb7a7742182b410d8b`

| file | SHA-256 |
|---|---|
| `endpoint-bundle.json` | `1790343eb9007e3aa79c0ed7ec602ac48a395cd7d7d04db927c80b7b81095b17` |
| `endpoint-environment-spec.json` | `c79110115207896b599ee44649cbfc3141edfc6227de5ab88a0e5310402495b0` |
| `fixture-index.json` | `4bd060fae82d18da98ba5a97c9ad6c103f71bd74c432df0cbbc58d112ae84179` |
| `project-wheel.json` | `e374821cce113ced053fe870c067fb2252ad32b649b5539ebb4847bf639aed22` |
| `resolution-lock.json` | `07092a4d5a2896a9a084398477b728b4a094733522e61180d331197e4a422cd1` |
| `resolved-wheel-set.json` | `072b784337e59f9e551c467ebf1ea8da5a737addd276470aca6aea577769d03c` |
| `wheels/proof_endpoint-1.0.0-py3-none-any.whl` | `a1616ae8fb7c8bd28972f586b67f23f7445c15fd6ba54dfd7f67621b206f9cde` |
| `wheels/proof_custom-1.0.0-py3-none-any.whl` | `77258c88f88aae13a7aa5c7ee6f44a8ce2fe5187de14a90dd964cd5991e001d2` |
