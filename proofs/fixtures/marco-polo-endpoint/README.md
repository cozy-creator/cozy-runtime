# Marco/Polo endpoint fixture

This is a real, weightless endpoint project for Tensorhub source-publication and dynamic
worker-placement proofs. Its project wheel contains only endpoint code: the base worker image
provides Cozy Runtime and `msgspec`.

The sole public entrypoint has this closed contract:

```json
{"message":"marco"}
```

returns:

```json
{"message":"polo"}
```

Build and exercise it from the Cozy Runtime repository:

```sh
uv build --wheel --project proofs/fixtures/marco-polo-endpoint --out-dir dist/marco-polo
uv run --with dist/marco-polo/marco_polo_endpoint-1.0.0-py3-none-any.whl \
  cozy-runtime --dir proofs/fixtures/marco-polo-endpoint describe --check
uv run --with dist/marco-polo/marco_polo_endpoint-1.0.0-py3-none-any.whl \
  cozy-runtime --dir proofs/fixtures/marco-polo-endpoint --json run marco message=marco
```

`uv.lock`, `endpoint.descriptor.json`, and the deterministic wheel are the exact inputs a
Tensorhub publication proof should upload. The current remote contract admits one placement per
worker, so publishing this fixture under two endpoint repositories can prove sequential same-pod
replacement with `cozy rent revise`; it does not prove simultaneous stacking. The currently
qualified RunPod base worker image is CUDA/GPU even though this endpoint itself performs no GPU
work.
