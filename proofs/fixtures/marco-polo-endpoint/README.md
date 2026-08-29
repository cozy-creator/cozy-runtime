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

Publish it through the real Creator product path:

```sh
cozy endpoint publish cozy/marco-polo --create --release acceptance-1 \
  --dir proofs/fixtures/marco-polo-endpoint \
  --profile torch2.13.0-cu130-cp312-linux-x86 \
  --reason "Marco/Polo acceptance"
```

Creator's fixed pure-Python packer produces the deterministic `py3-none-any` project wheel; this
fixture intentionally declares no PEP 517 backend. `uv.lock`, `endpoint.descriptor.json`, and that
wheel are the exact publication inputs. The current remote contract admits one placement per
worker, so publishing this fixture under two endpoint repositories can prove sequential same-pod
replacement with `cozy rent revise`; it does not prove simultaneous stacking. The approved RunPod
profiles use an RTX 4090 even though this endpoint itself performs no GPU work;
`endpoint.release.json` records that placement compatibility and declares no model closure.
