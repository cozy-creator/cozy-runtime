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

Publish it through Creator's endpoint-publish command, pointing `--dir` at this directory.
Creator's fixed pure-Python packer produces the deterministic `py3-none-any` project wheel; this
fixture intentionally declares no PEP 517 backend. `pyproject.toml`, `uv.lock`,
`endpoint.descriptor.json`, and that wheel are the exact publication inputs. There is no release
manifest and no author-selected worker profile: Tensorhub derives compatible base worker images
from the locked Python/package requirements. The current remote contract admits one placement per
worker, so publishing this fixture under two endpoint repositories can prove sequential same-pod
replacement with `cozy rent revise`; it does not prove simultaneous stacking.
