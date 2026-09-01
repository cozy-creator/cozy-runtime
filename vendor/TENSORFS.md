# TensorFS 0.0.6 wheel provenance

`tensorfs-0.0.6-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
is the temporary exact source for Runtime's `model-execution` extra until TensorFS is published on
PyPI.

- repository: `https://github.com/cozy-creator/tensorfs-v2.git`
- commit: `ba40d92bc68e33f10c92848733d407ab887d11c7`
- tree: `00e762f61c2e0bc01339b53b0d3ee17d8cae2567`
- build image: `ghcr.io/pyo3/maturin@sha256:0d1711efd2d3aab276f738bdcda48d378bceef2eeb7f8a8caf59451defddee72`
- `SOURCE_DATE_EPOCH`: `946684800`
- length: `2347973`
- SHA-256: `83bbf80e3279d1f5ced912731a2a259ad557c483b8cb0123fc4795c2383d8b1f`

The build is the Tensorhub base-worker-image recipe:

```sh
maturin build --release --locked --out /output \
  --interpreter /opt/python/cp312-cp312/bin/python \
  --manifest-path /build/tensorfs/crates/tensorfs-py/Cargo.toml
```

The source is a `git archive` of the exact tree, mounted read-only at `/build/tensorfs` in the
digest-pinned image. Two clean builds must produce the recorded bytes before this wheel changes.
