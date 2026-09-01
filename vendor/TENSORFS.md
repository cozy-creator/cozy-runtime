# TensorFS 0.0.7 wheel provenance

`tensorfs-0.0.7-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
is the temporary exact source for Runtime's `model-execution` extra until TensorFS is published on
PyPI.

- repository: `https://github.com/cozy-creator/tensorfs-v2.git`
- commit: `a9cfd601f3f8c0841907e2c2625021b463f0ef36`
- tree: `3c04d60b20f478d91801ca37278b5d23d1cd7237`
- GitHub Actions run: `33462103932`
- job: `99714229997` (`wheels (ubuntu-latest, x86_64, true)`)
- artifact: `9783565076` (`wheel-ubuntu-latest-x86_64`)
- length: `2360087`
- SHA-256: `a3296ef604061ce2984fdb74e329cf5e2daa69d7b45b924d5806b53ab9dc2521`

The TensorFS workflow built the wheel with its native x86_64 matrix entry:

```sh
maturin build --release --out dist
```

That same job installed this exact artifact on CPython 3.12 and passed TensorFS's real named-tensor
wheel smoke test before upload.
