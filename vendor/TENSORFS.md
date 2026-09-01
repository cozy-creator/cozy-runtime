# TensorFS 0.0.8 wheel provenance

`tensorfs-0.0.8-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
is the temporary exact source for Runtime's `model-execution` extra until TensorFS is published on
PyPI.

- repository: `https://github.com/cozy-creator/tensorfs-v2.git`
- artifact source commit: `041d688a7d6feb0d7014c3501c901d95a52d31bb`
- merged source commit: `fb61a676d9def0822416e69de2f2d075931cf48b`
- tree: `f32a028e6c718a36319d3249649cdaea5efecda1`
- GitHub Actions run: `33477004679`
- job: `99758254560` (`wheels (ubuntu-latest, x86_64, true)`)
- artifact: `9788569039` (`wheel-ubuntu-latest-x86_64`)
- length: `2359967`
- SHA-256: `9f2591e90a362d705f679e3bc287063975116ad273db26a667e317146bf6f6f2`

The TensorFS workflow built the wheel with its native x86_64 matrix entry:

```sh
maturin build --release --out dist
```

That same job installed this exact artifact on CPython 3.12 and passed TensorFS's real named-tensor
wheel smoke test before upload.
