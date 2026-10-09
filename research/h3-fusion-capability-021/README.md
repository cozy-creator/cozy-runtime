# Current-SDK H3 fusion gate

This is the proven five-case gate from ordinary CLI run5011, with dependency bounds
updated for Runtime0.21.6+ and TensorFS0.6.1+. Its numerical case construction is
unchanged: BF16, encoded FP8, and inactive/baked/unbaked Turbo hooks at the real
5376/14336 fusion widths. The integer SM compile argument and encoded leaf's
required `encoding` argument are explicit.

Run only after the benchmark owner installs the reviewed integrated Runtime:

```sh
cozy run ./research/h3-fusion-capability-021/verify --rental=kojirou \
  --tensorhub=http://127.0.0.1:8819 --json
```

All five constructors and compatibility checks pass on CPU with current Runtime
source, actual Diffusers0.41, and TensorFS0.6.1. Ordinary CLI `--describe --json`
also succeeds. These are preflight checks, not a new GPU arithmetic result.
The result records actual Runtime/Diffusers versions, applied substitutions, hook
and parameter custody, finite/repeat checks, and numerical error for each case.
