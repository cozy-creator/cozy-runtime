# Publishing Runtime

Creating an immutable `vX.Y.Z` tag does not start publication. `runtime-publish.yaml` has two
entrypoints with one per-tag concurrency group: source mode builds a candidate users don't get,
and prebuilt mode publishes it to PyPI (trusted publishing names this workflow file) once a
recorded gate passed it. Release testing is manual, at the release agent's discretion (cozy-creator
`release-gate/`, tracker design ci-v2). The tag must match a clean source checkout and its declared
version. Actual publication refuses versions already present on PyPI, including partial
uploads. Never move a published tag or fill an existing version from different source.

## Candidate: build from source

```sh
gh workflow run runtime-publish.yaml -f release_tag=vX.Y.Z -f mode=source
```

The hosted job builds the source and pure wheel, and adds each platform's kernel modules to make
the native wheels (Linux x86_64 and aarch64, Windows x86_64): `kernels.yaml` reuses cached modules
when `kernels/` is unchanged, and compiles otherwise. No wheel carries a machine: TensorD ships in
the `cozy` binary. An unreleased TensorFS is a git source on the public cozy-creator repository.
The release set and SHA256SUMS land on the tag's GitHub prerelease, the candidate; source mode never
publishes. Its summary prints the gate's `runtime_wheel=URL#sha256` and the `gated_sha256sums`.

A local authenticated source build followed by the prebuilt entrypoint is also supported.
This requires neither a new deploy key nor copying a personal token into the repository.
Use the tracked build/bundle scripts and record the files proven by the release checks.

## Promote a gated candidate

Test the candidate (cozy-creator `release-gate/`, the suites the release calls for) and record the
result, then publish exactly the candidate's files:

```sh
gh workflow run runtime-publish.yaml -f release_tag=vX.Y.Z -f mode=prebuilt \
  -f gated_sha256sums='<sha256> <filename> <sha256> <filename> ...' \
  -f gate_run=<release-gate run id | URL of the recorded hand gate>
```

A run id is checked: the gate passed, and its `candidate.json` holds one of these files.

This path needs no source credentials. It verifies tag identity, downloads the release assets
and requires the three native wheels, pure wheel and sdist for that same verified version. It
refuses wrong-version, missing, extra or changed files.
It does not import or execute downloaded artifact contents. The publishing job rechecks the hashes
after artifact transfer, publishes with OIDC, and checks PyPI serves the same file set
and hashes. Source build checks still belong to source mode.

## Rehearse without publication

Add `-f rehearsal=true` to either command. Rehearsal permits an already published
version while retaining clean tag/source checks and the selected build or asset checks.
The publishing job is skipped, so it never requests PyPI OIDC or uploads distributions.
For a workflow change, add `--ref <branch>` to test that branch's workflow against an
existing tag. A source rehearsal still needs access to private source dependencies;
missing access should fail at the early preflight. Do not create dummy public releases
or retry an already published version to test automation.

See [tracker #348](https://github.com/cozy-creator/tracker/issues/348) for the source/prebuilt
entrypoint repair and its rehearsal evidence.

## Kernel conformance

Kernel selection is code rules by tensor-core family (xs-065); no release ships measured data.
Before a rule names a new family's kernels, run the conformance suite by hand on a real card of
that family; it prints a report and exits 1 on any failure:

```sh
python -m cozy_runtime.cli.conformance
```
