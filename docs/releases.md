# Publishing Runtime

Creating an immutable `vX.Y.Z` tag does not start publication. Explicitly select one
entrypoint in `publish.yaml`; both use the same per-tag concurrency group and PyPI
trusted-publishing job. The tag must match a clean source checkout and its declared
version. Actual publication refuses versions already present on PyPI, including partial
uploads. Never move a published tag or fill an existing version from different source.

## Build from source

```sh
gh workflow run publish.yaml -f release_tag=vX.Y.Z -f mode=source
```

The hosted job builds the source and pure wheel, and adds each platform's kernel modules to make
the native wheels (Linux x86_64 and aarch64, Windows x86_64): `kernels.yaml` reuses cached modules
when `kernels/` is unchanged, and compiles otherwise. No wheel carries a machine: TensorD ships in
the `cozy` binary. An unreleased TensorFS is a git source on the public cozy-creator repository.

A local authenticated source build followed by the prebuilt entrypoint is also supported.
This requires neither a new deploy key nor copying a personal token into the repository.
Use the tracked build/bundle scripts and record the files proven by the release checks.

## Publish approved prebuilt files

Attach the tested distributions to the GitHub release for the immutable tag, record each
SHA-256, then select those exact files:

```sh
gh workflow run publish.yaml -f release_tag=vX.Y.Z -f mode=prebuilt \
  -f gated_sha256sums='<sha256> <filename> <sha256> <filename> ...'
```

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

## Qualification tables

The decode qualification ships as data (`internal/qualification_tables`), one table per card
architecture and torch minor, valid while `IDENTITY_SOURCES` are unchanged. When a release
changes them, measure every gate card's table on the candidate before tagging:

```sh
python -m cozy_runtime.cli.qualify src/cozy_runtime/internal/qualification_tables
```

A card without a current table measures what it is asked once per machine and keeps it.
