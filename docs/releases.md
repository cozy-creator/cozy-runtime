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

The hosted job builds the source and pure wheel, reuses a verified native donor when
the native inputs are unchanged (otherwise building the kernels), and bundles the
pinned TensorD binary. TensorD is public and needs no SSH deploy key. Its TensorFS
dependency is private, so the job checks source access before expensive build work.
Existing authenticated Git access works; the optional `TENSORFS_READ_KEY` repository
secret can supply a TensorFS read key. Without source access the job refuses with an
actionable message.

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

This path needs no TensorD or TensorFS source credentials. It verifies tag identity,
downloads the release assets and requires the native wheel, pure wheel and sdist for
that same verified version. It refuses wrong-version, missing, extra or changed files.
It does not import or execute downloaded artifact contents. The publishing job rechecks the hashes
after artifact transfer, publishes with OIDC, and checks PyPI serves the same file set
and hashes. Native donor pins and source build checks still belong to source mode.

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
