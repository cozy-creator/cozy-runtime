# Python type checking

Run `sh scripts/typecheck.sh` from this repository. It uses the strict mypy
configuration in pyproject.toml and the repository's .venv interpreter. Set
`COZY_TYPECHECK_PYTHON=/absolute/path/to/python` to check against another prepared
environment, such as the exact worker qualification cohort. Install the typed
first-party dependencies first; a missing Runtime/TensorFS/cozy-eval contract is
an error, not permission to ignore imports.

Use the same environment in BasedPyright (or Pyright); pyrightconfig.json sets
strict checking, .venv selection, and source/stub search paths. Strict editor diagnostics
provide early feedback; the configured mypy command remains the release gate.
Optional command arguments select focused files during development, but do not
replace the full configured check before publication. On the shared developer
machine, use CI or an existing remote worker for broad checks.

New functions require annotations. Use typed JSON envelopes, validated object
narrowing, native stubs, and small Protocols for dynamic integration boundaries.
Do not add blanket ignores, Any casts, or a baseline suppression file to obtain a
green result. Existing exceptions and outstanding errors are recorded in
[the typing rollout](https://github.com/cozy-creator/tracker/blob/master/tracker/cross-cutting/xs-034-strict-python-typing.md).

Adding this configuration does not establish that the entire tree is green.
Record the source commit, interpreter/checker versions, exact command and complete
result before claiming qualification. Generated and attributed upstream code may
have documented exclusions; first-party callers must consume their current stubs.
