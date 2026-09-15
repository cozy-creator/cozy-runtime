# Package dependency isolation

Status: accepted design, implementation and end-user qualification pending (proto-053).
Owner: `/root/package_venv_design`; branch `docs/package-venv-isolation`;
base `a502bfc8df88076b3c462ed96298d14e017d227d`.

## Failure and cause

H3's diffusers requires `huggingface-hub>=1.23.0,<2.0`, while the image provides
1.16.1. `package_requirements.validate` detects this as
`package_environment_dependency_base_conflict`. That detection is useful; making
the image's version the package's only provider is the architectural defect.
The earlier pydantic-core patch conflict is the same class of error.

The old proto-017/cr-052 policy intentionally gave local and rental execution
different semantics. `base_owned_wheel` drops supplied protected wheels regardless
of version; `_verify_inventory` rejects a second protected family; both uv
installers expose system site packages. The locked path consumes an image-filtered
subset and validates its union with the whole base. The legacy `BootInstall` path
repairs one distribution with `--no-deps`, cannot solve its transitive closure,
and is disabled under rented storage admission. Updating the image only moves the
next failure; it does not repair isolation.

## Contract

Every published package, editable project and unpublished script executes in its
own ordinary uv environment and executor process, locally and remotely. Runtime
owns the same environment preparation/validation rules on both paths. Creator may
capture a previously solved closure, but must not invent a rental-only dependency
policy. Source identity and dependency identity remain separate so editing a
script reuses unchanged immutable dependency bytes.

Resolve the whole selected closure from pyproject and wheel Requires-Dist,
including extras, markers and transitive requirements. Authored dependencies are
compatibility ranges under the existing authoring policy. A machine-generated
exact lock and exact artifact hashes capture a resolution for replay; they are
not a demand that authors pin exact versions. A supplied complete lock is honored
as a unit, or rejected for a real contradiction. Never silently change one locked
row to repair another. A ranges-only input is resolved once before preparation
completes, then captured; no resolver runs per inference or on a valid warm open.

The venv must contain its complete selected dependency closure. No inherited
system-site directory, ambient PYTHONPATH, user site, arbitrary SDK directory or
working-directory import fallback can supply missing package dependencies.
Installer environment variables and user/global uv configuration cannot inject
constraints, indexes or paths outside the explicit request. Standard-library
shadowing checks remain; collision with a separate base package is not a reason
to reject a private dependency.

The image inventory is an optimization input. Compatible exact artifacts may be
reused through immutable cache links or copied into the private generation. A
version string alone does not establish artifact identity. uv's shared cache
owns download/unpack deduplication; reflink/hardlink reuse is an optimization with
copy fallback. Never link mutable SDK files into a retained environment. Cache
cleanup must not chmod shared file inodes. Missing or incompatible baked packages
are installed privately, including Torch/Triton and their CUDA dependencies.
Alternate accelerator stacks warn about cold-start time, disk and unavailable
optimizations; they are not categorically rejected for differing from the image.

A venv does not replace the host GPU driver, kernel, architecture or CPython ABI.
Wheel tags/Requires-Python and actual driver/runtime support still apply. A native
extension built for a different Torch ABI may need a compatible wheel or rebuild;
matching a Python requirement range is not proof of native compatibility. Report
the concrete unsupported wheel/platform/driver or failed load, and preserve a
working package environment when another package fails.

## Runtime control versus package SDK

The host worker's control Runtime remains in its own immutable environment. The
package executor needs the Runtime SDK/implementation and TensorFS dependencies
that its package closure actually selects. Their versions must meet the explicit
control/executor protocol and native interoperability contracts; a different
huggingface-hub, pydantic-core or Torch is not a protocol mismatch. Do not enforce
compatibility by importing the host SDK through a shared site directory. Initially
the supported executor Runtime can be installed/captured into each venv with its
own dependency closure. Broader Runtime version interoperability requires actual
protocol evidence, not a speculative SDK split or a universal version-equality gate.

## Minimal migration

1. Keep existing wheel custody, content identities, receipts, atomic staging,
   storage admission and executor launch. Bump the internal environment content
   identity when isolation semantics change so inherited generations cannot be
   mistaken for complete isolated generations. Preserve old active generations.
2. Stop pruning closure entries by image ownership in Creator/publication and
   Runtime locked preparation. Supply a complete exact closure (including the
   executor Runtime) or solve declared ranges once during preparation. Keep the
   bounded exact wheel-fetch path; reserve space before large downloads.
3. Create uv venvs without system site packages. Install the complete closure;
   `--no-deps` is valid only after a complete resolution has been captured and
   independently validated. Delete per-distribution BootInstall repair and broad
   protected-family admission rules. Validate against only the private closure.
4. Use the existing explicit environment interpreter for both local and remote
   dispatch. Preserve credential/uid/process boundaries. Record actual selected
   dependencies and degraded accelerator-stack warnings at preparation.
5. Retain a shared immutable uv cache with accounted storage. Warm receipt reuse
   performs no install, solve or image dependency hashing. Compatible image
   refreshes do not invalidate unchanged private dependency bytes.

## Development and release gates

Any environment/dependency change must prove the counterexample, not merely add
a refusal or refresh a global constraint file. Run real uv tests using a controlled
wheel index: two packages require incompatible direct and transitive versions;
extras and markers select additional dependencies; the base carries a conflicting
version and an undeclared sentinel import. Both packages import their own closure,
the sentinel is unavailable, and the base and sibling files stay unchanged.
Include the H3 huggingface-hub conflict and pydantic/pydantic-core coupling.

Exercise both wheel and locked-input preparation, missing transitive rows,
malicious ambient uv/Python settings, interrupted and concurrent preparation,
cache deletion with completed generations retained, metadata integrity, and warm
reuse with networking/install disabled. A supplied alternate Torch closure must
pass metadata admission; native CUDA execution is qualified separately on a
supported driver with observed imports, versions and device operation.

The product gate is ordinary installed `cozy` CLI execution for published and
editable/script packages locally and on an already-authorized private worker,
including cold preparation, warm reuse and coexistence. Report measured paths,
versions, cache reuse and unchanged base evidence. Unit/Runtime tests are
supplemental; a drafted design, merged code or green metadata test is not proof
of remote H3 inference. No new paid rental or active-rental restart is implied.

## uv implementation references

Use an explicit `--python` target, `--no-config`, an owned `--cache-dir`, and a
sanitized child environment; `--no-config` alone is not an environment-variable
firewall. In particular UV_CONSTRAINT, UV_OVERRIDE, UV_BUILD_CONSTRAINT, index
settings, PYTHONPATH and PYTHONHOME must not alter the request. Configuration
selection stays centralized in Runtime's existing config module. See uv's
[environment reference](https://docs.astral.sh/uv/reference/environment/) and
[explicit interpreter targeting](https://docs.astral.sh/uv/pip/environments/).

uv supports concurrent cache use and locks target installations. Treat its cache
as uv-owned: perform pruning through uv cache commands, never hand-edit a shared
cache. Preserve separate private staging/atomic publication because uv's install
lock is not a Cozy receipt transaction. See [uv cache safety](https://docs.astral.sh/uv/concepts/cache/).

Tracking: [epic #202](https://github.com/cozy-creator/tracker-v2/issues/202),
[Runtime #546](https://github.com/cozy-creator/cozy-runtime/issues/546),
[Creator #590](https://github.com/cozy-creator/cozy/issues/590),
[Hub #648](https://github.com/cozy-creator/tensorhub-v2/issues/648).
