First-party memoized helpers need to bind named data and native build contracts, and Eval's synchronous quality entrypoint had no way to declare those dependencies. `App.entrypoint` now accepts the same optional memo version/dependencies as managed jobs. `MemoResource(package, name)` binds only that named resource's bytes; `MemoDistribution(name)` binds cheap installed version/wheel-tag/build provenance from the callee environment.

No caller/project inventory, RECORD, or native binary is scanned. Installation URLs and paths are excluded. Missing resource or required immutable metadata disables reuse without rejecting execution. Normal versioned wheels remain usable when uv omits optional archive hashes; editable/VCS-unpinned sources cannot claim immutable native identity.

Focused tests exercise real job and serving declarations: caller/install relocation reuses; helper/resource/native build changes invalidate; unsupported editable provenance disables reuse while valid execution continues. Existing automatic-identity/interface/static tests are included. Runtime>=0.18.30 is the API floor coordinated with prefetch PR647. No live installation, daemon, rental, native build or full local CI.

Eval and SDXL/H3 declaration migration and ordinary CLI/local/private recovery qualification are separate remaining gates (tracker235/236).
