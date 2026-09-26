Eval declares serving helper/resource/native memo dependencies, but the static reader rejected the two new App.entrypoint options before publication. Accept memo_version and memo_dependencies as installed-only declarations without executing package code or hashing local native metadata during static client inspection.

The existing builtin equality test also compared installed operation identities with the static contract, which deliberately omits environment-dependent identity. Require every actual builtin identity, then compare the entire remaining callable contract. The same assertion failed on the pre-change baseline.

Validation: five focused CPU controls pass: real TensorFS/NumPy/msgspec version invalidation, builtin static/actual contract equality, and a serving static declaration whose module/helper must never execute. Targeted Ruff and diff checks pass. No full local CI or inference was run.
