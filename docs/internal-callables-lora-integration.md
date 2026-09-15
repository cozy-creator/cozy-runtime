# Qualified LoRA and internal callable integration

This candidate combines released Runtime 0.18.3 with the existing f084c910
request-scoped FP32 LoRA implementation. It retains Runtime-owned execution and
internal callable admission, current attention and device initialization fixes,
and the existing adapter engine. It does not adopt the separate PEFT/TorchAO
replacement.

Generated wire 57 bindings come from worker-protocol b6e56391, preserving both
ordered adapters and the current native execution/install-reuse contract.

This source candidate needs focused native and ordinary Creator CLI qualification
before shared workstation deployment. It is not a new public Runtime release.
