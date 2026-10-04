# Managed application JSON precision

Ordinary machine-v1 parent/child CPU proof confirmed that valid seed 9007199254740993
is refused by `author._calls` before managed-child dispatch, because authored arguments
use the PackageInterface JCS writer. The same decoder rejects valid child results and
its integer/float normalization can change a typed union branch.

Use a separate application JSON entry point for typed managed-call payloads and results,
with exact integer values, finite IEEE float values and integer/float kind. Share bounded
strict JSON structural parsing/string rendering with the existing canonical implementation;
keep the protocol JCS public API and profile unchanged. Model artifacts, receipts, capture
and interface documents continue to use their own protocol profile.

Validate exact 64-bit seeds, number kind, ordered inputs, semantic formatting reattachment
and true-intent conflict through ordinary CLI/API typed parent/child jobs with coherent Rust
machine and executor-plane SDK wheels. Unit checks alone do not qualify that consumer path.
