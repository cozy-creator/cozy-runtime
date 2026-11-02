# Scoped Runtime validation

Ruff: baseline0, candidate0 after wrapping the new condition.
Mypy: baseline2, candidate2; no new or resolved findings. The environment's installed
TensorFS typing does not export varena, causing the shared attr-defined/no-any-return
findings. This is recorded, not treated as a clean full-project typecheck.

No mock CUDA or fabricated service test was added. Real same-config adoption and
fail-closed mismatch are staged in the paired Creator GPU fixture and remain unrun.
