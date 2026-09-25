Owner: /root/runtime_finish (continuing /root/runtime_lifecycle)
Purpose: Retain staged activation headroom and avoid restoring weights that the next request must evict.
Branch: fix/proto061-request-headroom-20260925
Base: bfdfba5e
PR: https://github.com/cozy-creator/cozy-runtime/pull/616
Scope: worker attempts, planning, residency arbitration, and focused regression tests.
GPU inference, CLI installations, and rental ownership remain with the root task.
