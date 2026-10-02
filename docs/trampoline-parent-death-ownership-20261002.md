# Trampoline keeps the parent-death signal across the credential drop

Linux clears `PR_SET_PDEATHSIG` when a process changes uid/gid. The trampoline armed it
before `setgid`/`setuid`, so every executor launched under a package identity lost its
SIGKILL-on-parent-death. It now arms again after every credential transition and re-checks
the expected parent before exec.

## Qualification (2026-10-02, CPU only, root `python:3.12-slim` container)

`tests/test_trampoline_parent_death.py`, Runtime wheel from this branch:

| TensorFS | Result |
|---|---|
| released 0.3.93 | 7 passed, 1 xfail (native Store-owner fscredentials clear the guard) |
| master `32291f4` (tensorfs#308) | 8 passed |
| unfixed master trampoline, same TensorFS | 2 failed (`guard 0 != 9`, foreign credentials) |

Nonroot: the root-only cases skip; `test_device_lanes.py` passes.

## Limits

Linux binds PDEATHSIG to the creating *thread*: launchers must spawn from a persistent
thread (the worker's `executor-launcher`, the machine's `ChildLauncher`). A parent outside
the PID namespace reads as 0 and cannot be re-checked.
