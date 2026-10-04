# Attributable context measurements

[Tracker #320](https://github.com/cozy-creator/tracker/issues/320) records the audit,
implementation and rental qualification. Device-wide memory deltas cannot identify
this executor's context when another process allocates concurrently.

At startup Runtime queries its own driver row on the sealed device after initial
library launches, subtracting torch's reservation. Later measurement uses the
plane's device UUID and subtracts its private/shared maps too. A hidden or unreadable
process row leaves the context unknown. Plane facts add `context_measurement`, with
`process_driver` identifying a value eligible to train machine context estimates.

The machine accepts older peers without this field, retains valid learned plan/shape
data, invalidates only old untrusted context cache rows, and persists each group's
rank-specific context on its own GPU. CPU migration and rank-selection tests cover
those contracts. Startup while another process changes memory, missing PID attribution,
low-memory completion and launch timing are required CUDA qualification gates.
