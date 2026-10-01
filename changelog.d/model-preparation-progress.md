Model preparation exposes native per-checkpoint byte availability through the existing
machine description. The agent samples these optional observations every two seconds
and forwards them on the ordinary preparation event stream. Shared waiters and
reconnected observers see the same transfer; observer loss does not cancel native work.

Each optional read has its own two-second budget. Missing, malformed, older-peer, or
blocked telemetry leaves the preparation answer unchanged. Preparation counters stay
separate from execution timing and do not sum shared model objects as unique traffic.
