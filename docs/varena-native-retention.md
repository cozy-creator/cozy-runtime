# Native demand retention cutover

This branch removes the Runtime cache-window and retained-prefix byte decisions.
It is independent of the preserved ownership-guard experiment (#1992/#2374).
Clearing a streaming window only removes early eviction: native process/device
limits, live/escaped pins, loading state, completion fences, capture protection and
graph pools remain authoritative. Existing graph/cache-owner retirement remains.

The intended change removes all three window writes and the prefix/replanning
arithmetic together. Runtime still declares component phases, source/binding
identity and explicit graph/hold ownership. Native demand chooses reclaim when
actual capacity requires it. Host cache eligibility must follow actual pages and
read history, not a Python retained prefix. Generic split advice and P3 room
compatibility remain named separate migrations; they must not reconstruct a
second retained-weight allowance.

The paired native API is `Arena.clear_streaming_limit()` (Creator #2372). It
changes only native intent metadata, does not reclaim or call the driver, and
preserves other capacity limits. No Python maximum integer is supplied.

Source and GPU validation remain pending at this initial draft checkpoint. The
c05 prefetch-only full H3 arm failed performance (+4.56% denoise/+5.29% recorded
weight-source traffic); no fallback or threshold relaxation is proposed. Actual
cold/warm model reads, two capacities, aliases, failed fills and full H3/SDXL must
qualify the combined retention change. Nothing is merged or released.
