# Pending disk dtype demand

A parked tensor retains its real source/fill dtype. A requested root dtype conversion is
queued until refill; reporting only that meta tensor's current nbytes understates both the
settled destination and the temporary coexistence of old and new conversion storage.

`tiers.LoadDemand` separates settled `resident` bytes from load/conversion `peak` bytes.
`Unit.nbytes` continues to describe current backing. Canonical Torch Module dtype closures
are recognized by the current interpreter's actual code objects, without deployment or
source-hash gates. Their metadata projections retain integer buffers, chained conversions,
and no-op storage. No meta dtype is rewritten to advertise backing that does not exist.
Unknown transformations, custom owner `_apply`, gradient state, or alias-changing stored
conversions have no established allocation bound and refuse before a parked conversion is
queued. Ordinary device-only requests retain Runtime's existing placement behavior.

For one dtype-conversion stage, the bound includes all old backing plus every changed
slot's new destination. Unchanged slots are not charged again. Sequential unit loads use
max(previous settled additions + next unit peak); a whole store fill accounts for all
source units before replaying their conversions. Planner scope admission, resident/stream
plans, explicit restoration and host backing demand consume these distinct quantities.
Existing reserved allocator segments remain charged; this does not credit fragmented cache.
Driver/context limits and OOM recovery remain in force, including external allocation races.

The production `Planner.adopt` conversion adapter invalidates old resource observations
before mutation and after success. A local component identity plus conversion epoch keeps
new estimates from being reused by a sibling whose original kind/input dtype happens to
match. This is measurement identity only: it changes no authored dtype, guidance mode,
checkpoint identity, lease or deployment compatibility.

CPU/meta tests exercise the real installed adapter, settled/peak arithmetic for common and
streamed units, mixed integer buffers, chained and no-op casts, shared meta backing, failed
fill and failed post-fill conversion rollback, and cross-component estimate isolation.
They do not qualify GPU allocation traces, kernel behavior or full model inference. The
metadata-anchor change remains a separate reviewed integration; it must pass all common
partitions to the same sequence helper and validate its topology in the conversion callback.
