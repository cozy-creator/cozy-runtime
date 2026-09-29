Machine updates retain admission ownership through candidate readiness, application
handoff and cleanup. A Runtime exit racing a completed installation cannot trigger
rollback of the committed journal. Failed receipt writes preserve their signing key
for repair, and wheel uploads leave status observation available.

Local and rented machines use the same guarded update transaction and independent
agent. Explicit Hub scope reaches preparation and execution; credential reconciliation
preserves the scope already accepted by the execution journal. Runtime source archives
include the Go agent source while ordinary Python builds remain toolchain-free.
