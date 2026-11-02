# Negotiated Runtime Start with a shared native owner

Owner: `/root/p3_owner_restart`, branch `codex/xs138-p4-shared-python`.
Base is fetched Runtime cb6c66bf3 plus coordinator61fe3233. The paired Creator
successor integrates native3df797f and changes only automatic Python acquisition
to the shared native service, with owner/client/closed facts.

Generic `CooperationOwner` construction inherits an existing service's settings.
Runtime's wire Config instead expresses the negotiated device Start contract.
Runtime therefore reads back the actual row, verifies its process/identity and
sharing/priority, and fails before publishing Memory or executing package code if
they disagree. It closes only its new lease, never configures another consumer's
service implicitly, and preserves the original failure when cleanup also fails.

Memory.attach_cooperation still requires a fresh, inactive CUDA Memory and fixed
Start intent. `_open_memory` already stages Memory until attachment succeeds; that
partial-initialization safeguard stays intact. Successful same-config adoption
does not need a new row. Request boundaries remain per-client native lifecycle;
failed construction submits no successful boundary or H measurement.

The staged Creator `owner-runtime/shared_python_clients` fixture exercises ordinary
Runtime nonfinal GPU progress and a real child-native service for successful
same-config adoption, mismatch refusal, capture-excluded final close and retry.
No GPU result or release is claimed from this source change. Public P4/ABI, PyArena
resource retirement and old-format removal are separate owners/deltas.
