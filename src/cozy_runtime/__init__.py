"""cozy-runtime: author surface, worker, disposable device executor."""

# worker-protocol's published identity: the proto package, the additive-train minor, and —
# the only one of the three that fences anything — the SCHEMA DIGEST (#530-A1). All three
# are READS of the generated `wire_identity.py`, never constants beside it that can drift
# from it, which is the whole lesson of a `WIRE_MINOR` that agreed while the bytes did not.
from cozy_runtime.protocol import SCHEMA_DIGEST, WIRE_MINOR, WIRE_SCHEMA_REV

__version__ = "0.0.3"

# The two independent semvers reported by `cozy-runtime version`.
PYTHON_CONTRACT_VERSION = "0.0.2"
WIRE_PROTOCOL_PACKAGE = "cozy.worker.v1"
WIRE_PROTOCOL_VERSION = (
    f"{WIRE_PROTOCOL_PACKAGE}+rev.{WIRE_SCHEMA_REV}+minor.{WIRE_MINOR} {SCHEMA_DIGEST}"
)
