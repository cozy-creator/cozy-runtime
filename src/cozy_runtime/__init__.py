"""cozy-runtime: author surface, supervisor, disposable device executor."""

__version__ = "0.0.1"

# The two independent semvers reported by `cozy-runtime version`.
PYTHON_CONTRACT_VERSION = "0.0.1"
# worker-protocol's published identity (th-024, master 5b07b79): the proto package plus
# its WIRE_MINOR file. ASSERTED, not linked: cr-007 wires the generated bindings in and
# this constant becomes a read of them rather than a copy of them.
WIRE_PROTOCOL_PACKAGE = "cozy.worker.v1"
WIRE_MINOR = 0
WIRE_PROTOCOL_VERSION = f"{WIRE_PROTOCOL_PACKAGE}+minor.{WIRE_MINOR}"
