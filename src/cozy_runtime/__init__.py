"""cozy-runtime: author surface, supervisor, disposable device executor."""

# worker-protocol's published identity (th-024, master 5b07b79): the proto package plus its
# WIRE_MINOR file. cr-007 wired the generated bindings in, so the minor is now a READ of the
# vendored copy rather than a second constant beside it that can drift from it.
from cozy_runtime.protocol import WIRE_MINOR

__version__ = "0.0.1"

# The two independent semvers reported by `cozy-runtime version`.
PYTHON_CONTRACT_VERSION = "0.0.1"
WIRE_PROTOCOL_PACKAGE = "cozy.worker.v1"
WIRE_PROTOCOL_VERSION = f"{WIRE_PROTOCOL_PACKAGE}+minor.{WIRE_MINOR}"
