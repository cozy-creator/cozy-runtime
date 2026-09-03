"""cozy-runtime: author surface, worker, disposable device executor."""

from importlib.metadata import version as _distribution_version

# Worker-protocol's runtime identity is the protobuf package major plus its additive minor.
# Exact generated-byte provenance is enforced by CI's vendored-drift check.
from cozy_runtime import _interpreter as _interpreter
from cozy_runtime.protocol import WIRE_MINOR

del _interpreter

__version__ = _distribution_version("cozy-runtime")

del _distribution_version

# The two independent semvers reported by `cozy-runtime version`.
PYTHON_CONTRACT_VERSION = "0.0.2"
WIRE_PROTOCOL_PACKAGE = "cozy.worker.v1"
WIRE_PROTOCOL_VERSION = f"{WIRE_PROTOCOL_PACKAGE}+minor.{WIRE_MINOR}"
