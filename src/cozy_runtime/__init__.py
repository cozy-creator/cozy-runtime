"""cozy-runtime: author surface and disposable device executor."""

from importlib.metadata import version as _distribution_version

from cozy_runtime import _interpreter as _interpreter

del _interpreter

__version__ = _distribution_version("cozy-runtime")

del _distribution_version

# The author contract reported by `cozy-runtime version`.
PYTHON_CONTRACT_VERSION = "0.0.2"
