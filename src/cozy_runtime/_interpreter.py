"""The one interpreter seat supported by this Runtime build."""

import sys
import sysconfig

# Requires-Python cannot distinguish standard CPython from the separate free-threaded ABI.
# Runtime and its native dependencies ship for one interpreter seat: ordinary cp314.
if (
    sys.implementation.name != "cpython"
    or sys.version_info[:2] != (3, 14)
    or sysconfig.get_config_var("Py_GIL_DISABLED") == 1
):
    raise RuntimeError("cozy-runtime requires standard CPython 3.14 (cp314, not cp314t)")
