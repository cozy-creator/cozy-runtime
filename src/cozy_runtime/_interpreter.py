"""The one interpreter seat supported by this Runtime build."""

import sys
import sysconfig

# Requires-Python cannot distinguish CPython from another interpreter implementation.
# Runtime and its native dependencies ship for one interpreter seat: ordinary CPython 3.12.
if (
    sys.implementation.name != "cpython"
    or sys.version_info[:2] != (3, 12)
    or sysconfig.get_config_var("Py_GIL_DISABLED") == 1
):
    raise RuntimeError("cozy-runtime requires standard CPython 3.12 (cp312)")
