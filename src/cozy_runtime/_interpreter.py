"""The standard CPython rolling window supported by this Runtime build."""

import json
import sys
import sysconfig
from pathlib import Path


def supported_minors() -> tuple[str, ...]:
    policy = json.loads(Path(__file__).with_name("python-window.json").read_text())
    return tuple(policy["minors"])


# Requires-Python cannot distinguish implementation or free-threaded ABI. The packaged
# policy is also consumed by worker admission, local Creator selection and image builds.
if (
    sys.implementation.name != "cpython"
    or f"{sys.version_info.major}.{sys.version_info.minor}" not in supported_minors()
    or sysconfig.get_config_var("Py_GIL_DISABLED") == 1
):
    raise RuntimeError(
        "cozy-runtime requires standard CPython in the supported window: "
        + ", ".join(supported_minors())
    )
