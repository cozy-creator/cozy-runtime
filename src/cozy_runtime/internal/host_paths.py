"""Fixed worker filesystem paths."""

from __future__ import annotations

from pathlib import Path
from typing import Final

TRANSFER_STAGING: Final = Path("/var/lib/cozy/tmp/transfers")
TENSORFS_ROOT: Final = Path("/var/lib/tensorfs")
BOOT_DATA: Final = Path("/var/lib/cozy/boot")
BOOT_ID: Final = Path("/run/cozy/bootstrap/pod-boot-id")
INSTALL_ROOT: Final = Path("/var/lib/cozy/installs")
MEDIA_ROOT: Final = Path("/var/lib/cozy/media")
READINESS_PAYLOAD: Final = Path("/run/cozy/bootstrap/readiness-payload")
#: What this worker is doing, rewritten every reporter tick for the pod supervisor's release
#: decision (`worker/activity.py`). It lives in the BOOTSTRAP directory and not beside
#: `control.addr`, because that is where the supervisor->worker handoff already is: the
#: supervisor creates this directory and owns it after its uid drop, while `/run/cozy/worker`
#: is created by this root worker at 0700 and the supervisor cannot even traverse it.
WORKER_ACTIVITY: Final = Path("/run/cozy/bootstrap/worker-activity")
TLS_CERTIFICATE: Final = Path("/run/cozy/bootstrap/tls.crt")
TLS_PRIVATE_KEY: Final = Path("/run/cozy/bootstrap/tls.key")
WORKER_CONTROL_ADDRESS: Final = Path("/run/cozy/worker/control.addr")
WORKER_STATE: Final = Path("/run/cozy/worker")
