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
TLS_CERTIFICATE: Final = Path("/run/cozy/bootstrap/tls.crt")
TLS_PRIVATE_KEY: Final = Path("/run/cozy/bootstrap/tls.key")
WORKER_CONTROL_ADDRESS: Final = Path("/run/cozy/worker/control.addr")
WORKER_STATE: Final = Path("/run/cozy/worker")
#: The Runtime's DURABLE attempt record (th-142 child 2). Under /var/lib and not /run: an
#: obligation that a restart erases is not a record. The supervisor creates the root and hands
#: it over; nothing but the Runtime reads or writes inside it.
WORKER_RECORD_ROOT: Final = Path("/var/lib/cozy/worker")
