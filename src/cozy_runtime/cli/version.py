"""`cozy-runtime version` — installed distribution and embedded build provenance."""

from __future__ import annotations

from cozy_runtime import PYTHON_CONTRACT_VERSION, WIRE_PROTOCOL_VERSION, __version__
from cozy_runtime._build_provenance import COMMIT
from cozy_runtime.cli.io import Result
from cozy_runtime.internal.config import RuntimeConfig


def run(config: RuntimeConfig, full: bool = False, fields: tuple[str, ...] = ()) -> Result:
    rows: tuple[tuple[str, str], ...] = (
        ("distribution", __version__),
        ("tag", f"v{__version__}"),
        ("commit", COMMIT[:12] if COMMIT != "unknown" else COMMIT),
        ("wire_protocol", WIRE_PROTOCOL_VERSION),
        ("python_contract", PYTHON_CONTRACT_VERSION),
    )
    if full:
        present = config.credentials.present()
        rows += (
            ("cozy_home", str(config.cozy_home)),
            ("credentials", ", ".join(present) if present else "0 present"),
        )
    return Result(
        rows=rows,
        next=("cozy-runtime new <name> — scaffold a package",),
        document={
            **{key: value for key, value in rows if not fields or key in fields},
            **(
                {"supports_guarded_restart": True}
                if not fields or "supports_guarded_restart" in fields
                else {}
            ),
            "next": ["cozy-runtime new <name> — scaffold a package"],
        },
    )
