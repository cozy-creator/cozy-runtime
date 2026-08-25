"""The author-facing surface — the ONLY cozy-runtime package endpoint/job code imports.

Structural fence (cozy-runtime.md §7, boundaries.md): this package may import schema and
typing libraries only. It cannot import cozy_runtime.internal, cozy_runtime.cli, the
worker protocol, TensorFS, Varena or any coordinator; it cannot read environment or config;
it names no device, allocator, compile or ingest surface. `scripts/fence.py` proves it.

cr-002a owns the real author core; this is the import-clean minimum `cozy-runtime new`
scaffolds against.
"""

from cozy_runtime.author._app import App, Registration, RegistrationError

__all__ = ["App", "Registration", "RegistrationError"]
