"""The vendored cozy.worker.v1 bindings are exactly the bytes SOURCE names.

Runtime does not own the worker protocol; ``src/cozy/worker/v1`` is a COPY of another
repository's generated output, and a copy is only a copy while somebody compares the
bytes. The failure this file exists to prevent is a consumer silently sitting on a stale
wire minor while the schema and its other consumers move on -- which has happened in this
workspace, and cost a whole lane to repair, because every guard that should have caught it
needed a credential CI does not always have.

So this needs no network, no secret and no checkout: it runs in every ``uv run pytest``.
The token-gated ``drift`` job in CI is the ceiling above it -- it regenerates from the
.proto with the pinned toolchain -- and this is the floor beneath.

Nothing here skips. A skip is indistinguishable from a pass, which is the whole defect.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from cozy_runtime.protocol import WIRE_MINOR

#: The vendored tree, and the manifest that pins it.
VENDOR_DIR = Path(__file__).resolve().parents[1] / "src" / "cozy" / "worker" / "v1"
SOURCE = VENDOR_DIR / "SOURCE"
WANT_REPOSITORY = "https://github.com/cozy-creator/worker-protocol"
#: What a re-vendor is, in one line, on every failure below.
REMEDY = (
    "re-vendor bindings and corpus together: uv run python scripts/vendor-worker-protocol.py "
    "<commit>"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def read_manifest() -> tuple[str, str, int, dict[str, str]]:
    """SOURCE parsed into its ``key=value`` lines, strictly.

    An unknown key is a typo, and a typo that parses is a digest nobody checks.
    """
    repository = commit = ""
    wire_minor = 0
    digests: dict[str, str] = {}
    assert SOURCE.is_file(), f"the vendored bindings have no SOURCE manifest at {SOURCE}"
    for number, line in enumerate(SOURCE.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        assert "=" in line, f"{SOURCE}:{number}: not a key=value line: {line!r}"
        key, value = line.split("=", 1)
        if key == "repository":
            repository = value
        elif key == "commit":
            commit = value
        elif key == "wire_minor":
            assert value.isdigit(), f"{SOURCE}:{number}: wire_minor {value!r} is not a number"
            wire_minor = int(value)
        elif key.endswith((".py", ".pyi")):
            assert value.startswith("sha256:") and len(value) == 71, (
                f"{SOURCE}:{number}: {key} digest is not a sha256: {value!r}"
            )
            digests[key] = value.removeprefix("sha256:").lower()
        else:
            raise AssertionError(f"{SOURCE}:{number}: unknown manifest key {key!r}")
    return repository, commit, wire_minor, digests


def test_source_manifest_is_exact() -> None:
    """Check the manifest itself before anything trusts it."""
    repository, commit, wire_minor, digests = read_manifest()
    assert repository == WANT_REPOSITORY, f"SOURCE repository is {repository!r}"
    assert COMMIT_RE.match(commit), f"SOURCE commit {commit!r} is not a full 40-hex commit"
    assert wire_minor > 0, (
        "SOURCE records no wire_minor; a re-vendor that forgets it must not look clean. " + REMEDY
    )
    assert digests, "SOURCE lists no generated files; the manifest is not guarding anything"


def test_vendored_bytes_match_manifest() -> None:
    """The bytes on disk are what SOURCE says, and SOURCE names exactly the files present.

    The second half is the one that catches a half-landed re-vendor: a generated file that
    was added, dropped, or renamed -- a retired ``artifact_limits.py`` left beside the
    current ``weights_limits.py``, say -- cannot sit there unlisted and unguarded.
    """
    _, _, _, digests = read_manifest()

    on_disk = {
        path.name
        for path in VENDOR_DIR.iterdir()
        # Everything here is generated: the directory is a namespace package with no
        # __init__.py and no Runtime-owned source. So an exemption list would only ever
        # be a way to let an unguarded file in.
        if path.is_file() and path.suffix in {".py", ".pyi"}
    }

    for name, want in sorted(digests.items()):
        path = VENDOR_DIR / name
        assert path.is_file(), f"SOURCE lists {name} but it is not vendored; {REMEDY}"
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == want, (
            f"{name} digest mismatch:\n  on disk {got}\n  SOURCE  {want}\n"
            f"the vendored bytes were edited without rewriting SOURCE; {REMEDY}"
        )

    unlisted = sorted(on_disk - set(digests))
    assert not unlisted, (
        f"{', '.join(unlisted)} vendored but absent from SOURCE; an unlisted generated file "
        f"is unguarded -- {REMEDY}"
    )


def test_wire_minor_matches_manifest() -> None:
    """The imported binding and its manifest agree about the wire level this build speaks.

    This is the exact shape of the incident: a re-vendor that half-lands, leaving the
    compiled-in minor and the recorded one disagreeing, with nothing to say so.
    """
    _, _, wire_minor, _ = read_manifest()
    assert wire_minor == WIRE_MINOR, (
        f"imported WIRE_MINOR = {WIRE_MINOR} but SOURCE records wire_minor = {wire_minor}. "
        f"The vendored code and its manifest disagree; {REMEDY}"
    )
