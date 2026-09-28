"""Every frozen canonical document renders byte-identically through THIS writer (xs-020).

Identity in `cozy.worker.v1` is the sha256 of a document's exact canonical bytes, so two
independent writers must produce byte-identical documents or nothing downstream — digests,
admission, terminals — agrees at all. worker-protocol freezes the corpus; the observers live
in the repos that consume it. This is Runtime's observer: the corpus is VENDORED at
``tests/testdata/worker-protocol`` and digest-fenced here, so the check runs in every
``uv run pytest`` with no token, no sibling checkout, and no skip. A skip is
indistinguishable from a pass, which is how three paid dispatches died on a drifted mirror
(#409) that a frozen fixture had described all along.

The vendored corpus carries its own SOURCE in the same ``key=value`` shape as the bindings'
(``src/cozy/worker/v1/SOURCE``), pinned to the SAME worker-protocol commit — one upstream
pin, not two — and the token-gated ``drift`` job in CI byte-compares both trees against the
real checkout at that commit.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cozy.worker.v1 import worker_pb2
from cozy_runtime.internal import canonical
from cozy_runtime.internal.worker.session import REPREPARE, read_placement_set
from cozy_runtime.protocol import WIRE_MINOR, documents

CORPUS = Path(__file__).resolve().parent / "testdata" / "worker-protocol"
SOURCE = CORPUS / "SOURCE"
BINDINGS_SOURCE = Path(__file__).resolve().parents[1] / "src" / "cozy" / "worker" / "v1" / "SOURCE"
WANT_REPOSITORY = "https://github.com/cozy-creator/worker-protocol-v2"
REMEDY = (
    "re-vendor bindings and corpus together: uv run python scripts/vendor-worker-protocol.py "
    "<commit>"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def read_corpus_manifest() -> tuple[str, str, int, dict[str, str]]:
    """The corpus SOURCE parsed into its ``key=value`` lines, strictly.

    An unknown key is a typo, and a typo that parses is a digest nobody checks.
    """
    repository = commit = ""
    wire_minor = 0
    digests: dict[str, str] = {}
    assert SOURCE.is_file(), f"the vendored corpus has no SOURCE manifest at {SOURCE}"
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
        elif key == "MANIFEST.json" or (
            key.startswith(("canonical/", "red/", "tolerated/")) and key.endswith((".json", ".bin"))
        ):
            assert value.startswith("sha256:") and len(value) == 71, (
                f"{SOURCE}:{number}: {key} digest is not a sha256: {value!r}"
            )
            digests[key] = value.removeprefix("sha256:").lower()
        else:
            raise AssertionError(f"{SOURCE}:{number}: unknown manifest key {key!r}")
    return repository, commit, wire_minor, digests


def test_corpus_source_manifest_is_exact() -> None:
    repository, commit, wire_minor, digests = read_corpus_manifest()
    assert repository == WANT_REPOSITORY, f"corpus SOURCE repository is {repository!r}"
    assert COMMIT_RE.match(commit), f"corpus SOURCE commit {commit!r} is not a full 40-hex commit"
    assert wire_minor > 0, "corpus SOURCE records no wire_minor. " + REMEDY
    assert "MANIFEST.json" in digests, "corpus SOURCE does not pin MANIFEST.json. " + REMEDY
    # Every canonical document travels as a pair: exact canonical bytes and protobuf twin.
    for name in digests:
        if name.startswith("canonical/"):
            stem, _, suffix = name.rpartition(".")
            twin = stem + (".bin" if suffix == "json" else ".json")
            assert twin in digests, f"corpus SOURCE lists {name} without its twin; {REMEDY}"


def test_corpus_pin_is_bindings_pin() -> None:
    """ONE upstream pin, not two.

    The corpus freezes the documents the pinned schema describes; re-vendoring the bindings
    without re-vendoring the corpus (or the reverse) leaves the two halves describing
    different wire levels — the half-landed state every fence in this plane exists to refuse.
    """
    _, corpus_commit, corpus_minor, _ = read_corpus_manifest()
    bindings = dict(
        line.split("=", 1) for line in BINDINGS_SOURCE.read_text().splitlines() if "=" in line
    )
    assert corpus_commit == bindings["commit"], (
        f"the bindings are vendored from {bindings['commit']} but the corpus from "
        f"{corpus_commit}; re-vendor both from one worker-protocol commit. {REMEDY}"
    )
    assert corpus_minor == WIRE_MINOR, (
        f"corpus SOURCE records wire_minor {corpus_minor} but the imported binding speaks "
        f"{WIRE_MINOR}; {REMEDY}"
    )


def test_corpus_bytes_match_manifest() -> None:
    """The corpus on disk is exactly what SOURCE says, and SOURCE names every file present."""
    _, _, _, digests = read_corpus_manifest()
    on_disk = {
        path.relative_to(CORPUS).as_posix()
        for path in CORPUS.rglob("*")
        if path.is_file() and path.name != "SOURCE"
    }
    for name, want in sorted(digests.items()):
        path = CORPUS / name
        assert path.is_file(), f"corpus SOURCE lists {name} but it is not vendored; {REMEDY}"
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == want, (
            f"{name} digest mismatch:\n  on disk {got}\n  SOURCE  {want}\n"
            f"the vendored corpus was edited without rewriting SOURCE; {REMEDY}"
        )
    unlisted = sorted(on_disk - set(digests))
    assert not unlisted, (
        f"{', '.join(unlisted)} vendored but absent from corpus SOURCE; an unlisted fixture "
        f"is unguarded — {REMEDY}"
    )


def test_canonical_documents() -> None:
    """Every frozen canonical document, through this writer and back through this reader.

    The corpus's own MANIFEST.json enumerates the documents, so a fixture frozen upstream
    cannot sit here unexercised: a document type with no binding fails, and a corpus that
    shrank to nothing fails the arms count rather than passing vacuously.
    """
    manifest = json.loads((CORPUS / "MANIFEST.json").read_text())
    assert manifest["wire_minor"] == WIRE_MINOR, (
        f"the frozen corpus was written at wire minor {manifest['wire_minor']} but the "
        f"imported binding speaks {WIRE_MINOR}; {REMEDY}"
    )
    canonical_rows = manifest["canonical"]
    arms = 0
    for name in sorted(canonical_rows):
        row = canonical_rows[name]
        cls = getattr(worker_pb2, row["type"].rpartition(".")[2], None)
        assert cls is not None, f"{name}: the vendored binding has no {row['type']}"
        message = cls()
        message.ParseFromString((CORPUS / "canonical" / f"{name}.bin").read_bytes())
        frozen = (CORPUS / "canonical" / f"{name}.json").read_bytes()
        mine, digest = documents.identity(message)
        assert mine == frozen, (
            f"{name}: this writer produced {len(mine)} B that differ from the frozen "
            f"document ({len(frozen)} B)"
        )
        assert documents.spell(digest) == row["id"], f"{name}: id differs from the frozen id"
        assert documents.doc_format(message.DESCRIPTOR.full_name) == row["document"], (
            f"{name}: format tag differs from the frozen document tag"
        )
        # The reading half decodes exactly the message the writer rendered.
        assert documents.parse(frozen, cls) == message, f"{name}: decode differs"
        arms += 1
    assert arms == len(canonical_rows) and arms > 0, (
        f"exercised {arms} of {len(canonical_rows)} canonical documents"
    )


def test_tolerated_documents_are_read() -> None:
    """Unknown keys at any depth and absent collections from another writer are read."""
    tolerated = json.loads((CORPUS / "MANIFEST.json").read_text())["tolerated"]
    assert tolerated
    for name, row in sorted(tolerated.items()):
        body = (CORPUS / "tolerated" / f"{name}.json").read_bytes()
        assert documents.spell(documents.digest_of(body)) == row["id"], name
        cls = getattr(worker_pb2, row["type"].rpartition(".")[2])
        documents.parse(body, cls)
        if cls is worker_pb2.PlacementSet:
            read_placement_set(
                worker_pb2.DesiredPlacementSet(
                    placement_set_digest=documents.digest_of(body),
                    placement_set_canonical_bytes=body,
                )
            )


def test_red_twins_refuse_ambiguous_encodings() -> None:
    """Duplicate keys, a float and non-canonical bytes are ambiguous encodings: all refuse."""
    for name, want_code in {
        "twin_duplicate_key": "duplicate_key",
        "twin_float": "non_integer_number",
        "twin_whitespace": "noncanonical_encoding",
    }.items():
        body = (CORPUS / "red" / f"{name}.json").read_bytes()
        with pytest.raises(documents.DocumentError) as refusal:
            documents.parse(body, worker_pb2.InvocationSpec)
        assert refusal.value.code == want_code, (
            f"{name}: refused as {refusal.value.code!r}, wanted {want_code!r}"
        )


def _first(document: dict[str, Any]) -> dict[str, Any]:
    placement: dict[str, Any] = document["placements"][0]
    return placement


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (
            lambda d: _first(d)["entrypoints"][0]["slots"][0].update(reference_model_id="x"),
            "unknown_model",
        ),
        (
            lambda d: _first(d)["entrypoints"][0]["slots"][0].update(
                adapters=[
                    {"component": "a", "model_id": "h3", "source_component": "a", "scale": " "}
                ]
            ),
            "invalid_scale",
        ),
        (
            lambda d: _first(d)["entrypoints"][0].update(entrypoint_binding_digest="sha256:X"),
            "malformed_digest",
        ),
        (lambda d: _first(d)["models"][0]["manifest"].pop("length"), "invalid_ref"),
        (
            lambda d: _first(d).update(development={"package": "a", "release": "1"}),
            "oneof_conflict",
        ),
        (lambda d: _first(d).pop("package"), "package_mode_invalid"),
        (lambda d: _first(d).update(placement_id=" padded"), "invalid_identifier"),
        (lambda d: _first(d).pop("installation_id"), REPREPARE),
        (lambda d: _first(d).update(package_interface={"digest": "sha256:" + "e" * 64}), REPREPARE),
        (lambda d: d["placements"].append(_first(d)), "duplicate_placement"),
    ],
)
def test_a_placement_set_is_refused_by_name(
    change: Callable[[dict[str, Any]], object], code: str
) -> None:
    """The typed decode keeps every refusal a caller classifies on."""
    document = json.loads((CORPUS / "tolerated" / "tolerate_absent_collection.json").read_bytes())
    change(document)
    raw = canonical.write(document)
    with pytest.raises(documents.DocumentError) as refused:
        read_placement_set(
            worker_pb2.DesiredPlacementSet(
                placement_set_digest=documents.digest_of(raw), placement_set_canonical_bytes=raw
            )
        )
    assert refused.value.code == code
