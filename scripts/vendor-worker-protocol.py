#!/usr/bin/env python3
"""Vendor worker-protocol's Python bindings and its canonical corpus from ONE commit.

    uv run python scripts/vendor-worker-protocol.py <commit> [--repo ../worker-protocol]

Reads the committed tree (`git archive`), never a working copy, and rewrites both halves
together so their pins cannot drift: `src/cozy/worker/v1/` from `gen/python/cozy/worker/v1/`
and `tests/testdata/worker-protocol/` from `fixtures/`. Each gets a SOURCE of repository,
commit, wire_minor and a sha256 per file; `test_vendored_protocol` and
`test_canonical_corpus` check exactly that.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ROOT / "src/cozy/worker/v1"
CORPUS = ROOT / "tests/testdata/worker-protocol"
CORPUS_MEMBERS = ("MANIFEST.json", "canonical", "red", "tolerated")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("commit")
    parser.add_argument("--repo", type=Path, default=ROOT.parent / "worker-protocol")
    args = parser.parse_args()
    git = ["git", "-C", str(args.repo)]
    commit = subprocess.run(
        [*git, "rev-parse", "--verify", f"{args.commit}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    archive = subprocess.run(
        [*git, "archive", commit, "WIRE_MINOR", "gen/python/cozy/worker/v1", "fixtures"],
        check=True,
        capture_output=True,
    ).stdout
    with tempfile.TemporaryDirectory() as raw, tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tree = Path(raw)
        tar.extractall(tree, filter="data")
        wire_minor = (tree / "WIRE_MINOR").read_text().strip()
        for name in _pinned(BINDINGS):
            (BINDINGS / name).unlink(missing_ok=True)
        generated = sorted((tree / "gen/python/cozy/worker/v1").iterdir())
        for path in generated:
            shutil.copyfile(path, BINDINGS / path.name)
        _write_source(
            BINDINGS,
            "https://github.com/cozy-creator/worker-protocol",
            commit,
            wire_minor,
            [path.name for path in generated],
        )
        for member in CORPUS_MEMBERS:
            target, source = CORPUS / member, tree / "fixtures" / member
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copyfile(source, target)
        files = [
            path.relative_to(CORPUS).as_posix()
            for path in CORPUS.rglob("*")
            if path.is_file() and path.name != "SOURCE"
        ]
        _write_source(
            CORPUS, "https://github.com/cozy-creator/worker-protocol-v2", commit, wire_minor, files
        )
    print(f"vendored worker-protocol {commit} (wire minor {wire_minor}): bindings and corpus")


def _pinned(directory: Path) -> list[str]:
    """Files the current SOURCE vouches for, so one the new commit dropped does not linger."""
    keys = [line.split("=", 1)[0] for line in (directory / "SOURCE").read_text().splitlines()]
    return [key for key in keys if key not in ("repository", "commit", "wire_minor")]


def _write_source(
    directory: Path, repository: str, commit: str, wire_minor: str, files: list[str]
) -> None:
    rows = [f"repository={repository}", f"commit={commit}", f"wire_minor={wire_minor}"]
    for name in sorted(files):
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        rows.append(f"{name}=sha256:{digest}")
    (directory / "SOURCE").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
