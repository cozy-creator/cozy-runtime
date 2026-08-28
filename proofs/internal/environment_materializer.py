"""Private generation-reuse, wheel-store, and tamper proof below the public proof-seat CLI.

The public environment proof deliberately requires a fresh root. Production replay reuses an
already published receipt-keyed generation, and production materialization hardlinks from the
shared wheel store, so these checks honestly exercise Runtime internals without pretending they
are part of the public CLI contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

from cozy_runtime.internal import child_env, environment, spawn
from cozy_runtime.internal.materialize_launch import environment_input


class ProofFailure(RuntimeError):
    """Private materializer evidence did not match its contract."""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or ())
    if len(args) != 1:
        raise ProofFailure("usage: environment.py PROOF_ROOT")
    root = Path(args[0]).resolve()
    request = _object(root / "request-cold.json")
    wheel_files = _paths(request.get("wheel_files"))
    source = environment_input(
        environment_spec_bytes=_bytes(request, "endpoint_environment_spec"),
        endpoint_bundle_bytes=_bytes(request, "endpoint_bundle"),
        resolved_wheel_set_bytes=_bytes(request, "resolved_wheel_set"),
        wheelhouse_manifest_bytes=_bytes(request, "wheelhouse_manifest"),
        wheel_paths=wheel_files,
    )
    environment_root = _path(request, "environment_root")
    uv = _path(request, "uv")
    python = _path(request, "python")
    expected = (root / "receipt-cold.json").read_bytes()
    expected_digest = _digest(expected)

    raced = environment.materialize(source, environment_root, uv=uv, python=python)
    _require(raced.reused, "publish-race replay did not select the existing generation")
    _require(raced.receipt_bytes == expected, "publish-race replay changed receipt bytes")

    warm = environment.materialize(
        source,
        environment_root,
        uv=uv,
        python=python,
        expected_receipt_digest=expected_digest,
    )
    _require(warm.reused, "receipt-keyed replay did not report reuse")
    _require(warm.receipt_bytes == expected, "receipt-keyed replay changed receipt bytes")

    _bytecode_proof(warm, source)
    _reuse_grant_proof(root, source, environment_root, uv, python, expected, expected_digest)

    site = _site(warm.generation)
    site_mode = site.stat().st_mode
    site.chmod(site_mode | stat.S_IWUSR)
    planted = site / "planted_unrecorded.py"
    planted.write_text("not installed by a wheel\n")
    planted.chmod(planted.stat().st_mode & ~0o222)
    site.chmod(site_mode)
    _refusal(
        "installed_unrecorded_file",
        lambda: environment.verify_generation(
            warm.generation,
            receipt_digest=warm.receipt_digest,
            environment_spec_digest=source.environment_spec_digest,
        ),
    )
    site.chmod(site_mode | stat.S_IWUSR)
    planted.unlink()
    site.chmod(site_mode)

    installed = site / "proof_environment_project" / "__init__.py"
    installed.chmod(installed.stat().st_mode | stat.S_IWUSR)
    installed.write_text("VALUE = 'tampered'\n")
    installed.chmod(installed.stat().st_mode & ~0o222)
    _refusal(
        "installed_record_mismatch",
        lambda: environment.verify_generation(
            warm.generation,
            receipt_digest=warm.receipt_digest,
            environment_spec_digest=source.environment_spec_digest,
        ),
    )
    # The reuse path is the last guard before an executor runs: `materialize` must refuse the
    # same corruption `verify_generation` just refused, not hand back a generation because its
    # receipt file still hashes correctly.
    _refusal(
        "installed_record_mismatch",
        partial(
            environment.materialize,
            source,
            environment_root,
            uv=uv,
            python=python,
            expected_receipt_digest=warm.receipt_digest,
        ),
    )
    _store_proof(root, source, uv, python, expected, expected_digest)
    print(
        "private environment materializer: publish-race and receipt-keyed reuse retained exact "
        "receipt bytes; unrecorded and tampered installed bytes refused; wheel-store "
        "materialization shared inodes at byte-identical receipts, survived concurrency, and "
        "refused a cross-device store and a tampered store entry; bytecode cached to the "
        "generation-keyed tree outside the immutable venv and was reused there; reuse "
        "skipped unreadable grants while every installing path still refused them"
    )
    return 0


def _bytecode_proof(
    warm: environment.InstalledEnvironment, source: environment.EnvironmentInput
) -> None:
    """Bytecode must reach a generation-keyed tree OUTSIDE the venv, and stay reused.

    Nothing timed here: the mechanism is what is proven.  An immutable, RECORD-closed venv
    cannot hold a ``.pyc`` at all -- it would be an ``installed_unrecorded_file`` refusal --
    so the only question worth an arm is whether the bytecode lands outside it and survives.
    """

    tree = warm.bytecode_cache
    _require(
        tree.name == warm.receipt_digest.removeprefix("sha256:")
        and tree.parent.name == environment.BYTECODE_DIR,
        f"the bytecode tree {tree} is not keyed by the receipt digest",
    )
    _require(warm.generation not in tree.parents, f"bytecode tree {tree} is inside the venv")
    _require(tree.is_dir(), "materialization did not open the bytecode tree")
    # The whole point is that the tree reaches the child: an allowlist row that the ERASE
    # pass does not clear, or one the seal drops, would leave every start recompiling.
    ambient = (("PATH", "/usr/bin"), (child_env.BYTECODE_CACHE_ENV, "/ambient/poison"))
    base = tuple((k, v) for k, v in ambient if not child_env.erased(k))
    sealed = spawn.seal(base, {child_env.BYTECODE_CACHE_ENV: str(tree)})
    _require(
        sealed.get(child_env.BYTECODE_CACHE_ENV) == str(tree),
        "the executor seal does not carry the generation's bytecode tree",
    )

    # Executor-shaped: the sealed environment only, and no -I/-E, which would ignore it.
    env = {"PATH": "/usr/bin", child_env.BYTECODE_CACHE_ENV: str(tree)}
    probe = [str(warm.python), "-s", "-c", "import proof_environment_project"]
    subprocess.run(probe, env=env, check=True, capture_output=True)
    cached = sorted(tree.rglob("*.pyc"))
    site = _site(warm.generation)
    _require(bool(cached), "an executor-shaped start cached no bytecode")
    _require(
        any(str(site) in str(path) for path in cached),
        f"no installed module's bytecode reached {tree}",
    )
    _require(
        not list((warm.generation / "venv").rglob("__pycache__")),
        "bytecode landed inside the immutable venv",
    )
    stamps = {path: path.stat().st_mtime_ns for path in cached}
    subprocess.run(probe, env=env, check=True, capture_output=True)
    _require(
        {path: path.stat().st_mtime_ns for path in sorted(tree.rglob("*.pyc"))} == stamps,
        "a warm start rewrote its cached bytecode instead of reusing it",
    )
    environment.verify_generation(
        warm.generation,
        receipt_digest=warm.receipt_digest,
        environment_spec_digest=source.environment_spec_digest,
    )


def _reuse_grant_proof(
    root: Path,
    source: environment.EnvironmentInput,
    environment_root: Path,
    uv: Path,
    python: Path,
    expected: bytes,
    expected_digest: str,
) -> None:
    """Wheel bytes are hashed on the path that installs them -- and on every such path."""

    absent = replace(
        source,
        grants=tuple(replace(grant, path=root / "no-such-wheel") for grant in source.grants),
    )
    reused = environment.materialize(
        absent,
        environment_root,
        uv=uv,
        python=python,
        expected_receipt_digest=expected_digest,
    )
    _require(
        reused.reused and reused.receipt_bytes == expected,
        "reuse of a published generation did not survive unreadable grants",
    )
    # A reuse that finds no generation must install, and installing must verify: an expected
    # digest is not a licence to skip the wheels.
    for target, digest in (
        (root / "environment-grant-cold", ""),
        (root / "environment-grant-absent", expected_digest),
    ):
        _refusal(
            "wheel_blob_unreadable",
            partial(
                environment.materialize,
                absent,
                target,
                uv=uv,
                python=python,
                expected_receipt_digest=digest,
            ),
        )
        _require(not target.exists(), f"a refused materialization left {target}")


def _store_proof(
    root: Path,
    source: environment.EnvironmentInput,
    uv: Path,
    python: Path,
    expected: bytes,
    expected_digest: str,
) -> None:
    """The store must share inodes, change no receipt byte, and fail closed."""

    store = root / "wheel-store"

    foreign = Path("/dev/shm") / f"cozy-wheel-store-{os.getpid()}"
    try:
        _refusal(
            "wheel_store_cross_device",
            lambda: environment.materialize(
                source,
                root / "environment-store-xdev",
                uv=uv,
                python=python,
                store=foreign,
            ),
        )
    finally:
        shutil.rmtree(foreign, ignore_errors=True)

    # Two placements materializing at once must not corrupt or half-populate an entry.
    with ThreadPoolExecutor(max_workers=2) as pool:
        raced = [
            future.result()
            for future in [
                pool.submit(
                    environment.materialize,
                    source,
                    root / f"environment-store-{name}",
                    uv=uv,
                    python=python,
                    store=store,
                )
                for name in ("a", "b")
            ]
        ]
    for installed in raced:
        _require(
            installed.receipt_bytes == expected,
            "wheel-store materialization changed the receipt bytes",
        )
        _require(installed.receipt_digest == expected_digest, "wheel-store receipt digest differs")

    # A third generation, now certainly a full store hit, must hardlink rather than unpack.
    third = environment.materialize(
        source, root / "environment-store-c", uv=uv, python=python, store=store
    )
    _require(third.receipt_bytes == expected, "store-hit materialization changed receipt bytes")

    shared = 0
    for path in sorted(_site(third.generation).rglob("*")):
        if not path.is_file():
            continue
        info = path.stat()
        _require(not path.is_symlink(), f"store linked a symlink: {path}")
        _require(not info.st_mode & 0o222, f"store linked a writable file: {path}")
        if info.st_nlink > 1:
            shared += 1
    _require(shared > 0, "store-hit generation shares no inode with the store")

    entries = sorted(p for p in store.rglob("entry.json"))
    _require(bool(entries), "store published no entry")
    for entry in entries:
        document = json.loads(entry.read_bytes())
        _require(
            document.get("format") == "cozy.runtime.WheelStoreEntry/1",
            f"store entry has the wrong format: {entry}",
        )
        _require(not entry.stat().st_mode & 0o222, f"store entry is mutable: {entry}")

    # A poisoned store entry is caught by RECORD verification, not trusted into a generation.
    victim = _project_file(store)
    original = victim.read_bytes()
    mode = victim.stat().st_mode
    victim.chmod(mode | stat.S_IWUSR)
    victim.write_bytes(b"VALUE = 'store tamper'\n")
    victim.chmod(mode)
    try:
        _refusal(
            "installed_record_mismatch",
            lambda: environment.materialize(
                source,
                root / "environment-store-tampered",
                uv=uv,
                python=python,
                store=store,
            ),
        )
    finally:
        victim.chmod(mode | stat.S_IWUSR)
        victim.write_bytes(original)
        victim.chmod(mode)


def _project_file(store: Path) -> Path:
    matches = sorted(store.rglob("tree/lib/python*/site-packages/proof_environment_project/*.py"))
    _require(len(matches) == 1, f"store holds {len(matches)} project sources, wanted one")
    return matches[0]


def _refusal(code: str, call: Callable[[], object]) -> None:
    try:
        call()
    except environment.EnvironmentRefusal as exc:
        _require(exc.code == code, f"wanted {code}, got {exc.code}: {exc.detail}")
    else:
        raise ProofFailure(f"{code} refusal did not fire")


def _site(generation: Path) -> Path:
    sites = list((generation / "venv").glob("lib/python*/site-packages"))
    _require(len(sites) == 1, f"generation has {len(sites)} site-packages directories")
    return sites[0]


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ProofFailure(f"{path} is not a JSON object")
    return value


def _paths(value: object) -> Mapping[str, Path]:
    if not isinstance(value, dict):
        raise ProofFailure("wheel_files is not an object")
    result: dict[str, Path] = {}
    for digest, path in value.items():
        if not isinstance(digest, str) or not isinstance(path, str):
            raise ProofFailure("wheel_files is not a string-to-string object")
        result[digest] = Path(path)
    return result


def _bytes(request: Mapping[str, object], name: str) -> bytes:
    return _path(request, name).read_bytes()


def _path(request: Mapping[str, object], name: str) -> Path:
    value = request.get(name)
    if not isinstance(value, str):
        raise ProofFailure(f"{name} is not a path string")
    return Path(value)


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ProofFailure(detail)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
