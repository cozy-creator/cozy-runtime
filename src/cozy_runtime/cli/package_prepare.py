"""Private Creator plumbing for one observed local base and package overlay."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import canonical, local_base, overlay
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.worker.child import ExecutorGone, ExecutorSupervision
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal, prepare_package
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def write_local_base(args: list[str], _opts: Options) -> Result:
    values = _values(args, {"--out"}, set())
    output = Path(values["--out"][0]).expanduser().resolve()
    try:
        digest, length, changed = local_base.write_manifest(output)
    except local_base.LocalBaseRefusal as exc:
        raise CliError(
            "local_base_invalid",
            str(exc),
            "install the current Cozy Runtime tool under CPython 3.12; "
            "Cozy will not resolve or replace it",
            Exit.structural,
        ) from exc
    return Result(
        rows=(("manifest", str(output)), ("digest", digest), ("changed", str(changed).lower())),
        document={
            "changed": changed,
            "digest": digest,
            "length": length,
            "path": str(output),
        },
    )


def prepare_local_package(config: RuntimeConfig, args: list[str], _opts: Options) -> Result:
    required = {
        "--artifact-cache",
        "--artifact-store",
        "--base-manifest",
        "--environment-root",
        "--package",
        "--project-wheel",
        "--release",
        "--release-digest",
    }
    values = _values(args, required, {"--dependency-wheel", "--model"})
    artifact_cache = Path(values["--artifact-cache"][0]).resolve()
    artifact_store = Path(values["--artifact-store"][0]).resolve()
    environment_root = Path(values["--environment-root"][0]).resolve()
    base_manifest = Path(values["--base-manifest"][0]).resolve()
    wheel_paths = [
        (pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL, Path(values["--project-wheel"][0]).resolve()),
        *(
            (pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL, Path(value).resolve())
            for value in values.get("--dependency-wheel", [])
        ),
    ]
    rows: list[pb.LocalDownloadFile] = []
    for kind, path in wheel_paths:
        try:
            info = path.stat()
            digest = _hash_file(path)
        except OSError as exc:
            raise CliError(
                "package_prepare_file_unreadable",
                str(path),
                "download the exact published wheel again",
                Exit.not_found,
            ) from exc
        rows.append(
            pb.LocalDownloadFile(
                digest=bytes.fromhex(digest.removeprefix("sha256:")),
                filename=path.name,
                kind=kind,
                length=info.st_size,
                path=str(path),
            )
        )
    rows.sort(key=lambda row: row.digest)
    package = values["--package"][0]
    release = values["--release"][0]
    release_digest = values["--release-digest"][0]
    models: list[dict[str, object]] = []
    for raw in values.get("--model", []):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CliError(
                "usage",
                "invalid modeled-package selection",
                "Creator supplies exact JSON",
                Exit.usage,
            ) from exc
        required_model = {
            "package",
            "slot",
            "model",
            "release",
            "lane",
            "manifest",
            "manifest_length",
        }
        if not isinstance(row, dict) or set(row) != required_model:
            raise CliError(
                "usage",
                "invalid modeled-package selection",
                "Creator supplies exact JSON",
                Exit.usage,
            )
        models.append(row)
    seed = canonical.write(
        {"package": package, "release": release, "release_digest": release_digest}
    )
    # Linux caps Unix socket paths at 108 bytes. The generation/cache path is
    # user-selectable and may be much longer, so disposable executor control
    # belongs in a private short system-temporary directory.
    scratch = Path(tempfile.mkdtemp(prefix="cozy-prepare-"))
    supervisor = ExecutorSupervision(
        root=scratch / "worker",
        python=sys.executable,
        base_env=config.child_base_env,
        cozy_home=config.cozy_home,
        scratch_root=scratch / "executor",
    )
    (scratch / "executor").mkdir(parents=True)

    def describe(installed: overlay.InstalledOverlay, distribution: str) -> bytes:
        supervisor.use_overlay(
            str(installed.python), str(installed.import_root), installed.content_digest
        )
        executor = supervisor.spawn(imposed={})
        output = scratch / "executor" / "descriptor.json"
        reply = executor.call(
            {"cmd": "describe_installed", "distribution": distribution, "output": str(output)},
            timeout=1.0,
            total_timeout=65.0,
        )
        if not reply.get("ok"):
            raise PreparationRefusal(
                "package_prepare_descriptor_failed", str(reply.get("code", "invalid reply"))
            )
        try:
            raw = output.read_bytes()
        except OSError as exc:
            raise PreparationRefusal(
                "package_prepare_descriptor_failed", "descriptor output is absent"
            ) from exc
        if reply.get("digest") != _digest(raw) or reply.get("length") != len(raw):
            raise PreparationRefusal("package_prepare_descriptor_failed", "invalid reply")
        return raw

    try:
        result = prepare_package(
            package_name=package,
            release=release,
            release_digest=release_digest,
            files=rows,
            placement_seed=seed,
            artifact_cache=artifact_cache,
            environment_root=environment_root,
            python=Path(sys.executable),
            describe=describe,
            verified=lambda _digest, _length, _path: None,
            base_manifest=base_manifest,
            artifact_store=artifact_store,
            models=models,
            allow_native_custom=True,
        )
    except (PreparationRefusal, ExecutorGone, OSError) as exc:
        name = getattr(exc, "code", "package_prepare_failed")
        detail = getattr(exc, "detail", str(exc))
        raise CliError(
            str(name),
            str(detail),
            "use wheels compatible with this machine or repair the current local Runtime base",
            Exit.validation,
        ) from exc
    finally:
        supervisor.close()
        shutil.rmtree(scratch, ignore_errors=True)

    placement_bytes = result.placement_set.placement_set_canonical_bytes
    placement = documents.read(placement_bytes, pb.PlacementSet)["placements"][0]
    descriptor_ref = placement["package_descriptor"]
    descriptor_path = artifact_cache / str(descriptor_ref["digest"]).removeprefix("sha256:")
    descriptor_raw = descriptor_path.read_bytes()
    return Result(
        rows=(("package", package), ("release", release)),
        document={
            "package": package,
            "release": release,
            "package_descriptor": _exact(descriptor_raw),
            "placement_set": _exact(placement_bytes),
        },
    )


def _values(args: list[str], required: set[str], repeatable: set[str]) -> dict[str, list[str]]:
    allowed = required | repeatable
    values: dict[str, list[str]] = {}
    rest = list(args)
    while rest:
        flag = rest.pop(0)
        if flag not in allowed or not rest or (flag in values and flag not in repeatable):
            raise CliError(
                "usage",
                f"invalid package preparation argument {flag!r}",
                "Creator supplies the complete private package preparation command",
                Exit.usage,
            )
        values.setdefault(flag, []).append(rest.pop(0))
    missing = sorted(required - set(values))
    if missing:
        raise CliError(
            "usage",
            "package preparation is missing " + ", ".join(missing),
            "Creator supplies every required path and package identity",
            Exit.usage,
        )
    return values


def _exact(raw: bytes) -> dict[str, object]:
    return {
        "canonical_bytes_base64": base64.b64encode(raw).decode(),
        "digest": _digest(raw),
        "length": len(raw),
    }


def _hash_file(path: Path) -> str:
    measured = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            measured.update(chunk)
    return "sha256:" + measured.hexdigest()


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()
