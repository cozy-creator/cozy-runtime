"""Private Creator plumbing for one observed local base and package environment."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import canonical, package_environment
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.worker.child import (
    INTERFACE_DOCUMENT,
    ExecutorGone,
    ExecutorSupervision,
)
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    prepare_published_local,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def prepare_local_package(config: RuntimeConfig, args: list[str], _opts: Options) -> Result:
    required = {
        "--artifact-cache",
        "--tensorfs-root",
        "--environment-python",
        "--locked-requirements",
        "--package",
        "--release",
    }
    values = _values(args, required, {"--model"})
    artifact_cache = Path(values["--artifact-cache"][0]).resolve()
    tensorfs_root = Path(values["--tensorfs-root"][0]).resolve()
    environment_python = _venv_python(values["--environment-python"][0])
    lock_path = Path(values["--locked-requirements"][0]).resolve()
    try:
        locked = package_environment.read_locked_requirements(lock_path.read_bytes())
    except (OSError, package_environment.EnvironmentRefusal) as exc:
        raise CliError(
            "package_prepare_locked_requirements_invalid",
            str(exc),
            "supply the release's exact locked-requirements export",
            Exit.validation,
        ) from exc
    package = values["--package"][0]
    release = values["--release"][0]
    try:
        installed_environment = package_environment.open_existing_environment(
            environment_python,
            canonical.digest({"locked_requirements": locked.digest}),
        )
    except package_environment.EnvironmentRefusal as exc:
        raise CliError(
            exc.code,
            exc.detail,
            "recreate the package's uv environment from its exact lock",
            Exit.validation,
        ) from exc
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
        if (
            not isinstance(row, dict)
            or not required_model <= row.keys()
            or set(row) - required_model - {"adapters"}
        ):
            raise CliError(
                "usage",
                "invalid modeled-package selection",
                "Creator supplies exact JSON",
                Exit.usage,
            )
        models.append(row)
    seed = canonical.write(
        {
            "package": package,
            "release": release,
            "locked_requirements_digest": locked.digest,
        }
    )
    # Linux caps Unix socket paths at 108 bytes. The generation/cache path is
    # user-selectable and may be much longer, so disposable executor control
    # belongs in a private short system-temporary directory.
    scratch = Path(tempfile.mkdtemp(prefix="cozy-prepare-"))
    supervisor = ExecutorSupervision(
        root=scratch / "worker",
        python=str(environment_python),
        base_env=config.child_base_env,
        cozy_home=config.cozy_home,
    )

    def describe(installed: package_environment.InstalledEnvironment, distribution: str) -> bytes:
        supervisor.use_environment(str(installed.python), installed.content_digest)
        executor = supervisor.spawn(imposed={})
        output = supervisor.handoff(executor.epoch, INTERFACE_DOCUMENT)
        with executor.watched("describe_installed"):
            reply = executor.call(
                {"cmd": "describe_installed", "distribution": distribution, "output": str(output)},
                timeout=None,
            )
        if not reply.get("ok"):
            raise PreparationRefusal(
                "package_prepare_interface_failed", str(reply.get("code", "invalid reply"))
            )
        try:
            raw = output.read_bytes()
        except OSError as exc:
            raise PreparationRefusal(
                "package_prepare_interface_failed", "package-interface output is absent"
            ) from exc
        if reply.get("digest") != _digest(raw) or reply.get("length") != len(raw):
            raise PreparationRefusal("package_prepare_interface_failed", "invalid reply")
        return raw

    try:
        result = prepare_published_local(
            package_name=package,
            release=release,
            locked=locked,
            models=models,
            placement_seed=seed,
            artifact_cache=artifact_cache,
            tensorfs_root=tensorfs_root,
            install_root=installed_environment.generation,
            python=environment_python,
            describe=describe,
            installed_environment=installed_environment,
            # The same job-plan root the worker session stages into: without it a
            # job-only package has no runnable callable and local install refuses.
            job_plan_root=config.cozy_home / "job-plans",
        )
    except (PreparationRefusal, ExecutorGone, OSError) as exc:
        name = getattr(exc, "code", "package_prepare_failed")
        detail = getattr(exc, "detail", str(exc))
        raise CliError(
            str(name),
            str(detail),
            "use a lock compatible with this machine or repair the current local Runtime base",
            Exit.validation,
        ) from exc
    finally:
        supervisor.close()
        shutil.rmtree(scratch, ignore_errors=True)

    placement_bytes = result.placement_set.placement_set_canonical_bytes
    placement = documents.read(placement_bytes, pb.PlacementSet)["placements"][0]
    interface_ref = placement["package_interface"]
    staged_interface = artifact_cache / str(interface_ref["digest"]).removeprefix("sha256:")
    interface_raw = staged_interface.read_bytes()
    return Result(
        rows=(("package", package), ("release", release)),
        document={
            "package": package,
            "release": release,
            "package_interface": _exact(interface_raw),
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
                "Creator supplies the complete local package preparation command",
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


def _venv_python(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.parent.resolve() / path.name


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
