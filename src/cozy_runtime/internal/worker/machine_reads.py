"""The machine's own reads (wire 66): its runs, installed packages, weights and environment.

Each answer comes from the fact's owner: the execution journal, the installations, the TensorFS
store (through the `tfs` of this Runtime's own TensorFS build) and this process's measurements.
A fact this machine cannot read is left empty; a read never fails over one row.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sysconfig
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import fill, job_plan, package_interface, python_interpreters
from cozy_runtime.internal.config import package_install_environment
from cozy_runtime.internal.package_environment import EnvironmentRefusal
from cozy_runtime.internal.package_installation import PUBLISHED_PREFIX, open_installation
from cozy_runtime.internal.worker.workspace_executions import Preparation, Run
from cozy_runtime.protocol import MIN_COMPATIBLE_WIRE_MINOR, WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker

#: When this Runtime began serving, for `DescribeMachine`.
STARTED_MS = time.time_ns() // 1_000_000
_UNREADABLE = (OSError, ValueError, msgspec.MsgspecError, documents.DocumentError)


# ------------------------------------------------------------------ runs


class _JobPlan(msgspec.Struct, frozen=True):
    job: str = ""


def target(worker: Worker, run: Run) -> pb.MachineExecutionTarget:
    """What a run executes: its offer's installation and entrypoint, named by the placement
    it was prepared under."""
    try:
        offer = pb.AttemptOffer.FromString(run.offer)
        spec = documents.parse(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
        prepared = Preparation.read(run.preparation)
    except _UNREADABLE:
        return pb.MachineExecutionTarget()
    job = spec.HasField("job")
    installation = spec.job.installation_id if job else spec.installation_id
    found = pb.MachineExecutionTarget(installation_id=installation, job=job)
    held = prepared.installations.get(installation)
    if held is None:
        return found
    placement = held.placement
    chosen = placement.development if placement.HasField("development") else placement.package
    found.package, found.release = chosen.package, chosen.release
    if job:
        found.entrypoint = _job_name(worker, installation, spec.job.job_descriptor_id)
    else:
        found.entrypoint = next(
            (
                entry.name
                for entry in placement.entrypoints
                if documents.spell(entry.entrypoint_binding_digest)
                == spec.serving.entrypoint_binding_digest
            ),
            "",
        )
    return found


def _job_name(worker: Worker, installation: str, descriptor: str) -> str:
    try:
        plan = job_plan.path(worker.config.cozy_home / "job-plans", installation, descriptor)
        return msgspec.json.decode(plan.read_bytes(), type=_JobPlan).job
    except _UNREADABLE:
        return ""


# ------------------------------------------------------------------ packages


class _Installed(msgspec.Struct, frozen=True):
    """An `installation.json`, as far as a listing reads it."""

    package: str
    release: str
    sdk: dict[str, str] = {}


def packages(worker: Worker) -> pb.PackageList:
    """Every installation this machine can open, sorted by package, release and id."""
    root = worker.options.install_root
    directory = root / "installations" if root is not None else None
    if root is None or directory is None or not directory.is_dir():
        return pb.PackageList()
    rows = []
    for path in directory.iterdir():
        record = path / "installation.json"
        try:
            installed = msgspec.json.decode(record.read_bytes(), type=_Installed)
            open_installation(root, path.name)  # an unfinished install is not one it holds
            installed_ms = record.stat().st_mtime_ns // 1_000_000
        except (*_UNREADABLE, EnvironmentRefusal):
            continue
        rows.append(
            pb.MachinePackage(
                installation_id=path.name,
                package=installed.package,
                release=installed.release,
                origin="runtime"
                if installed.package.startswith("runtime/")
                else "release"
                if path.name.startswith(PUBLISHED_PREFIX)
                else "local",
                installed_at_ms=installed_ms,
                sdk=[
                    pb.ImageDistribution(distribution=name, version=version)
                    for name, version in sorted(installed.sdk.items())
                ],
                entrypoints=_entrypoints(path / "package-interface.json"),
            )
        )
    rows.sort(key=lambda row: (row.package, row.release, row.installation_id))
    return pb.PackageList(packages=rows)


def _entrypoints(path: Path) -> list[str]:
    try:
        interface = package_interface.parse(path.read_bytes(), str(path))
    except (*_UNREADABLE, ConformanceError):
        return []
    return sorted({entry.name for _, entry in interface.callables() if not entry.internal})


# ------------------------------------------------------------------ models


class _Row(msgspec.Struct, frozen=True):
    """One `tfs repo list --rows` line."""

    kind: str
    org: str
    name: str
    manifest_sha256: str
    manifest_length: int
    version: str = ""
    lane: str = ""
    source_selection: str = ""


class _Usage(msgspec.Struct, frozen=True):
    """One `tfs repo usage --rows` line: a repository's, or (kind "store") the store's."""

    kind: str
    org: str = ""
    name: str = ""
    bytes_total: int = 0
    bytes_unique: int = 0
    bytes_unique_sum: int = 0
    bytes_unreferenced: int = 0


def models(worker: Worker) -> pb.ModelList:
    """The store's repository rows and usage, as `tfs` reports them, and which manifests the
    store holds complete."""
    root = worker.options.tensorfs_root
    if root is None:
        return pb.ModelList()
    rows = [msgspec.json.decode(line, type=_Row) for line in _tfs_rows("list", root)]
    usage = [msgspec.json.decode(line, type=_Usage) for line in _tfs_rows("usage", root)]
    complete = set(fill.open_store(root).complete_cozytensors_manifests())
    listing = pb.ModelList(
        models=[
            pb.MachineModel(
                kind=row.kind,
                repository=f"{row.org}/{row.name}",
                version=row.version,
                lane=row.lane,
                source_selection=row.source_selection,
                manifest=pb.Ref(
                    digest=bytes.fromhex(row.manifest_sha256), length=row.manifest_length
                ),
                complete="sha256:" + row.manifest_sha256 in complete,
            )
            for row in rows
        ],
        repositories=[
            pb.RepositoryUsage(
                repository=f"{row.org}/{row.name}",
                bytes_total=row.bytes_total,
                bytes_unique=row.bytes_unique,
            )
            for row in usage
            if row.kind == "repo"
        ],
        store=next(
            (
                pb.StoreUsage(
                    bytes_total=row.bytes_total,
                    bytes_unique_sum=row.bytes_unique_sum,
                    bytes_unreferenced=row.bytes_unreferenced,
                    filesystem=filesystem(root),
                )
                for row in usage
                if row.kind == "store"
            ),
            pb.StoreUsage(filesystem=filesystem(root)),
        ),
    )
    listing.models.sort(key=lambda row: (row.repository, row.version, row.lane))
    listing.repositories.sort(key=lambda row: row.repository)
    return listing


def _tfs_rows(verb: str, root: Path) -> list[bytes]:
    """`tfs repo <verb> <root> --rows`, from the TensorFS build this Runtime imports."""
    tfs = shutil.which("tfs", path=sysconfig.get_path("scripts"))
    if tfs is None:
        raise OSError("this Runtime's TensorFS has no tfs")
    with tempfile.TemporaryDirectory(prefix="cozy-tfs-rows-") as scratch:
        out = Path(scratch) / "rows.jsonl"
        answer = subprocess.run(
            [tfs, "repo", verb, str(root), "--rows", str(out)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            env={},  # the store is named; tfs needs nothing of this process's environment
        )
        if answer.returncode:
            raise OSError(f"tfs repo {verb} refused: {answer.stderr.strip()[-512:]}")
        return [line for line in out.read_bytes().splitlines() if line.strip()]


# ------------------------------------------------------------------ environment


def filesystem(path: Path) -> pb.MachineFilesystem:
    try:
        observed = os.statvfs(path)
    except OSError:
        return pb.MachineFilesystem(path=str(path))
    return pb.MachineFilesystem(
        path=str(path),
        total_bytes=observed.f_blocks * observed.f_frsize,
        available_bytes=observed.f_bavail * observed.f_frsize,
    )


def devices(worker: Worker) -> list[pb.MachineDevice]:
    return [
        pb.MachineDevice(
            ordinal=ordinal,
            name=gpu["device_name"],
            uuid=gpu["device_uuid"],
            memory_bytes=gpu["memory_bytes"],
            pci_bus_id=gpu["pci_bus_id"],
            driver_version=gpu["driver_version"],
        )
        for ordinal, gpu in enumerate(worker.options.gpus)
    ]


def accelerator_backend(worker: Worker) -> str:
    """A backend without a measured inventory (a flag-launched `serve`) is unknown."""
    options = worker.options
    return (
        options.accelerator_backend if options.gpus or options.accelerator_backend == "none" else ""
    )


def runtime(worker: Worker) -> pb.MachineRuntime:
    """This Runtime's half of `DescribeMachine`."""
    described = pb.MachineRuntime(
        version=_version("cozy-runtime"),
        wire_minor=WIRE_MINOR,
        minimum_wire_minor=MIN_COMPATIBLE_WIRE_MINOR,
        tensorfs_version=_version("tensorfs"),
        python_version=platform.python_version(),
        torch_version=_version("torch"),
        uv_version=_uv_version(),
        accelerator_backend=accelerator_backend(worker),
        executor_uid_isolation=worker.options.executor_uid >= 0,
        devices=devices(worker),
        resources=worker.resources(),
        execution_workspace_id=worker.executions.workspace_id if worker.executions else "",
        started_at_unix_ms=STARTED_MS,
    )
    if worker.options.tensorfs_root is not None:
        described.store.CopyFrom(filesystem(worker.options.tensorfs_root))
    try:
        found = python_interpreters.available(root=worker.config.managed_python_root)
    except python_interpreters.InterpreterRefusal:
        found = ()
    described.interpreters.extend(
        pb.PythonInterpreter(version=item.version, abi=item.abi) for item in found
    )
    return described


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _uv_version() -> str:
    uv = shutil.which("uv")
    if uv is None:
        return ""
    answer = subprocess.run(
        [uv, "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        env=package_install_environment(),
    )
    words = answer.stdout.split()
    return words[1] if answer.returncode == 0 and len(words) >= 2 else ""
