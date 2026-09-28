"""A root named by its published release, or by an unpublished installation this machine
already holds: this machine installs the published closure it lacks, reading install facts and
callees at its catalog, resolves every Model slot for its own devices, prepares the placement,
and mints the capture and offer a controller used to build. Nothing is journaled before the
receipt, so an interrupted preparation restarts on the next identical submission."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
from packaging.version import Version

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill, job_plan, package_installation, package_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import (
    machine_byte_results,
    machine_capture,
    machine_model_defaults,
    machine_model_resolve,
    machine_release_catalog,
    machine_source_models,
)
from .machine_capture import Installation
from .machine_child_target import Target
from .plan import JobBinding
from .servicer_context import ServicerContext
from .workspace import WorkspaceRefusal
from .workspace_calls import Call
from .workspace_executions import ExecutionWorkspaceRefusal, Receipt

if TYPE_CHECKING:
    from .session import Worker

#: A root's output bound, as the controller's own serving roots grant it.
MAX_OUTPUT_BYTES = 512 << 20
#: Finished preparations held for a submission that has not come back; oldest leave first.
MAX_HELD = 32
#: Release install facts held; each is immutable, the oldest leaves first.
MAX_FACTS = 128
#: A root's published callee closure.
MAX_CLOSURE = 128

#: A device job's resident-memory ceiling, as the controller's root jobs are given it.
JOB_RSS_CAP = 8 << 30
#: The accelerator half of the base-owned Python stack: requiring one needs a device.
ACCELERATOR_DISTRIBUTIONS = frozenset({"torch", "torchaudio", "torchvision", "triton"})
#: What the resolved record journals of each slot's selection.
RECORDED = (
    "parameter",
    "repository",
    "release",
    "lane",
    "gpus",
    "manifest",
    "source",
    "resolved",
    "profiles",
)


def _requirement(line: str) -> str:
    """A locked requirement's normalized distribution name."""
    name = re.match(r"[A-Za-z0-9._-]*", line.strip())
    return re.sub(r"[-_.]+", "-", name.group(0).lower()) if name else ""


class Preparing(Exception):
    """The release root is still being prepared; the message is its progress, and `moved` of
    `total` bytes have landed while its Models download."""

    def __init__(self, progress: str, moved: int = 0, total: int = 0) -> None:
        super().__init__(progress)
        self.moved, self.total = moved, total


@dataclass
class _Prepared:
    key: str  # package@release of the root's code
    target: Target
    declared: package_interface.CallableDoc
    placement: dict[str, Json]
    models: list[dict[str, Json]]
    installed: dict[str, Installation]
    edges: list[tuple[str, str]]
    placement_set: bytes = b""  # a served root's prepared placement, as its set's exact bytes


@dataclass
class _Job:
    request: bytes
    version: int = 0
    progress: str = "preparing"
    moved: int = 0
    total: int = 0
    noted: float = 0.0  # when a byte count last woke the waiting submitter
    done: bool = False
    result: _Prepared | None = None
    error: BaseException | None = None


def _key(package: str, release: str) -> str:
    return f"{package}@{release}"


class ReleaseRoots:
    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.changed = threading.Condition()
        self.jobs: dict[tuple[str, str], _Job] = {}
        # Install facts by release, read at this machine's catalog: immutable, read once.
        self.facts: dict[str, pb.DeferredInstallation] = {}

    def submit(
        self, owner: str, request: pb.MachineExecutionSubmit, context: ServicerContext
    ) -> pb.MachineExecutionReceipt:
        executions = self.worker.executions
        assert executions is not None
        request_id = request.offer.request_id
        receipt = executions.accepted(
            owner, request.submission_id, request_id, request.expected_execution_workspace_id
        )
        if receipt is not None:
            return self._receipt(receipt)
        identity = (owner, request.submission_id)
        asked = pb.MachineExecutionSubmit()
        asked.CopyFrom(request)
        asked.ClearField("claim")
        asked.ClearField("source_credentials")
        raw = asked.SerializeToString(deterministic=True)
        with self.changed:
            job = self.jobs.get(identity)
            if job is not None and job.request != raw:
                job = None  # the controller changed what it asks: prepare that instead
            if job is None:
                self._plan(request.release_root)
                job = _Job(raw)
                self.jobs[identity] = job
                held = [key for key, held in self.jobs.items() if held.done]
                for key in held[: max(0, len(self.jobs) - MAX_HELD)]:
                    del self.jobs[key]
                threading.Thread(
                    target=self._run,
                    args=(owner, request, job),
                    name=f"release-root-{request_id}"[:64],
                    daemon=True,
                ).start()
            self._wait(job, context)
            if not job.done:
                raise Preparing(job.progress, job.moved, job.total)
            del self.jobs[identity]
        if job.error is not None:
            raise job.error
        assert job.result is not None
        return self._accept(owner, request, job.result)

    def _wait(self, job: _Job, context: ServicerContext) -> None:
        """Until the job finishes or reports news, or the caller goes away."""
        gone = threading.Event()

        def leave() -> None:
            with self.changed:
                gone.set()
                self.changed.notify_all()

        if not context.add_callback(leave):
            leave()  # the RPC already ended: its callback will never run
        seen = job.version
        while not job.done and job.version == seen and not gone.is_set():
            self.changed.wait()

    def _note(self, job: _Job, progress: str, moved: int = 0, total: int = 0) -> None:
        """A new stage wakes the waiting submitter at once; a byte count, once a second."""
        with self.changed:
            now = time.monotonic()
            if progress == job.progress and (
                (moved, total) == (job.moved, job.total) or now - job.noted < 1
            ):
                return
            job.progress, job.moved, job.total, job.noted = progress, moved, total, now
            job.version += 1
            self.changed.notify_all()

    def _installed(self, key: str) -> Installation | None:
        """The placement this machine holds for a release with the fewest bound Models."""
        package, _, release = key.partition("@")
        with self.worker.control_lock:
            held = [
                placement
                for placement in self.worker.prepared_installations.values()
                if placement.document.HasField("package")
                and placement.document.package.package == package
                and placement.document.package.release == release
            ]
        if not held:
            return None
        selected = min(held, key=lambda placement: len(placement.document.models))
        return Installation(documents.body(selected.document), selected.installation_id)

    def _plan(self, root: pb.ReleaseRoot) -> None:
        if (
            not root.package
            or not root.entrypoint.isidentifier()
            or bool(root.release) == bool(root.installation_id)
        ):
            raise WorkspaceRefusal(
                "release root names no package, entrypoint and one release or installation"
            )

    def _root(self, root: pb.ReleaseRoot) -> tuple[str, Installation | None]:
        """The root's package@release, and the unpublished installation it names, which this
        machine must already hold: the controller prepares it and asks again."""
        if not root.installation_id:
            return _key(root.package, root.release), None
        with self.worker.control_lock:
            held = next(
                (
                    placement
                    for placement in self.worker.prepared_installations.values()
                    if placement.installation_id == root.installation_id
                    and placement.document.HasField("development")
                    and placement.document.development.package == root.package
                ),
                None,
            )
        if held is None:
            raise ExecutionWorkspaceRefusal(
                "release_root_installation_absent",
                f"this machine holds no installation {root.installation_id} of {root.package}",
            )
        return _key(root.package, held.document.development.release), Installation(
            documents.body(held.document), held.installation_id
        )

    def _facts(self, key: str) -> pb.DeferredInstallation:
        """A release's install facts, read at this machine's catalog once."""
        with self.changed:
            held = self.facts.get(key)
        if held is None:
            package, _, release = key.partition("@")
            held = machine_release_catalog.facts(
                machine_model_resolve.own(self.worker), package, release
            )
            with self.changed:
                self.facts[key] = held
                for old in list(self.facts)[: max(0, len(self.facts) - MAX_FACTS)]:
                    del self.facts[old]
        return held

    def _locked(self, key: str, held: Installation | None, captured: bool = False) -> bytes:
        """A release's lock: its installation's here, else its install facts'. A captured
        installation has no release at the Hub: only its own lock, if it has one."""
        if held is not None:
            path = (
                Path(self.worker.options.install_root or "")
                / "installations"
                / held.installation_id
                / "requirements.txt"
            )
            if path.is_file():
                return path.read_bytes()
        return b"" if captured else self._facts(key).preparation.locked_requirements

    def _closure(
        self, root: str, captured: Installation | None
    ) -> tuple[dict[str, Installation], list[tuple[str, str]]]:
        """The releases of the root's closure this machine holds, and the closure's edges,
        each release's callees read from its own lock (a captured root's from its own)."""
        installed: dict[str, Installation] = {}
        edges: set[tuple[str, str]] = set()
        pending, seen = [root], set[str]()
        while pending:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > MAX_CLOSURE:
                raise WorkspaceRefusal("published callable closure exceeds its bound")
            own = key == root and captured is not None
            held = captured if own else self._installed(key)
            if held is not None:
                installed[key] = held
            package = key.partition("@")[0]
            lock = self._locked(key, held, own)
            for locked in machine_release_catalog.callees(package, lock):
                callee = self._served(locked)
                edges.add((key, callee))
                pending.append(callee)
        return installed, sorted(edges)

    def _served(self, locked: str) -> str:
        """A callee release a caller's lock names is a floor, not a pin: the newest release
        this machine's catalog publishes at or above it serves the call, read once per
        package until the controller that publishes names it to `forget`."""
        package, _, release = locked.partition("@")
        newest = self.worker.resolutions.newest
        if package not in newest:
            try:
                newest[package] = machine_release_catalog.newest(
                    machine_model_resolve.own(self.worker), package
                )
            except WorkspaceRefusal:
                return locked
        return (
            _key(package, newest[package])
            if Version(newest[package]) > Version(release)
            else locked
        )

    def describe(self, selection: pb.PackageSelection) -> pb.DescribedRelease:
        """A published release as this machine reads it: the newest when none is named, read
        once until the controller that publishes names the package to `forget`."""
        package, release = selection.package, selection.release
        newest = self.worker.resolutions.newest
        if not release:
            release = newest.get(package) or machine_release_catalog.newest(
                machine_model_resolve.own(self.worker), package
            )
            newest[package] = release
        key = _key(package, release)
        held = self._installed(key)
        raw = held.interface if held is not None else self._facts(key).preparation.package_interface
        return pb.DescribedRelease(package=package, release=release, package_interface=raw)

    def _run(self, owner: str, request: pb.MachineExecutionSubmit, job: _Job) -> None:
        try:
            job.result = self._prepare(owner, request, job)
        except BaseException as exc:
            job.error = (
                exc
                if isinstance(exc, WorkspaceRefusal)
                else WorkspaceRefusal(f"release root preparation failed: {exc}"[:1024])
            )
        finally:
            with self.changed:
                job.done = True
                job.version += 1
                self.changed.notify_all()

    def _install(self, row: pb.DeferredInstallation) -> Installation:
        from .session import read_placement_set  # session imports this module's RPC

        prepared = pb.PreparePackageSetRequest()
        prepared.CopyFrom(row.preparation)
        prepared.install_root = str(Path(self.worker.options.install_root or ""))
        entries = read_placement_set(self.worker.prepare_package_set(prepared).placement_set)
        if len(entries) != 1 or entries[0].package.package != row.package:
            raise WorkspaceRefusal("release root preparation returned another installation")
        return Installation(documents.body(entries[0]), entries[0].installation_id)

    def _prepare(self, owner: str, request: pb.MachineExecutionSubmit, job: _Job) -> _Prepared:
        root, worker = request.release_root, self.worker
        assert worker.machine_calls is not None
        key, captured = self._root(root)
        installed, edges = self._closure(key, captured)
        if key not in installed:
            self._note(job, f"installing {key}")
            installed[key] = self._install(self._facts(key))
        prepared = installed[key]
        body = package_interface.read_bytes(prepared.interface, "release root")
        section, kind = ("jobs", "job") if root.job else ("entrypoints", "inference entrypoint")
        declaration = next(
            (entry for entry in body[section] if entry["name"] == root.entrypoint), None
        )
        if declaration is None:
            raise WorkspaceRefusal(f"{key} serves no {kind} {root.entrypoint}")
        declared = msgspec.convert(declaration, package_interface.CallableDoc)
        if declared.internal:
            raise WorkspaceRefusal("internal_callable: an internal callable cannot be a root")
        binding = None
        if root.job:
            descriptor = package_interface.job_descriptor_id(body, root.entrypoint)
            plan = job_plan.path(
                worker.config.cozy_home / "job-plans", prepared.installation_id, descriptor
            )
            binding = JobBinding.read(json.loads(plan.read_bytes()))
        target = Target(
            prepared.installation_id,
            root.entrypoint,
            declaration,
            msgspec.to_builtins(prepared),
            binding,
            machine_capture.operation_identity(declared),
            False,
            prepared.installation_id,
        )
        models = self._resolve(
            root,
            declaration,
            machine_source_models.credentials(request.source_credentials),
            lambda stage, done, total: self._note(job, stage, done, total),
        )
        if root.job:
            # A job reads its Models as inputs: land each catalog checkpoint here now.
            machine_model_defaults.materialize(
                worker,
                models,
                None,
                progress=lambda done, total: self._note(
                    job, "downloading model weights", done, total
                ),
            )
            return _Prepared(
                key, target, declared, dict(prepared.placement), models, installed, edges
            )
        serving = worker.machine_calls.serving
        call = Call(
            request.offer.request_id,
            (1 << 32) - 1,
            0,
            0,
            b"",
            b"",
            "release-root." + uuid.uuid4().hex,
            b"",
            False,
            b"",
            "",
            "",
        )
        serving.observers[call.child_request] = lambda frame: self._progress(job, frame)
        try:
            ready = serving._prepare(
                owner,
                call,
                target,
                pb.ChildCallRequest(),
                model_arguments={},
                defaults=models,
                retention_prefix="preparation/",
            )
            ready.access.close()
        finally:
            serving.observers.pop(call.child_request, None)
            serving._release_preparation(owner, call.child_request)
        placement_set = canonical_json.encode(
            {"format": "cozy.worker.v1.PlacementSet/1", "placements": [ready.placement]}
        )
        self._register(target, ready.placement, placement_set)
        return _Prepared(
            key, target, declared, ready.placement, models, installed, edges, placement_set
        )

    def _resolve(
        self,
        root: pb.ReleaseRoot,
        declaration: Mapping[str, Json],
        credentials: Mapping[str, str],
        note: machine_source_models.Note,
    ) -> list[dict[str, Json]]:
        """Each slot for this machine. A manifest it already holds needs no catalog read; the
        package's bindings are read once and its catalog slots then resolve beside each other,
        once per package."""
        slots = declaration.get("models", [])
        if not slots:
            return []
        given = {choice.parameter: choice for choice in root.models}
        pins = {
            parameter: self._pin(choice)
            for parameter, choice in given.items()
            if choice.HasField("manifest")
        }
        parameters = [str(model["path"]).rpartition(".")[2] for model in slots]
        sourced = {
            parameter for parameter in parameters if parameter in given and given[parameter].source
        }
        open_slots = [
            (model, parameter)
            for model, parameter in zip(slots, parameters, strict=True)
            if parameter not in sourced and not (parameter in pins and pins[parameter]["length"])
        ]
        # Pinned and sourced slots read nothing; an open one reads this machine's own Hub.
        catalog = (
            machine_model_resolve.own(self.worker, root.package)
            if open_slots
            else machine_model_resolve.Catalog("")
        )
        fits = machine_model_defaults.catalog_fits(machine_model_defaults.fitter(self.worker))
        laddered = [
            model
            for model, parameter in open_slots
            if parameter not in pins and not (parameter in given and given[parameter].repository)
        ]
        if laddered and not root.package.startswith("local/"):
            catalog.binding(root.package, str(laddered[0]["path"]))
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            resolving = {
                parameter: pool.submit(
                    machine_model_resolve.slot,
                    catalog,
                    root.package,
                    model,
                    parameter,
                    _catalog_choice(given.get(parameter), pins.get(parameter)),
                    fits,
                    root.owner,
                )
                for model, parameter in zip(slots, parameters, strict=True)
                if parameter not in sourced
            }
            resolved = {parameter: row.result() for parameter, row in resolving.items()}
        # A provider source is made here, in order: headers first, only its profile's members.
        return [
            machine_source_models.slot(self.worker, given[parameter], credentials, note)
            if parameter in sourced
            else resolved[parameter]
            for parameter in parameters
        ]

    def _pin(self, choice: pb.ModelChoice) -> dict[str, Json]:
        """A pinned manifest, its length read from this machine's store when unstated."""
        if len(choice.manifest.digest) != 32:
            raise WorkspaceRefusal(f"{choice.parameter} pins a manifest with no exact digest")
        digest = documents.spell(choice.manifest.digest)
        return {"digest": digest, "length": choice.manifest.length or self._held_length(digest)}

    def _held_length(self, digest: str) -> int:
        """The length of a manifest this machine's store holds, else 0 (the catalog says)."""
        root = self.worker.options.tensorfs_root
        try:
            raw = bytes(fill.store(root).manifest(digest)["manifest"]) if root else b""
        except Exception:  # not held here
            return 0
        return len(raw) if raw and documents.spell(hashlib.sha256(raw).digest()) == digest else 0

    def _progress(self, job: _Job, frame: Mapping[str, object]) -> None:
        position, total, stage = frame.get("position"), frame.get("total"), frame.get("stage")
        if isinstance(position, int) and isinstance(total, int) and total > 0:
            self._note(job, "downloading model weights", position, total)
        elif (
            frame.get("kind") == "progress"
            and isinstance(stage, str)
            and ("Downloading" in stage or "compatibility" in stage)
        ):
            self._note(job, stage)

    def _register(self, target: Target, placement: dict[str, Json], placement_set: bytes) -> None:
        """Keep the prepared placement, so the next identical root matches it at once."""
        worker = self.worker
        stored = worker._placement_from_entry(
            documents.from_body(placement, pb.Placement), hashlib.sha256(placement_set).digest()
        )
        stored.installed = package_installation.open_installation(
            Path(worker.options.install_root or ""), target.installation_id
        )
        with worker.control_lock:
            worker.prepared_installations[stored.prepared_key] = stored

    def _capture(self, prepared: _Prepared) -> tuple[bytes, bytes, dict[str, Installation]]:
        """The callable inventory this machine would have been sent: installed and deferred."""
        installed = prepared.installed
        names: dict[str, tuple[str, str]] = {}  # key -> (field prefix, capture name)
        interfaces: dict[str, package_interface.PackageInterface] = {}
        packages: list[pb.InstalledPackage] = []
        rows: list[pb.DeferredInstallation] = []
        keys = {prepared.key}
        keys.update(key for edge in prepared.edges for key in edge)
        for key in sorted(keys):
            held = installed.get(key)
            if held is not None:
                raw = held.interface
                package, _, release = key.partition("@")
                packages.append(
                    pb.InstalledPackage(
                        installation_id=held.installation_id,
                        package=package,
                        release=release,
                        package_interface=raw,
                    )
                )
                names[key] = ("installation_id", held.installation_id)
                interfaces[key] = package_interface.parse(raw, "installed release")
            else:
                row = self._facts(key)
                rows.append(row)
                names[key] = ("deferred_key", key)
                interfaces[key] = package_interface.parse(
                    row.preparation.package_interface, "deferred release"
                )
        bindings: list[pb.MachineCallableBinding] = []

        def bind(caller: str, callee: str, own: bool) -> None:
            body = interfaces[callee]
            for entry in (*body.jobs, *body.entrypoints):
                invocable = entry.invocable
                if invocable is msgspec.UNSET or (entry.internal and not own):
                    continue
                binding = pb.MachineCallableBinding(
                    module=invocable.module, export=invocable.export, entrypoint=entry.name
                )
                setattr(binding, "caller_" + names[caller][0], names[caller][1])
                setattr(binding, "callee_" + names[callee][0], names[callee][1])
                bindings.append(binding)

        bind(prepared.key, prepared.key, True)
        for edge in prepared.edges:
            bind(*edge, False)
        bindings.sort(
            key=lambda row: (
                row.caller_installation_id or row.caller_deferred_key,
                row.module,
                row.export,
            )
        )
        packages.sort(key=lambda row: row.installation_id)
        capture = pb.MachineExecutionCapture(
            root_installation_id=prepared.target.installation_id,
            installed_packages=packages,
            deferred_installations=rows,
            bindings=bindings,
        )
        raw, digest = documents.identity(capture)
        return raw, digest, {key: held for key, held in installed.items() if key in keys}

    def _accept(
        self, owner: str, request: pb.MachineExecutionSubmit, prepared: _Prepared
    ) -> pb.MachineExecutionReceipt:
        worker, root = self.worker, request.release_root
        executions = worker.executions
        assert executions is not None
        placement = prepared.placement
        payload = request.payload_canonical_bytes
        digest = documents.spell(hashlib.sha256(payload).digest())
        bindings, access = machine_model_defaults.inputs(prepared.models)
        spec = pb.InvocationSpec(
            installation_id=prepared.target.installation_id,
            payload_digest=digest,
            inputs=[
                pb.InputBinding(
                    input_id="payload",
                    digest=digest,
                    length=len(payload),
                    kind_mime="application/json",
                ),
                *root.inputs,
                *bindings,
            ],
            outputs=[
                pb.OutputBinding(output_id=path, max_bytes=bound)
                for path, bound in machine_byte_results.paths(
                    prepared.target.declaration["result"], bound=MAX_OUTPUT_BYTES
                )
            ],
            deadline_unix_ms=root.deadline_unix_ms,
            attention_kernel=root.attention_kernel,
        )
        if root.job:
            desired, placement_id = self._job(root, prepared, spec), ""
        else:
            served = documents.parse(prepared.placement_set, pb.PlacementSet).placements[0]
            desired, placement_id = self._serving(root, prepared, served, spec), served.placement_id
        spec.inputs.sort(key=lambda entry: entry.input_id)
        raw, identity = documents.identity(spec)
        offer = pb.AttemptOffer(
            request_id=request.offer.request_id,
            attempt_ordinal=1,
            placement_id=placement_id,
            invocation_spec_canonical_bytes=raw,
            invocation_spec_digest=identity,
            grant=pb.DeliveryGrant(
                invocation_spec_digest=identity,
                inputs=[
                    pb.InputAccess(
                        input_id="payload",
                        url="data:application/json;base64," + base64.b64encode(payload).decode(),
                    ),
                    *root.input_access,
                    *access,
                ],
                outputs=[pb.OutputAccess(output_id=output.output_id) for output in spec.outputs],
            ),
        )
        weights = {row.output_id for row in _weights_outputs(prepared.declared)}
        for output in offer.grant.outputs:
            if root.weights_destination and output.output_id in weights:
                # `--upload-to`: Runtime sends this weights output to the named repository.
                output.url = "model://" + root.weights_destination
        offer.grant.inputs.sort(key=lambda entry: entry.input_id)
        capture, capture_digest, installed = self._capture(prepared)
        installations = {
            held.installation_id: Installation(
                placement
                if held.installation_id == prepared.target.installation_id
                else held.placement,
                held.installation_id,
            )
            for held in installed.values()
        }
        assert worker.machine_calls is not None
        preparation = machine_capture.Preparation(
            installations=installations,
            state=base64.b64encode(desired.SerializeToString(deterministic=True)).decode(),
            runtime_builtin=worker.machine_calls.builtins.capture(),
        )
        worker.machine_sources.hold(owner, offer.request_id, request.source_credentials)
        receipt = worker.submit_execution(
            request.claim,
            request.submission_id,
            capture_digest,
            offer,
            expected_execution_workspace_id=request.expected_execution_workspace_id,
            capture_document=capture,
            preparation=preparation.encode(),
            publication_authorization_id=request.publication_authorization_id,
            arguments=canonical_json.decode(payload) if root.job else None,
            owner_memo=request.owner_memo,
        )
        worker.machine_sources.declare(owner, offer.request_id)
        executions.record(
            owner,
            offer.request_id,
            "resolved",
            {
                "package": root.package,
                "release": prepared.key.partition("@")[2],
                "installation_id": prepared.target.installation_id,
                "models": [
                    {name: row[name] for name in RECORDED if name in row} for row in prepared.models
                ],
            },
        )
        return self._receipt(receipt)

    def _serving(
        self,
        root: pb.ReleaseRoot,
        prepared: _Prepared,
        served: pb.Placement,
        spec: pb.InvocationSpec,
    ) -> pb.DesiredWorkerState:
        """The serving invocation of the prepared placement, and that placement as state."""
        selected = next((row for row in served.entrypoints if row.name == root.entrypoint), None)
        if selected is None:
            raise WorkspaceRefusal("release root has no complete prepared Model binding")
        binding = documents.spell(selected.entrypoint_binding_digest)
        spec.serving.CopyFrom(
            pb.ServingInvocationSpec(
                entrypoint_binding_digest=binding,
                attempt_binding_id=binding,
                bindings_digest=documents.spell(served.bindings_digest),
            )
        )
        if root.HasField("capture"):
            spec.capture.CopyFrom(root.capture)
            spec.outputs.append(pb.OutputBinding(output_id="runtime.capture", max_bytes=64 << 20))
        width = max((row.gpus for row in machine_model_defaults.decode(prepared.models)), default=0)
        return pb.DesiredWorkerState(
            wire_minor=self.worker.fence.wire_minor,
            posture=pb.POSTURE_ACCEPTING,
            placement_set=pb.DesiredPlacementSet(
                placement_set_digest=hashlib.sha256(prepared.placement_set).digest(),
                placement_set_canonical_bytes=prepared.placement_set,
                execution_gpus=width,
            ),
        )

    def _job(
        self, root: pb.ReleaseRoot, prepared: _Prepared, spec: pb.InvocationSpec
    ) -> pb.DesiredWorkerState:
        """The job invocation and its directive, by the rules a controller used to apply."""
        target = prepared.target
        assert target.binding is not None
        spec.outputs.extend(
            pb.OutputBinding(
                output_id=row.output_id, mime_type=row.mime_type, max_bytes=row.max_bytes
            )
            for row in _weights_outputs(prepared.declared)
        )
        spec.job.CopyFrom(
            pb.JobInvocationSpec(
                installation_id=target.installation_id,
                job_descriptor_id=target.binding.job_descriptor_id,
            )
        )
        if root.publication_grant:
            spec.job.publication_contract.CopyFrom(
                pb.PublicationContract(grant_id=root.publication_grant, outputs=spec.outputs)
            )
        device = self._needs_device(root, prepared)
        return pb.DesiredWorkerState(
            wire_minor=self.worker.fence.wire_minor,
            posture=pb.POSTURE_ACCEPTING,
            job=pb.JobDirective(
                installation_id=target.installation_id,
                job_descriptor_id=target.binding.job_descriptor_id,
                resource_caps=pb.ResourceCaps(
                    device_required=device, max_rss_bytes=JOB_RSS_CAP if device else 0
                ),
                device_count=1 if device else 0,
                orchestration=not device,
            ),
        )

    def _needs_device(self, root: pb.ReleaseRoot, prepared: _Prepared) -> bool:
        """A job's own `accelerator` decides; an undeclared one follows its locked closure,
        except a composer that holds no Model or weights and only calls published callees."""
        target, declared = prepared.target, prepared.declared
        if declared.accelerator is not msgspec.UNSET:
            return declared.accelerator
        locked = (
            Path(self.worker.options.install_root or "")
            / "installations"
            / target.installation_id
            / "requirements.txt"
        )
        try:
            names = {_requirement(line) for line in locked.read_text().splitlines()}
        except FileNotFoundError:
            names = set()
        if not names & ACCELERATOR_DISTRIBUTIONS:
            return False
        composes = any(caller == _key(root.package, root.release) for caller, _ in prepared.edges)
        return not (composes and not declared.models and not _weights_outputs(declared))

    @staticmethod
    def _receipt(receipt: Receipt) -> pb.MachineExecutionReceipt:
        return pb.MachineExecutionReceipt(**asdict(receipt))


def _weights_outputs(
    declared: package_interface.CallableDoc,
) -> tuple[package_interface.WeightsOutputDeclaration, ...]:
    return () if declared.weights_outputs is msgspec.UNSET else declared.weights_outputs


def _catalog_choice(
    choice: pb.ModelChoice | None, pin: dict[str, Json] | None
) -> dict[str, Json] | None:
    """A catalog choice as the resolver reads it: its checkpoint names and any pinned manifest."""
    if choice is None:
        return None
    row: dict[str, Json] = {
        "repository": choice.repository,
        "release": choice.release,
        "lane": choice.lane,
    }
    if pin is not None:
        row["manifest"] = pin
    return row
