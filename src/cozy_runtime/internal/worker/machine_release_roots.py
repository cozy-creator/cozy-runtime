"""A root named by its published release: this machine installs it when it holds no
installation, resolves every Model slot for its own devices, prepares the placement, and
mints the capture and offer a controller used to build. Nothing is journaled before the
receipt, so an interrupted preparation restarts on the next identical submission."""

from __future__ import annotations

import base64
import hashlib
import threading
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.protobuf import json_format

from cozy_runtime import canonical_json
from cozy_runtime.internal import package_installation, package_interface
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import (
    machine_byte_results,
    machine_deferred,
    machine_model_defaults,
    machine_model_resolve,
    machine_source_models,
)
from .machine_child_target import Target
from .workspace import WorkspaceRefusal
from .workspace_calls import Call
from .workspace_executions import Receipt

if TYPE_CHECKING:
    from .session import Worker

#: A root's output bound, as the controller's own serving roots grant it.
MAX_OUTPUT_BYTES = 512 << 20
#: Finished preparations held for a submission that has not come back; oldest leave first.
MAX_HELD = 32


class Preparing(Exception):
    """The release root is still being prepared; the message is its progress."""


class InstallationsAbsent(WorkspaceRefusal):
    """These releases have no installation here and no install facts rode the submission."""

    def __init__(self, keys: list[str]) -> None:
        super().__init__("install facts are absent for " + ", ".join(keys))
        self.keys = keys


@dataclass
class _Prepared:
    target: Target
    placement: dict[str, Any]
    models: list[dict[str, Any]]
    installed: dict[str, dict[str, Any]]


@dataclass
class _Job:
    request: bytes
    version: int = 0
    progress: str = "preparing"
    percent: int = -1
    done: bool = False
    result: _Prepared | None = None
    error: BaseException | None = None
    installed: dict[str, dict[str, Any]] = field(default_factory=dict)


def _key(package: str, release: str) -> str:
    return f"{package}@{release}"


class ReleaseRoots:
    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.changed = threading.Condition()
        self.jobs: dict[tuple[str, str], _Job] = {}
        # Install facts a submission carried, by release: one immutable document each, so a
        # later root naming an uninstalled callee need not carry them again.
        self.facts: dict[str, pb.DeferredInstallation] = {}

    def submit(
        self, owner: str, request: pb.MachineExecutionSubmit, context: Any
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
        asked.release_root.ClearField("installations")  # remembered by release, below
        raw = asked.SerializeToString(deterministic=True)
        with self.changed:
            job = self.jobs.get(identity)
            if job is not None and job.request != raw:
                job = None  # the controller changed what it asks: prepare that instead
            if job is None:
                job = _Job(raw, installed=self._plan(request.release_root))
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
                raise Preparing(job.progress)
            del self.jobs[identity]
        if job.error is not None:
            raise job.error
        assert job.result is not None
        return self._accept(owner, request, job.result)

    def _wait(self, job: _Job, context: Any) -> None:
        """Until the job finishes or reports news, or the caller goes away."""
        gone = threading.Event()

        def leave() -> None:
            with self.changed:
                gone.set()
                self.changed.notify_all()

        if hasattr(context, "add_callback"):
            context.add_callback(leave)
        seen = job.version
        while not job.done and job.version == seen and not gone.is_set():
            self.changed.wait()

    def _note(self, job: _Job, progress: str, percent: int = -1) -> None:
        with self.changed:
            if progress == job.progress and percent == job.percent:
                return
            job.progress, job.percent = progress, percent
            job.version += 1
            self.changed.notify_all()

    def _installed(self, key: str) -> dict[str, Any] | None:
        """The placement this machine holds for a release with the fewest bound Models."""
        package, _, release = key.partition("@")
        with self.worker.control_lock:
            held = [
                placement
                for placement in self.worker.prepared_installations.values()
                if not placement.development
                and placement.package.get("package") == package
                and placement.package.get("release") == release
            ]
        if not held:
            return None
        selected = min(held, key=lambda placement: len(placement.models))
        return {"placement": dict(selected.document), "installation_id": selected.installation_id}

    def _plan(self, root: pb.ReleaseRoot) -> dict[str, dict[str, Any]]:
        """Every release the closure names that this machine holds; refuse what has no facts."""
        if not root.package or not root.release or not root.entrypoint.isidentifier():
            raise WorkspaceRefusal("release root names no package release and entrypoint")
        local = bool(self.worker.config.bootstrap_credential)
        if root.catalog_origin and not machine_model_defaults.valid_origin(
            root.catalog_origin, local=local
        ):
            raise WorkspaceRefusal("release root names an invalid catalog origin")
        for row in root.installations:
            if row.key != _key(row.package, row.release):
                raise WorkspaceRefusal("release root misnames install facts")
            with self.changed:
                self.facts[row.key] = row
        keys = {_key(root.package, root.release)}
        keys.update(key for edge in root.callees for key in (edge.caller, edge.callee))
        installed = {key: held for key in keys if (held := self._installed(key)) is not None}
        with self.changed:
            absent = sorted(keys - installed.keys() - self.facts.keys())
        if absent:
            raise InstallationsAbsent(absent)
        return installed

    def _facts(self, key: str) -> pb.DeferredInstallation:
        with self.changed:
            return self.facts[key]

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

    def _install(self, row: pb.DeferredInstallation) -> dict[str, Any]:
        from .session import read_placement_set  # session imports this module's RPC

        prepared = pb.PreparePackageSetRequest()
        prepared.CopyFrom(row.preparation)
        prepared.install_root = str(Path(self.worker.options.install_root or ""))
        entries = read_placement_set(self.worker.prepare_package_set(prepared).placement_set)
        if len(entries) != 1 or entries[0].get("package", {}).get("package") != row.package:
            raise WorkspaceRefusal("release root preparation returned another installation")
        return {"placement": entries[0], "installation_id": entries[0]["installation_id"]}

    def _prepare(self, owner: str, request: pb.MachineExecutionSubmit, job: _Job) -> _Prepared:
        root, worker = request.release_root, self.worker
        assert worker.machine_calls is not None
        key = _key(root.package, root.release)
        installed = dict(job.installed)
        if key not in installed:
            self._note(job, f"installing {key}")
            installed[key] = self._install(self._facts(key))
        prepared = installed[key]
        interface = package_interface.read_bytes(
            base64.b64decode(prepared["placement"]["package_interface"], validate=True),
            "release root",
        )
        declaration = next(
            (entry for entry in interface["entrypoints"] if entry["name"] == root.entrypoint), None
        )
        if declaration is None:
            raise WorkspaceRefusal(f"{key} serves no inference entrypoint {root.entrypoint}")
        if declaration.get("internal", False):
            raise WorkspaceRefusal("internal_callable: an internal callable cannot be a root")
        target = Target(
            prepared["installation_id"],
            root.entrypoint,
            declaration,
            prepared,
            None,
            str(declaration.get("invocable", {}).get("operation_identity", "")),
            False,
            prepared["installation_id"],
        )
        models = self._resolve(
            root,
            declaration,
            machine_source_models.credentials(request.source_credentials),
            lambda stage, done, total: self._note(job, stage, 100 * done // total if total else -1),
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
        self._register(target, ready.placement)
        return _Prepared(target, ready.placement, models, installed)

    def _resolve(
        self,
        root: pb.ReleaseRoot,
        declaration: Mapping[str, Any],
        credentials: Mapping[str, str],
        note: machine_source_models.Note,
    ) -> list[dict[str, Any]]:
        slots = declaration.get("models", [])
        if not slots:
            return []
        choices = {
            choice.parameter: json_format.MessageToDict(choice, preserving_proto_field_name=True)
            for choice in root.models
        }
        pinned = {
            parameter
            for parameter, choice in choices.items()
            if choice.get("source") or choice.get("manifest", {}).get("length")
        }
        if not root.catalog_origin and any(
            model["path"].rpartition(".")[2] not in pinned for model in slots
        ):
            raise WorkspaceRefusal("release root needs a catalog origin to resolve its Models")
        catalog = machine_model_resolve.Catalog(root.catalog_origin)
        fits = machine_model_defaults.fitter(self.worker)
        rows = []
        for model in slots:
            parameter = model["path"].rpartition(".")[2]
            choice = choices.get(parameter)
            if choice and choice.get("source"):
                # A provider source is made here: headers first, only its profile's members.
                rows.append(
                    machine_source_models.slot(self.worker, parameter, choice, credentials, note)
                )
                continue
            if choice and "manifest" in choice:
                choice["manifest"] = {
                    "digest": documents.spell(base64.b64decode(choice["manifest"]["digest"])),
                    "length": int(choice["manifest"].get("length", 0)),
                }
            rows.append(
                machine_model_resolve.slot(catalog, root.package, model, parameter, choice, fits)
            )
        return rows

    def _progress(self, job: _Job, frame: Mapping[str, Any]) -> None:
        position, total = frame.get("position"), frame.get("total")
        if isinstance(position, int) and isinstance(total, int) and total > 0:
            self._note(job, "downloading model weights", 100 * position // total)
        elif frame.get("kind") == "progress" and frame.get("stage"):
            stage = str(frame["stage"])
            if "Downloading" in stage or "compatibility" in stage:
                self._note(job, stage)

    def _register(self, target: Target, placement: dict[str, Any]) -> None:
        """Keep the prepared placement, so the next identical root matches it at once."""
        worker = self.worker
        raw = canonical_json.encode(
            {"format": "cozy.worker.v1.PlacementSet/1", "placements": [placement]}
        )
        stored = worker._placement_from_entry(placement, hashlib.sha256(raw).digest())
        stored.installed = package_installation.open_installation(
            Path(worker.options.install_root or ""), target.installation_id
        )
        with worker.control_lock:
            worker.prepared_installations[stored.prepared_key] = stored

    def _capture(
        self, root: pb.ReleaseRoot, prepared: _Prepared
    ) -> tuple[bytes, bytes, dict[str, Any]]:
        """The callable inventory this machine would have been sent: installed and deferred."""
        installed = prepared.installed
        names: dict[str, tuple[str, str]] = {}  # key -> (field prefix, capture name)
        interfaces: dict[str, dict[str, Any]] = {}
        packages: list[pb.InstalledPackage] = []
        rows: list[pb.DeferredInstallation] = []
        keys = {_key(root.package, root.release)}
        keys.update(key for edge in root.callees for key in (edge.caller, edge.callee))
        for key in sorted(keys):
            held = installed.get(key)
            if held is not None:
                raw = base64.b64decode(held["placement"]["package_interface"], validate=True)
                package, _, release = key.partition("@")
                packages.append(
                    pb.InstalledPackage(
                        installation_id=held["installation_id"],
                        package=package,
                        release=release,
                        package_interface=raw,
                    )
                )
                names[key] = ("installation_id", held["installation_id"])
                interfaces[key] = package_interface.read_bytes(raw, "installed release")
            else:
                row = self._facts(key)
                rows.append(row)
                names[key] = ("deferred_key", key)
                interfaces[key] = machine_deferred.interface(
                    json_format.MessageToDict(row, preserving_proto_field_name=True)
                )
        bindings: list[pb.MachineCallableBinding] = []

        def bind(caller: str, callee: str, own: bool) -> None:
            body = interfaces[callee]
            for entry in [*body.get("jobs", []), *body.get("entrypoints", [])]:
                invocable = entry.get("invocable")
                if not invocable or (entry.get("internal", False) and not own):
                    continue
                binding = pb.MachineCallableBinding(
                    module=invocable["module"], export=invocable["export"], entrypoint=entry["name"]
                )
                setattr(binding, "caller_" + names[caller][0], names[caller][1])
                setattr(binding, "callee_" + names[callee][0], names[callee][1])
                bindings.append(binding)

        root_key = _key(root.package, root.release)
        bind(root_key, root_key, True)
        for edge in sorted({(edge.caller, edge.callee) for edge in root.callees}):
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
            catalog_origin=root.catalog_origin,
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
        selected = next(
            (row for row in placement["entrypoints"] if row["name"] == root.entrypoint), None
        )
        if selected is None:
            raise WorkspaceRefusal("release root has no complete prepared Model binding")
        payload = request.payload_canonical_bytes
        digest = documents.spell(hashlib.sha256(payload).digest())
        bindings, access = machine_model_defaults.inputs(prepared.models)
        spec = pb.InvocationSpec(
            installation_id=placement["installation_id"],
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
            serving=pb.ServingInvocationSpec(
                entrypoint_binding_digest=selected["entrypoint_binding_digest"],
                attempt_binding_id=selected["entrypoint_binding_digest"],
                bindings_digest=placement["bindings_digest"],
            ),
        )
        if root.HasField("capture"):
            spec.capture.CopyFrom(root.capture)
            spec.outputs.append(pb.OutputBinding(output_id="runtime.capture", max_bytes=64 << 20))
        spec.inputs.sort(key=lambda entry: entry.input_id)
        raw, identity = documents.identity(spec)
        offer = pb.AttemptOffer(
            request_id=request.offer.request_id,
            attempt_ordinal=1,
            placement_id=placement["placement_id"],
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
        offer.grant.inputs.sort(key=lambda entry: entry.input_id)
        placement_bytes = canonical_json.encode(
            {"format": "cozy.worker.v1.PlacementSet/1", "placements": [placement]}
        )
        width = max((int(row.get("gpus", 0)) for row in prepared.models), default=0)
        desired = pb.DesiredWorkerState(
            wire_minor=worker.fence.wire_minor,
            posture=pb.POSTURE_ACCEPTING,
            placement_set=pb.DesiredPlacementSet(
                placement_set_digest=hashlib.sha256(placement_bytes).digest(),
                placement_set_canonical_bytes=placement_bytes,
                execution_gpus=width,
            ),
        )
        capture, capture_digest, installed = self._capture(root, prepared)
        installations = {
            held["installation_id"]: {
                "placement": placement
                if held["installation_id"] == prepared.target.installation_id
                else held["placement"],
                "installation_id": held["installation_id"],
            }
            for held in installed.values()
        }
        preparation: dict[str, Any] = {
            "installations": installations,
            "state": base64.b64encode(desired.SerializeToString(deterministic=True)).decode(),
        }
        assert worker.machine_calls is not None
        if sdk := worker.machine_calls.builtins.capture():
            preparation["runtime_builtin"] = sdk
        worker.machine_sources.hold(owner, offer.request_id, request.source_credentials)
        receipt = worker.submit_execution(
            request.claim,
            request.submission_id,
            capture_digest,
            offer,
            expected_execution_workspace_id=request.expected_execution_workspace_id,
            max_attempts=request.max_attempts or 1,
            retry_delay_ms=request.retry_delay_ms,
            capture_document=capture,
            preparation=canonical_json.encode(preparation),
            publication_authorization_id=request.publication_authorization_id,
            owner_memo=request.owner_memo,
        )
        worker.machine_sources.declare(owner, offer.request_id)
        executions.record(
            owner,
            offer.request_id,
            "resolved",
            {
                "package": root.package,
                "release": root.release,
                "installation_id": prepared.target.installation_id,
                "models": [
                    {
                        name: row[name]
                        for name in (
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
                        if name in row
                    }
                    for row in prepared.models
                ],
            },
        )
        return self._receipt(receipt)

    @staticmethod
    def _receipt(receipt: Receipt) -> pb.MachineExecutionReceipt:
        return pb.MachineExecutionReceipt(**asdict(receipt))
