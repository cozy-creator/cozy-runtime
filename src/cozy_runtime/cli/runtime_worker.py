"""The one Runtime worker every machine runs: a pod rooted at `/`, a laptop at its Host's root.

Privilege is detected, never configured: started as root it isolates package executors under
their own uid and hands its handoff files to the Host's dropped uid; started as a user it runs
executors as that user and hands nothing over.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import signal
import stat
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from cozy_runtime.cli.io import CliError, Options, emit_error
from cozy_runtime.internal import (
    host_paths,
    package_environment,
    proctree,
    readiness,
    storage_admission,
)
from cozy_runtime.internal.config import (
    ConfigError,
    WorkerHostConfig,
    read_worker_host_config,
    runtime_config_from_host,
)
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.placement_materialization import prepare_boot
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker import machine_publication
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.downloads import tls_certificate_digest
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.session import Worker, WorkerOptions

POD_EXECUTOR_UID = 65533
POD_EXECUTOR_GID = 65533
POD_SUPERVISOR_UID = 65532
POD_SUPERVISOR_GID = 65532
SUPERVISOR_STOP_FD = 3


@dataclass(frozen=True, slots=True)
class Identity:
    """What this process's uid lets it do: the one privilege fact the worker branches on."""

    executor_uid: int = -1
    executor_gid: int = -1
    host_owner: tuple[int, int] | None = None

    @classmethod
    def detect(cls) -> Identity:
        euid, egid = os.geteuid(), os.getegid()
        if euid == 0 and egid == 0:
            return cls(POD_EXECUTOR_UID, POD_EXECUTOR_GID, (POD_SUPERVISOR_UID, POD_SUPERVISOR_GID))
        if euid == 0 or egid == 0:
            raise LaunchRefusal(
                "worker_identity_invalid", f"uid {euid} gid {egid} is neither root nor a user"
            )
        return cls()


def main(
    argv: Sequence[str] | None = None,
    *,
    gpus: Sequence[readiness.RuntimeGPU] | None = None,
) -> int:
    """Boot empty, host WorkerControl, prove both listeners, and publish readiness.

    `gpus` replaces the driver measurement for an in-process launcher (a virtual inventory);
    the console entry point never passes it.
    """

    opts = Options()
    try:
        if argv is None:
            argv = sys.argv[1:]
        if list(argv) == ["capabilities", "--json"]:
            print(
                json.dumps(
                    {
                        "capabilities": [
                            "machine-supervisor/1",
                            "runtime-model-preparation/1",
                            "runtime-model-overrides/1",
                            "machine-public-reads/1",
                        ]
                    }
                )
            )
            return 0
        if argv:
            raise LaunchRefusal("worker_arguments_forbidden", "fixed worker takes no arguments")
        identity = Identity.detect()
        config = read_worker_host_config()
        _arm_supervisor_lifetime()
        storage_admission.inherit(config.storage_admission_fd)
        return _run(config, identity, gpus)
    except ConfigError as err:
        emit_error(CliError(err.name, err.message, err.remedy, err.code), opts)
        return int(err.code)
    except LaunchRefusal as err:
        emit_error(
            CliError(
                err.code,
                err.detail,
                "repair the base worker image or host handoff and launch a new worker",
                Exit.structural,
            ),
            opts,
        )
        return int(Exit.structural)
    except Exception as err:
        emit_error(
            CliError(
                "internal",
                f"unexpected worker fault: {type(err).__name__}",
                "report this Runtime build and worker generation",
                Exit.internal,
            ),
            opts,
        )
        return int(Exit.internal)
    finally:
        with contextlib.suppress(OSError):
            os.close(SUPERVISOR_STOP_FD)


def _run(
    config: WorkerHostConfig,
    identity: Identity,
    virtual_gpus: Sequence[readiness.RuntimeGPU] | None = None,
) -> int:
    layout = config.layout
    boot_id = readiness.pod_boot_id(layout.boot_id)
    certificate_der = readiness.certificate(layout.tls_certificate)
    try:
        base = package_environment.observe_base()
    except package_environment.EnvironmentRefusal as exc:
        raise LaunchRefusal(exc.code, exc.detail) from exc
    if base.accelerator_backend not in {"cuda", "none"}:
        raise LaunchRefusal("runtime_accelerator_backend_invalid", base.accelerator_backend)
    required = base.accelerator_backend == "cuda"
    gpus = (
        readiness.runtime_gpus(config.child_base_env, config.visible_devices, required=required)
        if virtual_gpus is None
        else readiness.validated_gpus(list(virtual_gpus), required=required)
    )
    tensorfs_root = prepare_boot(config)
    options = _worker_options(config, identity, tensorfs_root, boot_id, gpus, certificate_der)
    runtime_config = runtime_config_from_host(config)
    layout.worker_state.mkdir(parents=True, exist_ok=True, mode=0o700)
    layout.worker_control_address.unlink(missing_ok=True)
    host = GrpcControlHost(
        f"127.0.0.1:{config.worker_internal_port}",
        layout.worker_control_address,
        tls_cert=layout.tls_certificate,
        tls_key=layout.tls_private_key,
    )
    worker = Worker(runtime_config, options, host)
    host.preparer = worker.prepare_package_set
    host.model_source_preparer = worker.prepare_model_source
    host.model_source_releaser = worker.release_model_source
    host.derived_retainer = worker.retain_derived_result
    host.derived_result_releaser = worker.release_derived_result
    host.numerical_environment = worker.numerical_environment
    host.workspace_service = worker.workspace_service
    host.store_collector = worker.collect_store_garbage
    host.checkpoint_pager = worker.checkpoint_page
    host.checkpoint_transfer = worker.checkpoint_transfer
    host.weights_checkpoint_validator = worker.weights.validate_checkpoint
    host.local_package_preparer = worker.prepare_local_package
    host.unpublished_placement_preparer = worker.prepare_unpublished_placement
    # The loopback weights seam (proto-025). The supervisor opens Exchange per control
    # session and calls Upload; both are the worker's own WeightsExchange.
    host.weights = worker.weights

    def ready(_address: str) -> None:
        if worker.stop.is_set():
            raise LaunchRefusal("supervisor_stop_before_readiness", "fd 3 closed before ready")
        certificate_pem = layout.tls_certificate.read_bytes()
        readiness.prove_tls_leaf(config.worker_internal_port, certificate_der, owner="worker")
        readiness.prove_worker_foreign_refusal(config.worker_internal_port, certificate_pem)
        readiness.prove_media_foreign_refusal(config.media_internal_port, certificate_der)
        payload = readiness.receipt_payload(
            control_public_key_ed25519_b64url=config.control_public_key_ed25519_b64url,
            media_token_sha256=config.media_token_sha256,
            pod_boot_id_value=boot_id,
            certificate_der=certificate_der,
            worker_internal_port=config.worker_internal_port,
            media_internal_port=config.media_internal_port,
            gpus=gpus,
            process_incarnation=config.process_incarnation,
        )
        # Claims open before the payload that tells a claimant to claim.
        worker.mark_claim_ready()
        readiness.publish(layout.readiness_payload, payload, owner=identity.host_owner)

    host.on_bound = ready

    def stop_worker() -> None:
        worker.request_stop()
        host.stop()

    def stop_signal(_signum: int, _frame: object) -> None:
        stop_worker()

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGHUP, _restart_handler(worker))
    threading.Thread(
        target=_watch_supervisor,
        args=(worker, stop_worker),
        daemon=True,
        name="supervisor-fd",
    ).start()
    return worker.run()


def _worker_options(
    config: WorkerHostConfig,
    identity: Identity,
    tensorfs_root: Path,
    boot_id: str,
    gpus: list[readiness.RuntimeGPU],
    certificate_der: bytes,
) -> WorkerOptions:
    """Construct the worker core directly; no public ``serve`` parser participates."""

    layout = config.layout
    hubs, skipped = machine_publication.registrations(layout.machine_hubs)
    for line in skipped:
        print(f"[worker] {line}", file=sys.stderr, flush=True)
    return WorkerOptions(
        root=layout.worker_state,
        worker_id=config.worker_id,
        instance_id=boot_id,
        worker_boot_id=boot_id,
        sole_supervisor=config.control_mode == "supervisor",
        process_incarnation=config.process_incarnation,
        ownership_required=layout.readiness_envelope.exists(),
        owns_tensorfs_store=True,
        accelerator_backend="cuda" if gpus else "none",
        devices=",".join(str(row["device_index"]) for row in gpus),
        gpus=tuple(gpus),
        grant_roots=(str(layout.media_root), str(host_paths.input_media_root())),
        install_root=layout.install_root,
        artifact_cache=layout.prepare_cache,
        tensorfs_root=tensorfs_root,
        publication_authority=PublicationAuthority(
            config.tensorhub_origin,
            config.worker_id,
            config.tensorhub_worker_token,
            layout.tensorhub_ca if layout.tensorhub_ca.is_file() else None,
            config.object_storage_hosts,
        )
        if config.tensorhub_origin
        else None,
        hubs=hubs,
        hub_access_path=layout.machine_hubs,
        worker_tls_certificate_digest=tls_certificate_digest(certificate_der),
        claim_ready=False,
        executor_uid=identity.executor_uid,
        executor_gid=identity.executor_gid,
        # The controller reads a bundle through ReadMachineExecutionTriage. It stays under
        # the media subtree, readable by the Host's uid, while `/v1/triage` still serves it.
        triage_root=layout.triage_root,
        triage_owner=identity.host_owner,
        # The Host's one window into whether this machine is being used (th-142): nobody
        # attached, nothing in flight, and nothing moving is idle.
        activity_path=layout.worker_activity,
        activity_owner=identity.host_owner,
        memo=config.memo,
    )


def _arm_supervisor_lifetime() -> None:
    """Bind this worker to the Host that exec'd it and its EOF-only fd 3."""

    try:
        info = os.fstat(SUPERVISOR_STOP_FD)
    except OSError as exc:
        raise LaunchRefusal("supervisor_stop_pipe_absent", "inherited fd 3 is absent") from exc
    if not stat.S_ISFIFO(info.st_mode):
        raise LaunchRefusal("supervisor_stop_pipe_invalid", "inherited fd 3 is not a pipe")
    parent = os.getppid()
    try:
        proctree.arm_parent_death()
    except (OSError, proctree.ProcessTreeUnsupported) as exc:
        raise LaunchRefusal("supervisor_parent_death_unavailable", str(exc)) from exc
    if os.getppid() != parent:
        raise LaunchRefusal(
            "supervisor_parent_changed", "pod-supervisor exited while PDEATHSIG was armed"
        )


def _restart_handler(worker: Worker) -> Callable[[int, object], None]:
    def restart_signal(_signum: int, _frame: object) -> None:
        # The supervisor takes the lock on another thread; signals cannot reenter its RLock.
        worker.schedule_idle_restart()

    return restart_signal


def _watch_supervisor(
    worker: Worker, stop: Callable[[], None], descriptor: int = SUPERVISOR_STOP_FD
) -> None:
    """EOF stops cooperatively; data is a structural refusal, never a command."""

    os.set_blocking(descriptor, False)
    poller = select.poll()
    poller.register(descriptor, select.POLLIN | select.POLLHUP | select.POLLERR)
    while not worker.stop.is_set():
        if worker.idle_restart_pending.is_set():
            worker.idle_restart_pending.clear()
            try:
                worker.request_idle_restart()
            except Exception as exc:
                # Unreadable activity cannot turn an idle-only update into a process crash.
                worker.note(
                    "restart_refused", f"idle restart check failed: {type(exc).__name__}: {exc}"
                )
        if not poller.poll(250):
            continue
        try:
            data = os.read(descriptor, 1)
        except BlockingIOError:
            continue
        if data:
            worker.exit_code = int(Exit.structural)
            print("[worker] fd 3 carried data; only EOF is valid", file=sys.stderr, flush=True)
        stop()
        return
    stop()  # a local restart request also ends the host's blocking serve loop


if __name__ == "__main__":
    raise SystemExit(main())
