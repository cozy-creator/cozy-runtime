"""Fixed environment-neutral Runtime worker for Cozy and base worker image hosts."""

from __future__ import annotations

import contextlib
import os
import select
import signal
import stat
import sys
import threading
from collections.abc import Callable, Sequence

from cozy_runtime.cli.io import CliError, Options, emit_error
from cozy_runtime.internal import host_paths, proctree, readiness
from cozy_runtime.internal.config import (
    ConfigError,
    WorkerHostConfig,
    read_worker_host_config,
    runtime_config_from_host,
)
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.placement_materialization import (
    PreparedBoot,
    baked_base_manifest,
    control_runtime_digest,
    prepare_boot,
)
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.downloads import DownloadClient, tls_certificate_digest
from cozy_runtime.internal.worker.session import Worker, WorkerOptions

POD_EXECUTOR_UID = 65533
POD_EXECUTOR_GID = 65533
POD_SUPERVISOR_UID = 65532
POD_SUPERVISOR_GID = 65532
SUPERVISOR_STOP_FD = 3


def main(argv: Sequence[str] | None = None) -> int:
    """Boot empty, host WorkerControl, prove both listeners, and publish readiness."""

    opts = Options()
    try:
        if argv is None:
            argv = sys.argv[1:]
        if argv:
            raise LaunchRefusal("worker_arguments_forbidden", "fixed worker takes no arguments")
        if os.geteuid() != 0 or os.getegid() != 0:
            raise LaunchRefusal("worker_identity_invalid", "pod worker must start as UID/GID 0")
        config = read_worker_host_config()
        _arm_supervisor_lifetime()
        return _run(config)
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


def _run(config: WorkerHostConfig) -> int:
    boot_id = readiness.pod_boot_id(host_paths.BOOT_ID)
    certificate_der = readiness.certificate(host_paths.TLS_CERTIFICATE)
    base = baked_base_manifest()
    accelerator_backend = dict(base.platform_target)["accelerator_backend"]
    gpus = readiness.runtime_gpus(
        config.child_base_env,
        deadline=config.lease_expiry_unix,
        accelerator_backend=accelerator_backend,
    )
    prepared = prepare_boot(config)
    options = _worker_options(
        config,
        prepared,
        boot_id,
        gpus,
        certificate_der=certificate_der,
        accelerator_backend=accelerator_backend,
        control_runtime_digest_value=control_runtime_digest(manifest=base),
    )
    runtime_config = runtime_config_from_host(config)
    host_paths.WORKER_STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    host_paths.WORKER_CONTROL_ADDRESS.unlink(missing_ok=True)
    host = GrpcControlHost(
        f"0.0.0.0:{config.worker_internal_port}",
        host_paths.WORKER_CONTROL_ADDRESS,
        tls_cert=host_paths.TLS_CERTIFICATE,
        tls_key=host_paths.TLS_PRIVATE_KEY,
    )
    worker = Worker(runtime_config, options, host)

    def ready(_address: str) -> None:
        if worker.stop.is_set():
            raise LaunchRefusal("supervisor_stop_before_readiness", "fd 3 closed before ready")
        certificate_pem = host_paths.TLS_CERTIFICATE.read_bytes()
        readiness.prove_tls_leaf(
            config.worker_internal_port,
            certificate_der,
            deadline=config.lease_expiry_unix,
            owner="worker",
        )
        readiness.prove_worker_foreign_refusal(
            config.worker_internal_port,
            certificate_pem,
            deadline=config.lease_expiry_unix,
        )
        readiness.prove_media_foreign_refusal(
            config.media_internal_port,
            certificate_der,
            deadline=config.lease_expiry_unix,
        )
        payload = readiness.receipt_payload(
            control_public_key_ed25519_b64url=config.control_public_key_ed25519_b64url,
            media_token_sha256=config.media_token_sha256,
            pod_boot_id_value=boot_id,
            certificate_der=certificate_der,
            worker_internal_port=config.worker_internal_port,
            media_internal_port=config.media_internal_port,
            gpus=gpus,
        )
        readiness.publish(
            host_paths.READINESS_PAYLOAD,
            payload,
            owner=(POD_SUPERVISOR_UID, POD_SUPERVISOR_GID),
        )
        worker.mark_claim_ready()

    host.on_bound = ready

    def stop_worker() -> None:
        worker.stop.set()
        worker.outbound.put(None)
        worker.work.put(None)
        host.stop()

    def stop_signal(_signum: int, _frame: object) -> None:
        stop_worker()

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    threading.Thread(
        target=_watch_supervisor,
        args=(worker, stop_worker),
        daemon=True,
        name="supervisor-fd",
    ).start()
    return worker.run()


def _worker_options(
    config: WorkerHostConfig,
    prepared: PreparedBoot,
    boot_id: str,
    gpus: list[readiness.RuntimeGPU],
    *,
    certificate_der: bytes,
    accelerator_backend: str,
    control_runtime_digest_value: str,
) -> WorkerOptions:
    """Construct the worker core directly; no public ``serve`` parser participates."""

    if accelerator_backend == "cuda" and not gpus:
        raise LaunchRefusal("runtime_gpus_invalid", "CUDA base measured no GPUs")
    if accelerator_backend == "none" and gpus:
        raise LaunchRefusal("runtime_gpus_invalid", "CPU base measured a GPU inventory")
    if accelerator_backend not in {"cuda", "none"}:
        raise LaunchRefusal("runtime_accelerator_backend_invalid", accelerator_backend)
    return WorkerOptions(
        root=host_paths.WORKER_STATE,
        worker_id=config.worker_id,
        instance_id=boot_id,
        worker_boot_id=boot_id,
        control_runtime_digest=control_runtime_digest_value,
        accelerator_backend=accelerator_backend,
        devices=",".join(str(row["device_index"]) for row in gpus),
        grant_roots=(str(host_paths.MEDIA_ROOT),),
        environment_root=host_paths.ENVIRONMENT_ROOT,
        artifact_cache=prepared.artifact_cache,
        artifact_store=prepared.artifact_store,
        download_client=DownloadClient(
            host_paths.TLS_CERTIFICATE,
            host_paths.TLS_PRIVATE_KEY,
            config.tensorhub_origin,
        ),
        worker_tls_certificate_digest=tls_certificate_digest(certificate_der),
        no_warm=True,
        claim_ready=False,
        executor_uid=POD_EXECUTOR_UID,
        executor_gid=POD_EXECUTOR_GID,
    )


def _arm_supervisor_lifetime() -> None:
    """Bind this root worker to PID 1 and its EOF-only fd 3."""

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


def _watch_supervisor(worker: Worker, stop: Callable[[], None]) -> None:
    """EOF stops cooperatively; data is a structural refusal, never a command."""

    os.set_blocking(SUPERVISOR_STOP_FD, False)
    poller = select.poll()
    poller.register(SUPERVISOR_STOP_FD, select.POLLIN | select.POLLHUP | select.POLLERR)
    while not worker.stop.is_set():
        if not poller.poll(250):
            continue
        try:
            data = os.read(SUPERVISOR_STOP_FD, 1)
        except BlockingIOError:
            continue
        if data:
            worker.exit_code = int(Exit.structural)
            print("[worker] fd 3 carried data; only EOF is valid", file=sys.stderr, flush=True)
        stop()
        return


if __name__ == "__main__":
    raise SystemExit(main())
