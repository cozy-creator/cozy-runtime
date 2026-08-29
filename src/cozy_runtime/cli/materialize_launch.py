"""Fixed base-worker-image boot adapter entrypoint.

The pod boots to READY-BUT-EMPTY. No provision document is read and nothing is fetched —
GPU inventory measured, worker started with zero placements, both listeners proven to
refuse a foreign credential, receipt published. The first placement arrives afterwards
over rev-2's `DesiredPlacementSet` from the connected RecordOwner, and is treated
identically to placements 2..N — it is FETCHED through the standing `ArtifactGrant` and
MATERIALIZED by the same acquisition walk every later revision runs (cr-049).

Nothing here re-measures the pod's own image. The hub pinned the base worker image by OCI digest
before this process existed, so every byte baked into it is already attested; the pod
reports only what it OBSERVED — its GPUs, its listeners, the token hashes it was handed.
"""

from __future__ import annotations

import contextlib
import os
import select
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from cozy_runtime.cli.io import CliError, Options, emit_error
from cozy_runtime.internal import proctree, readiness
from cozy_runtime.internal.config import ConfigError, read_adapter_config
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.materialize_launch import (
    ENVIRONMENT_ROOT,
    LaunchRefusal,
    PreparedBoot,
    baked_base_manifest,
    control_runtime_digest,
    prepare_boot,
)

WORKER_ROOT = Path("/run/cozy/worker")
POD_EXECUTOR_UID = 65533
POD_EXECUTOR_GID = 65533
POD_SUPERVISOR_UID = 65532
POD_SUPERVISOR_GID = 65532
SUPERVISOR_STOP_FD = 3
# Shared with pod-supervisor's podmedia server. This is persistent pod data, unlike /run's
# restart-local control state: PutInput returns file:// paths beneath this exact root and
# the worker's grant authorizer must recognize the same tree without widening it.
MEDIA_ROOT = Path("/var/lib/cozy/media")


def main(argv: Sequence[str] | None = None) -> int:
    """Boot empty, prove both listeners, publish the receipt, then supervise the worker."""

    opts = Options()
    process: subprocess.Popen[bytes] | None = None
    try:
        if argv is None:
            argv = sys.argv[1:]
        if argv:
            raise LaunchRefusal("adapter_arguments_forbidden", "fixed adapter takes no arguments")
        config = read_adapter_config()
        _arm_supervisor_lifetime()
        boot_id = readiness.pod_boot_id(config.pod_boot_id_path)
        certificate_der = readiness.certificate(config.tls_cert_path)
        base = baked_base_manifest()
        platform = dict(base.platform_target)
        accelerator_backend = platform["accelerator_backend"]
        gpus = readiness.runtime_gpus(
            config.child_base_env,
            deadline=config.readiness_deadline_unix,
            accelerator_backend=accelerator_backend,
        )
        prepared = prepare_boot(config)
        # The measured inventory is REPORTED, never compared here: the pod has no
        # independent authority on what the provider was told to deliver. Tensorhub's
        # binder joins `runtime_gpus` against the frozen attempt before it trusts the peer.
        process = _start_worker(
            config,
            prepared,
            boot_id,
            gpus,
            accelerator_backend=accelerator_backend,
            control_runtime_digest_value=control_runtime_digest(manifest=base),
        )
        readiness.wait_worker_listener(
            WORKER_ROOT, process, deadline=config.readiness_deadline_unix
        )
        certificate_pem = config.tls_cert_path.read_bytes()
        readiness.prove_worker_foreign_refusal(
            config.worker_internal_port,
            certificate_pem,
            deadline=config.readiness_deadline_unix,
        )
        readiness.prove_media_foreign_refusal(
            config.media_internal_port,
            certificate_der,
            deadline=config.readiness_deadline_unix,
        )
        payload = readiness.receipt_payload(
            renter_token_sha256=config.renter_token_sha256,
            pod_boot_id_value=boot_id,
            certificate_der=certificate_der,
            worker_internal_port=config.worker_internal_port,
            media_internal_port=config.media_internal_port,
            gpus=gpus,
        )
        # pod-supervisor drops to its request-serving uid before this handoff arrives. The
        # adapter is the sole writer; the supervisor gets read-only ownership of the exact
        # inode only after its bytes are flushed and before its atomic link is published.
        readiness.publish(
            config.receipt_payload_path,
            payload,
            owner=(POD_SUPERVISOR_UID, POD_SUPERVISOR_GID),
        )
        return _supervise(process, supervisor_fd=SUPERVISOR_STOP_FD)
    except ConfigError as err:
        emit_error(CliError(err.name, err.message, err.remedy, err.code), opts)
        return int(err.code)
    except LaunchRefusal as err:
        emit_error(
            CliError(
                err.code,
                err.detail,
                "repair the base worker image or the create-time environment and launch a "
                "new attempt",
                Exit.structural,
            ),
            opts,
        )
        return int(Exit.structural)
    except Exception as err:
        emit_error(
            CliError(
                "internal",
                f"unexpected adapter fault: {type(err).__name__}",
                "report this Runtime build and launch attempt",
                Exit.internal,
            ),
            opts,
        )
        return int(Exit.internal)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=10)


def _start_worker(
    config: object,
    prepared: PreparedBoot,
    boot_id: str,
    gpus: list[readiness.RuntimeGPU],
    *,
    accelerator_backend: str,
    control_runtime_digest_value: str,
) -> subprocess.Popen[bytes]:
    # Local import keeps the public entrypoint small while preserving precise static types.
    from cozy_runtime.internal.config import AdapterConfig

    assert isinstance(config, AdapterConfig)
    WORKER_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    (WORKER_ROOT / "control.addr").unlink(missing_ok=True)
    child_env = dict(config.child_base_env)
    child_env.update(
        {
            # Creator remains the private pod's sole RecordOwner. It holds the renter token
            # preimage and both media and WorkerControl verify against the same bounded hash
            # set the pod received; no Hub-owned Claim credential exists in this process.
            "COZY_ACCESS_TOKEN_SHA256": ",".join(
                "sha256:" + item for item in config.renter_token_sha256
            ),
            "COZY_HOME": "/run/cozy",
            "PYTHONNOUSERSITE": "1",
        }
    )
    command = _worker_command(
        config,
        prepared,
        boot_id,
        gpus,
        accelerator_backend=accelerator_backend,
        control_runtime_digest_value=control_runtime_digest_value,
    )
    return subprocess.Popen(
        command,
        cwd="/",
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
    )


def _arm_supervisor_lifetime() -> None:
    """Bind the root adapter to PID1 and its one fixed EOF-only stop pipe."""

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


def _worker_command(
    config: object,
    prepared: PreparedBoot,
    boot_id: str,
    gpus: list[readiness.RuntimeGPU],
    *,
    accelerator_backend: str,
    control_runtime_digest_value: str,
) -> list[str]:
    """The fixed dynamic-pod worker argv, kept pure so the launch contract is executable."""

    from cozy_runtime.internal.config import AdapterConfig

    assert isinstance(config, AdapterConfig)
    if accelerator_backend == "cuda" and not gpus:
        raise LaunchRefusal("runtime_gpus_invalid", "CUDA base measured no GPUs")
    if accelerator_backend == "none" and gpus:
        raise LaunchRefusal("runtime_gpus_invalid", "CPU base measured a GPU inventory")
    if accelerator_backend not in {"cuda", "none"}:
        raise LaunchRefusal("runtime_accelerator_backend_invalid", accelerator_backend)
    devices = ",".join(str(row["device_index"]) for row in gpus)
    grant_roots = str(MEDIA_ROOT)
    return [
        sys.executable,
        "-I",
        "-m",
        "cozy_runtime.cli.main",
        "serve",
        # A base worker image becomes useful by staging the requested endpoint and model,
        # not by running an arbitrary synthetic inference before it can advertise the
        # placement. Endpoint code is dynamic and untrusted to be cheap: a handler blocked
        # in its first CUDA read previously left an otherwise prepared pod ACTIVATING
        # forever. Fixed remote pods therefore skip the optional boot warm pass. This does
        # not change hand-started/local `serve`, whose explicit flag still defaults off.
        "--no-warm",
        "--socket",
        f"0.0.0.0:{config.worker_internal_port}",
        "--out",
        str(WORKER_ROOT),
        # The attempt id IS the worker's name on the rev-2 wire: a Claim naming another
        # worker is refused, and every Fault is attributed to it. This is the whole reason
        # COZY_ACQUISITION_ATTEMPT_ID survived the environment diet the ordinal and the
        # rental id did not — it is spent here, not echoed back to the hub.
        "--worker-id",
        config.acquisition_attempt_id,
        "--instance-id",
        boot_id,
        # No `--release-id`: a pod that hosts no placement HAS no endpoint release. The
        # artifact field carries the image-owned CONTROL RUNTIME WHEEL identity measured
        # from the baked complete WheelhouseManifest, never an OCI self-reference.
        "--control-runtime-digest",
        control_runtime_digest_value,
        "--accelerator-backend",
        accelerator_backend,
        # The trusted worker stays root. Every endpoint executor crosses the trampoline
        # into this fixed unprivileged identity after no_new_privs; inability to make that
        # transition is a pre-bind structural refusal.
        "--executor-uid",
        str(POD_EXECUTOR_UID),
        "--executor-gid",
        str(POD_EXECUTOR_GID),
        "--devices",
        devices,
        "--grant-root",
        grant_roots,
        "--environment-root",
        str(ENVIRONMENT_ROOT),
        "--artifact-cache",
        str(prepared.artifact_cache),
        "--artifact-store",
        str(prepared.artifact_store),
        "--artifact-object-set",
        str(prepared.artifact_object_set),
        "--tls-cert",
        str(config.tls_cert_path),
        "--tls-key",
        str(config.tls_key_path),
    ]


def _supervise(process: subprocess.Popen[bytes], *, supervisor_fd: int) -> int:
    """Keep the adapter coupled to its worker.

    pod-supervisor owns media and whole-pod supervision.
    """

    stopped = False
    invalid_stop_pipe = False
    terminate_at = 0.0

    os.set_blocking(supervisor_fd, False)
    stop_poll = select.poll()
    stop_poll.register(supervisor_fd, select.POLLIN | select.POLLHUP | select.POLLERR)

    def request_stop() -> None:
        nonlocal stopped, terminate_at
        stopped = True
        if process.poll() is None and not terminate_at:
            process.terminate()
            terminate_at = time.monotonic()

    def stop(signum: int, frame: object) -> None:
        del signum, frame
        request_stop()

    prior_term = signal.signal(signal.SIGTERM, stop)
    prior_int = signal.signal(signal.SIGINT, stop)
    try:
        while True:
            if stop_poll.poll(0):
                chunk = os.read(supervisor_fd, 1)
                if chunk:
                    invalid_stop_pipe = True
                    print(
                        "materialize-launch: supervisor stop pipe carried data; only EOF "
                        "is a command",
                        file=sys.stderr,
                        flush=True,
                    )
                request_stop()
            try:
                code = process.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if terminate_at and time.monotonic() - terminate_at >= 10:
                    process.kill()
                    try:
                        code = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        return int(Exit.internal)
                    break
    finally:
        signal.signal(signal.SIGTERM, prior_term)
        signal.signal(signal.SIGINT, prior_int)
        os.close(supervisor_fd)
    if invalid_stop_pipe:
        return int(Exit.structural)
    if stopped:
        return int(Exit.ok)
    if code == 0:
        return int(Exit.internal)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
