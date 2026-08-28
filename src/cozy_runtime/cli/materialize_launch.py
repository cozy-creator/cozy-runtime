"""Fixed generic-substrate boot adapter entrypoint.

The pod boots to READY-BUT-EMPTY. No provision document is read and nothing is fetched —
GPU inventory measured, worker started with zero placements, both listeners proven to
refuse a foreign credential, receipt published. The first placement arrives afterwards
over rev-2's `DesiredPlacementSet` from the connected RecordOwner, and is treated
identically to placements 2..N — it is FETCHED through the standing `ArtifactGrant` and
MATERIALIZED by the same acquisition walk every later revision runs (cr-049).

Nothing here re-measures the pod's own image. The hub pinned the substrate by OCI digest
before this process existed, so every byte baked into it is already attested; the pod
reports only what it OBSERVED — its GPUs, its listeners, the token hashes it was handed.
"""

from __future__ import annotations

import contextlib
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from cozy_runtime.cli.io import CliError, Options, emit_error
from cozy_runtime.internal import readiness
from cozy_runtime.internal.config import ConfigError, read_adapter_config
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.materialize_launch import (
    ENVIRONMENT_ROOT,
    LaunchRefusal,
    PreparedBoot,
    prepare_boot,
)

WORKER_ROOT = Path("/run/cozy/worker")
MEDIA_ROOT = Path("/run/cozy/media")


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
        boot_id = readiness.pod_boot_id(config.pod_boot_id_path)
        certificate_der = readiness.certificate(config.tls_cert_path)
        gpus = readiness.runtime_gpus(
            config.child_base_env, deadline=config.readiness_deadline_unix
        )
        prepared = prepare_boot(config)
        # The measured inventory is REPORTED, never compared here: the pod has no
        # independent authority on what the provider was told to deliver. Tensorhub's
        # binder joins `runtime_gpus` against the frozen attempt before it trusts the peer.
        process = _start_worker(config, prepared, boot_id, gpus)
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
        readiness.publish(config.receipt_payload_path, payload)
        return _supervise(process)
    except ConfigError as err:
        emit_error(CliError(err.name, err.message, err.remedy, err.code), opts)
        return int(err.code)
    except LaunchRefusal as err:
        emit_error(
            CliError(
                err.code,
                err.detail,
                "repair the substrate image or the create-time environment and launch a "
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
) -> subprocess.Popen[bytes]:
    # Local import keeps the public entrypoint small while preserving precise static types.
    from cozy_runtime.internal.config import AdapterConfig

    assert isinstance(config, AdapterConfig)
    WORKER_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    (WORKER_ROOT / "control.addr").unlink(missing_ok=True)
    child_env = dict(config.child_base_env)
    child_env.update(
        {
            # `sha256:`-PREFIXED, and the prefix is the whole grammar (#560g): a bare hex
            # entry is parsed as a PATH to a hash file, so the joined bare hashes this
            # passed before made every renter Claim refuse on a rented pod. The adapter
            # holds a literal hash SET, so it must spell one.
            "COZY_ACCESS_TOKEN_SHA256": ",".join(
                "sha256:" + item for item in config.renter_token_sha256
            ),
            "COZY_HOME": "/run/cozy",
            "PYTHONNOUSERSITE": "1",
        }
    )
    command = _worker_command(config, prepared, boot_id, gpus)
    return subprocess.Popen(
        command,
        cwd="/",
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
    )


def _worker_command(
    config: object,
    prepared: PreparedBoot,
    boot_id: str,
    gpus: list[readiness.RuntimeGPU],
) -> list[str]:
    """The fixed dynamic-pod worker argv, kept pure so the launch contract is executable."""

    from cozy_runtime.internal.config import AdapterConfig

    assert isinstance(config, AdapterConfig)
    devices = ",".join(str(row["device_index"]) for row in gpus)
    grant_roots = str(MEDIA_ROOT)
    return [
        sys.executable,
        "-I",
        "-m",
        "cozy_runtime.cli.main",
        "serve",
        # A generic substrate becomes useful by staging the requested endpoint and model,
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
        # No `--release-id`: a pod that hosts no placement HAS no endpoint release, and the
        # worker's empty `release_id` is the honest answer an owner reads back. No
        # `--artifact-digest` for the same reason: the only substrate identity this pod
        # could measure is its own image, which the hub pinned by digest, so "absent stays
        # absent" (#481) is the honest answer there too.
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


def _supervise(process: subprocess.Popen[bytes]) -> int:
    """Keep the adapter coupled to its worker; cozy-pod owns media and whole-pod supervision."""

    stopped = False
    terminate_at = 0.0

    def stop(signum: int, frame: object) -> None:
        nonlocal stopped, terminate_at
        stopped = True
        if process.poll() is None:
            process.terminate()
            terminate_at = terminate_at or time.monotonic()

    prior_term = signal.signal(signal.SIGTERM, stop)
    prior_int = signal.signal(signal.SIGINT, stop)
    try:
        while True:
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
    if stopped:
        return int(Exit.ok)
    if code == 0:
        return int(Exit.internal)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
