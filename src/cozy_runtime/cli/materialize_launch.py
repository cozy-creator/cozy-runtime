"""Fixed generic-substrate materialize/launch adapter entrypoint."""

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
    prepare_launch,
)

WORKER_ROOT = Path("/run/cozy/worker")
MEDIA_ROOT = Path("/run/cozy/media")
UV_PATH = Path("/opt/cozy/bin/uv")


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize one frozen provision closure, prove listeners, then supervise worker."""

    opts = Options()
    process: subprocess.Popen[bytes] | None = None
    try:
        if argv is None:
            argv = sys.argv[1:]
        if argv:
            raise LaunchRefusal("adapter_arguments_forbidden", "fixed adapter takes no arguments")
        config = read_adapter_config()
        boot_id = readiness.pod_boot_id(config.pod_boot_id_path)
        certificate_der, certificate_ip = readiness.certificate(config.tls_cert_path)
        gpus = readiness.runtime_gpus(
            config.child_base_env, deadline=config.readiness_deadline_unix
        )
        prepared = prepare_launch(config, uv=UV_PATH)
        capacity = prepared.spec["capacity"]
        if len(gpus) != capacity["accelerator_count"] or any(
            row["device_name"] != capacity["accelerator_model"] for row in gpus
        ):
            raise LaunchRefusal(
                "runtime_gpu_mismatch", "measured GPU inventory differs from provision capacity"
            )
        process = _start_worker(config, prepared, boot_id, gpus)
        readiness.wait_worker_listener(
            WORKER_ROOT, process, deadline=config.readiness_deadline_unix
        )
        certificate_pem = config.tls_cert_path.read_bytes()
        readiness.prove_worker_foreign_refusal(
            config.worker_internal_port,
            certificate_pem,
            certificate_ip,
            deadline=config.readiness_deadline_unix,
        )
        readiness.prove_media_foreign_refusal(
            config.media_internal_port,
            certificate_der,
            certificate_ip,
            deadline=config.readiness_deadline_unix,
        )
        payload = readiness.receipt_payload(
            spec=prepared.spec,
            provision_spec_digest=config.provision_spec_digest,
            provision_bundle_digest=config.provision_bundle_digest,
            provider_pod_id=config.provider_pod_id,
            pod_boot_id_value=boot_id,
            certificate_der=certificate_der,
            installed_receipt_digest=prepared.installed.receipt_digest,
            placement_set_digest=prepared.placement_set_digest,
            gpus=gpus,
            artifact_sources=prepared.artifact_sources,
            network_cache_degradations=prepared.network_cache_degradations,
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
                "repair the immutable provision closure or substrate and launch a new attempt",
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
    prepared: object,
    boot_id: str,
    gpus: list[readiness.RuntimeGPU],
) -> subprocess.Popen[bytes]:
    # Local imports keep the public entrypoint small while preserving precise static types.
    from cozy_runtime.internal.config import AdapterConfig
    from cozy_runtime.internal.materialize_launch import PreparedLaunch

    assert isinstance(config, AdapterConfig)
    assert isinstance(prepared, PreparedLaunch)
    WORKER_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    (WORKER_ROOT / "control.addr").unlink(missing_ok=True)
    control_ref = prepared.spec["execution_substrate"]["control_runtime"]
    devices = ",".join(str(row["device_index"]) for row in gpus)
    child_env = dict(config.child_base_env)
    child_env.update(
        {
            "COZY_ACCESS_TOKEN_SHA256": ",".join(config.renter_token_sha256),
            "COZY_HOME": "/run/cozy",
            "PYTHONNOUSERSITE": "1",
        }
    )
    grant_roots = str(MEDIA_ROOT)
    command = [
        sys.executable,
        "-I",
        "-m",
        "cozy_runtime.cli.main",
        "serve",
        "--socket",
        f"0.0.0.0:{config.worker_internal_port}",
        "--out",
        str(WORKER_ROOT),
        "--worker-id",
        config.provider_pod_id,
        "--instance-id",
        boot_id,
        "--release-id",
        prepared.endpoint_release,
        "--artifact-digest",
        str(control_ref["digest"]),
        "--devices",
        devices,
        "--grant-root",
        grant_roots,
        "--environment-root",
        str(ENVIRONMENT_ROOT),
        "--artifact-store",
        str(prepared.artifact_store),
        "--artifact-object-set",
        str(prepared.artifact_object_set),
        "--tls-cert",
        str(config.tls_cert_path),
        "--tls-key",
        str(config.tls_key_path),
    ]
    return subprocess.Popen(
        command,
        cwd="/",
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
    )


def _supervise(process: subprocess.Popen[bytes]) -> int:
    """Keep the adapter coupled to its worker; XS owns media and whole-pod supervision."""

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
