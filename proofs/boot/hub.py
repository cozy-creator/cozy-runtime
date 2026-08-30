"""cr-049's hub stand-in: author one package release, then serve its exact bytes.

This module plays the two roles the pod is not: the BUILD side that authors a release's
canonical documents (tensorhub's `releaseassembly` + `wheelset` + `placementset`, whose Go
authors produce these same shapes), and the ARTIFACT ORIGIN those documents are fetched
from. It is deliberately a repository proof driver and not a shipped component — what it
proves is the WORKER's half: that a pod which boots empty converges to a live placement from
a signed `EnsureArtifacts` plus one `DesiredPlacementSet` and nothing else.

The receipt is derived from exact base-family and package-wheel identities. This module
materializes the overlay once to learn it; the worker independently materializes the same
overlay and must produce byte-identical receipt bytes without copying the base environment.
"""

from __future__ import annotations

import base64
import http.server
import json
import ssl
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from proof_data import Wheel, canonical, digest, installed_base_ownership, inventory, ref, wheel

from cozy_runtime.internal import canonical as canon
from cozy_runtime.internal import descriptor as desc
from cozy_runtime.internal import local, overlay
from cozy_runtime.internal.discovery import discover_installed
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

PACKAGE_RELEASE = "proof-boot-release-1"
PLACEMENT_ID = "cr049-first-placement"
ARTIFACT_PORT = 43102


@dataclass
class Release:
    """Every exact byte string this release is made of, keyed by its own digest."""

    objects: dict[str, bytes] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    wheel_paths: dict[str, Path] = field(default_factory=dict)
    environment_spec: bytes = b""
    package_bundle: bytes = b""
    resolved_wheel_set: bytes = b""
    wheelhouse_manifest: bytes = b""
    descriptor: bytes = b""
    receipt: bytes = b""
    placement_set: bytes = b""
    plan_refs: list[tuple[str, int]] = field(default_factory=list)

    def add(self, raw: bytes, kind: str) -> bytes:
        self.objects[digest(raw)] = raw
        self.kinds[digest(raw)] = kind
        return raw

    def add_wheel(self, path: Path) -> Wheel:
        entry = inventory(path)
        self.objects[entry.digest] = path.read_bytes()
        self.kinds[entry.digest] = "wheel"
        self.wheel_paths[entry.digest] = path
        return entry


def platform_target(python_abi: str, libc: str) -> dict[str, str]:
    return {
        "accelerator_abi": "none",
        "accelerator_backend": "none",
        "libc": libc,
        "os_arch": "linux/amd64",
        "python_abi": python_abi,
    }


def author(
    *,
    application: str,
    project_wheel: Path,
    wheelhouse_wheels: list[Path],
    target: dict[str, str],
    build_root: Path,
    python: Path,
    base_manifest: Path,
) -> Release:
    """Author the whole closure a Placement reaches, and measure its receipt."""

    release = Release()
    project = release.add_wheel(project_wheel)
    house = sorted((inventory(path) for path in wheelhouse_wheels), key=lambda e: e.distribution)
    if len(house) != 8:
        raise AssertionError(f"boot proof base carries {len(house)} control wheels, want 8")

    base_distributions, base_import_roots = installed_base_ownership()
    profile = {
        "accelerator_build": "cpu",
        "os_cpu": "linux-x86",
        "python_abi": target["python_abi"],
        "torch_release": "2.13.0",
    }
    wheelhouse = release.add(
        canonical(
            {
                "base_distributions": base_distributions,
                "base_import_roots": base_import_roots,
                "compatibility_profile": profile,
                "format": "WheelhouseManifest/3",
                "minimum_cuda_version": "none",
                "minimum_driver_version": "none",
                "platform_target": dict(target),
                "wheels": [wheel(entry, import_roots=True) for entry in house],
            }
        ),
        "wheelhouse",
    )
    resolved = release.add(
        canonical(
            {
                "compatibility_profile": profile,
                "format": "ResolvedWheelSet/3",
                "overlay_bytes": 0,
                "platform_target": dict(target),
                "resolution_lock_digest": digest(b"cr-049-resolution-lock"),
                "resolution_lock_length": len(b"cr-049-resolution-lock"),
                "wheelhouse_manifest_digest": digest(wheelhouse),
                "wheels": [],
            }
        ),
        "wheelset",
    )

    # The descriptor is BUILT from the project wheel, never hand-written: the executor
    # re-derives it from the installed overlay and refuses a published one that differs.
    sys.path.append(str(project_wheel))
    try:
        document = desc.build(discover_installed(application))
    finally:
        sys.path.remove(str(project_wheel))
    descriptor_bytes = release.add(desc.canonical_bytes(document), "package_descriptor")
    # This exact ref proves publication provenance but is deliberately not a
    # runtime grant subject. The executable source payload is project_wheel.
    source_tree = b"cr-049-boot-proof-source-tree"
    bundle = release.add(
        canonical(
            {
                "package_descriptor": ref(descriptor_bytes),
                "package_release_id": PACKAGE_RELEASE,
                "format": "tensorhub.package_bundle/3",
                "project_wheel": wheel(project, import_roots=True),
                "resolved_wheel_set": ref(resolved),
                "source_tree": ref(source_tree),
                "wheelhouse_manifest": ref(wheelhouse),
            }
        ),
        "bundle",
    )
    spec = release.add(
        documents.canonical_bytes(
            pb.PackageEnvironmentSpec(
                compatibility_profile=pb.CompatibilityProfile(**profile),
                platform_target=pb.PlatformTarget(**target),
                package_bundle_digest=documents.digest_of(bundle),
                project_wheel_digest=documents.raw(project.digest),
                wheelhouse_manifest_digest=documents.digest_of(wheelhouse),
            )
        ),
        "environment",
    )
    release.environment_spec = spec
    release.package_bundle = bundle
    release.resolved_wheel_set = resolved
    release.wheelhouse_manifest = wheelhouse
    release.descriptor = descriptor_bytes

    # The image builder would bake this exact small family manifest beside the installed
    # official PyTorch environment. This proof writes the stand-in before either independent
    # overlay materialization and then treats it as read-only.
    base_manifest.parent.mkdir(parents=True, exist_ok=True)
    base_manifest.write_bytes(wheelhouse)
    base_manifest.chmod(0o444)

    from cozy_runtime.internal.placement_materialization import package_overlay_input

    source = package_overlay_input(
        environment_spec_bytes=spec,
        package_bundle_bytes=bundle,
        resolved_wheel_set_bytes=resolved,
        wheelhouse_manifest_bytes=wheelhouse,
        wheel_paths={project.digest: project_wheel},
    )
    installed = overlay.materialize(
        source,
        build_root,
        python=python,
        base_manifest=base_manifest,
    )
    release.receipt = release.add(installed.receipt_bytes, "receipt")

    # The binding-plan closure. A weightless package's plans, model-root document and
    # descriptor are exactly what tensorhub authors for a model-free release.
    weightless = local.weightless_release_documents(document)
    if weightless.descriptor != descriptor_bytes:
        raise RuntimeError("the weightless closure and the bundle name different descriptors")
    plans = [release.add(raw, "plan") for _, raw in sorted(weightless.plans.items())]
    release.plan_refs = sorted((digest(raw), len(raw)) for raw in plans)
    model_object_set = release.add(weightless.artifact_object_set, "model_object_set")

    placements = [
        {
            "placement_id": PLACEMENT_ID,
            "binding_plans": [
                {
                    "digest": plan_digest,
                    "kind": "plan",
                    "length": length,
                    "subject_id": plan_digest,
                }
                for plan_digest, length in release.plan_refs
            ],
            "package_descriptor_digest": digest(descriptor_bytes),
            "package_release_id": PACKAGE_RELEASE,
            "environment_spec_digest": digest(spec),
            "installed_environment_receipt_digest": digest(installed.receipt_bytes),
            "model_object_set": {
                "digest": digest(model_object_set),
                "kind": "model_object_set",
                "length": len(model_object_set),
                "subject_id": digest(model_object_set),
            },
        }
    ]
    release.placement_set = canon.write(
        {"format": documents.doc_format("cozy.worker.v1.PlacementSet"), "placements": placements}
    )
    return release


class _Origin(http.server.BaseHTTPRequestHandler):
    """Exact bytes at their own content address. No redirect, no transform, no listing."""

    objects: ClassVar[dict[str, bytes]] = {}
    kinds: ClassVar[dict[str, str]] = {}
    overlap_subjects: ClassVar[frozenset[str]] = frozenset()
    overlap_barrier: ClassVar[threading.Barrier | None] = None
    overlapped: ClassVar[set[str]] = set()
    delayed_model_subject: ClassVar[str] = ""
    creator_public_key: ClassVar[bytes] = b""
    worker_certificate_der: ClassVar[bytes] = b""
    resolved_mtls: ClassVar[bool] = False
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        spelled = "sha256:" + self.path.rsplit("/", 1)[-1]
        body = _Origin.objects.get(spelled) if self.path.startswith("/artifacts/") else None
        if body is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if spelled in _Origin.overlap_subjects:
            barrier = _Origin.overlap_barrier
            assert barrier is not None
            barrier.wait(timeout=10)
            _Origin.overlapped.add(spelled)
            if spelled == _Origin.delayed_model_subject:
                # Keep the model leg observably live after the package wheel is admitted;
                # the Runtime proof must spawn its executor during this window.
                time.sleep(2)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/v1/worker/artifacts/resolve":
            self._json(
                404,
                {
                    "error": {"code": "not_found", "message": "not found"},
                    "server_time_unix": int(time.time()),
                },
            )
            return
        if self.connection.getpeercert(binary_form=True) != _Origin.worker_certificate_der:
            self._json(
                403,
                {
                    "error": {"code": "worker_identity", "message": "wrong leaf"},
                    "server_time_unix": int(time.time()),
                },
            )
            return
        _Origin.resolved_mtls = True
        try:
            length = int(self.headers.get("Content-Length", "0"))
            assert 0 < length <= 1 << 20
            request = json.loads(self.rfile.read(length))
            assert set(request) == {
                "creator_signature",
                "delegation_canonical_bytes",
                "model_manifest_ids",
                "package_release_ids",
            }
            delegation_bytes = base64.b64decode(
                request["delegation_canonical_bytes"], validate=True
            )
            signature_text = request["creator_signature"]
            signature = base64.urlsafe_b64decode(signature_text + "=" * (-len(signature_text) % 4))
            Ed25519PublicKey.from_public_bytes(_Origin.creator_public_key).verify(
                signature, delegation_bytes
            )
            delegation = documents.read(delegation_bytes, pb.ArtifactDelegation)
            packages = request["package_release_ids"]
            models = request["model_manifest_ids"]
            assert packages == sorted(set(packages))
            assert models == sorted(set(models))
            assert set(packages).issubset(delegation.get("package_release_ids", []))
            assert set(models).issubset(delegation.get("model_manifest_ids", []))
            now = int(time.time())
            expires = min(int(delegation["expires_at_unix"]), now + 600)
            assert expires > now
        except Exception as exc:
            self._json(
                400,
                {
                    "error": {"code": "invalid", "message": type(exc).__name__},
                    "server_time_unix": int(time.time()),
                },
            )
            return
        self._json(
            200,
            {
                "artifacts": [
                    {
                        "digest": spelled,
                        "kind": _Origin.kinds[spelled],
                        "length": len(raw),
                        "url": (
                            f"https://worker.tensorhub.com:{ARTIFACT_PORT}/artifacts/{spelled[7:]}"
                        ),
                    }
                    for spelled, raw in sorted(_Origin.objects.items())
                ],
                "expires_at_unix": expires,
                "server_time_unix": now,
            },
        )

    def _json(self, status: int, value: object) -> None:
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_artifacts(
    release: Release,
    certificate: Path,
    key: Path,
    worker_certificate: Path,
    creator_public_key: bytes,
) -> None:
    _Origin.objects = dict(release.objects)
    _Origin.kinds = dict(release.kinds)
    source = next(digest for digest, kind in release.kinds.items() if kind == "wheel")
    models = next(digest for digest, kind in release.kinds.items() if kind == "model_object_set")
    _Origin.overlap_subjects = frozenset({source, models})
    _Origin.overlap_barrier = threading.Barrier(2)
    _Origin.overlapped = set()
    _Origin.delayed_model_subject = models
    _Origin.creator_public_key = creator_public_key
    _Origin.worker_certificate_der = ssl.PEM_cert_to_DER_cert(worker_certificate.read_text())
    _Origin.resolved_mtls = False
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 443), _Origin)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(certificate), str(key))
    context.load_verify_locations(cafile=str(worker_certificate))
    context.verify_mode = ssl.CERT_REQUIRED
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    artifact_server = http.server.ThreadingHTTPServer(("127.0.0.1", ARTIFACT_PORT), _Origin)
    artifact_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    artifact_context.load_cert_chain(str(certificate), str(key))
    artifact_server.socket = artifact_context.wrap_socket(artifact_server.socket, server_side=True)
    threading.Thread(target=artifact_server.serve_forever, daemon=True).start()


def overlap_proven() -> bool:
    return _Origin.overlapped == set(_Origin.overlap_subjects)


def resolve_mtls_proven() -> bool:
    return _Origin.resolved_mtls
