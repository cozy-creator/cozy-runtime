"""cr-049's hub stand-in: author one endpoint release, then serve its exact bytes.

This module plays the two roles the pod is not: the BUILD side that authors a release's
canonical documents (tensorhub's `releaseassembly` + `wheelset` + `placementset`, whose Go
authors produce these same shapes), and the ARTIFACT ORIGIN those documents are fetched
from. It is deliberately a repository proof driver and not a shipped component — what it
proves is the WORKER's half: that a pod which boots empty converges to a live placement from
a `DesiredPlacementSet` plus an `ArtifactGrantUpdate` and nothing else.

The receipt is derived from exact base-family and endpoint-wheel identities. This module
materializes the overlay once to learn it; the worker independently materializes the same
overlay and must produce byte-identical receipt bytes without copying the base environment.
"""

from __future__ import annotations

import http.server
import ssl
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from proof_data import Wheel, canonical, digest, installed_base_ownership, inventory, ref, wheel

from cozy_runtime.internal import canonical as canon
from cozy_runtime.internal import descriptor as desc
from cozy_runtime.internal import local, overlay
from cozy_runtime.internal.discovery import discover_installed
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

ENDPOINT_RELEASE = "proof-boot-release-1"
PLACEMENT_ID = "cr049-first-placement"


@dataclass
class Release:
    """Every exact byte string this release is made of, keyed by its own digest."""

    objects: dict[str, bytes] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    wheel_paths: dict[str, Path] = field(default_factory=dict)
    environment_spec: bytes = b""
    endpoint_bundle: bytes = b""
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

    def subjects(self) -> list[pb.ArtifactSubject]:
        """Sorted by digest, exactly the bytes this release authorizes and no others."""

        return [
            pb.ArtifactSubject(
                digest=documents.raw(spelled),
                subject_id=spelled,
                kind=self.kinds[spelled],
                length=len(raw),
            )
            for spelled, raw in sorted(self.objects.items())
        ]

    def locations(self, base: str) -> list[pb.ArtifactLocation]:
        return [
            pb.ArtifactLocation(
                digest=documents.raw(spelled), url=f"{base}/artifacts/{spelled[7:]}"
            )
            for spelled in sorted(self.objects)
        ]


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
    """Author the whole closure a `PlacementSpec` reaches, and measure its receipt."""

    release = Release()
    project = release.add_wheel(project_wheel)
    house = sorted((inventory(path) for path in wheelhouse_wheels), key=lambda e: e.distribution)
    if len(house) != 5:
        raise AssertionError(f"boot proof base carries {len(house)} control wheels, want 5")

    base_distributions, base_import_roots = installed_base_ownership()
    profile = {
        "accelerator_build": target["accelerator_abi"],
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
                "minimum_cuda_version": "12.6",
                "minimum_driver_version": "560.28.03",
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
    descriptor_bytes = release.add(desc.canonical_bytes(document), "descriptor")
    evaluated = release.add(canonical({"format": "proof.EvaluatedConfig/1"}), "config")
    # These exact refs prove publication provenance but are deliberately not
    # runtime grant subjects. The executable source payload is project_wheel.
    source_archive = b"cr-049-boot-proof-source"
    source_lock = b"cr-049-boot-proof-lock"
    bundle = release.add(
        canonical(
            {
                "descriptor": ref(descriptor_bytes),
                "endpoint_release_id": ENDPOINT_RELEASE,
                "evaluated_config": ref(evaluated),
                "format": "tensorhub.endpoint_bundle/2",
                "project_wheel": wheel(project, import_roots=True),
                "resolved_wheel_set": ref(resolved),
                "source_archive": ref(source_archive),
                "source_lock": ref(source_lock),
                "wheelhouse_manifest": ref(wheelhouse),
            }
        ),
        "bundle",
    )
    spec = release.add(
        documents.canonical_bytes(
            pb.EndpointEnvironmentSpec(
                compatibility_profile=pb.CompatibilityProfile(**profile),
                platform_target=pb.PlatformTarget(**target),
                endpoint_bundle_digest=documents.digest_of(bundle),
                project_wheel_digest=documents.raw(project.digest),
                wheelhouse_manifest_digest=documents.digest_of(wheelhouse),
            )
        ),
        "environment",
    )
    release.environment_spec = spec
    release.endpoint_bundle = bundle
    release.resolved_wheel_set = resolved
    release.wheelhouse_manifest = wheelhouse
    release.descriptor = descriptor_bytes

    # The image builder would bake this exact small family manifest beside the installed
    # official PyTorch environment. This proof writes the stand-in before either independent
    # overlay materialization and then treats it as read-only.
    base_manifest.parent.mkdir(parents=True, exist_ok=True)
    base_manifest.write_bytes(wheelhouse)
    base_manifest.chmod(0o444)

    from cozy_runtime.internal.materialize_launch import endpoint_overlay_input

    source = endpoint_overlay_input(
        environment_spec_bytes=spec,
        endpoint_bundle_bytes=bundle,
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

    # The binding-plan closure. A weightless endpoint's plans, model-root document and
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
            "spec": {
                "binding_plans": [
                    {
                        "digest": plan_digest,
                        "kind": "plan",
                        "length": length,
                        "subject_id": plan_digest,
                    }
                    for plan_digest, length in release.plan_refs
                ],
                "descriptor_digest": digest(descriptor_bytes),
                "endpoint_release_id": ENDPOINT_RELEASE,
                "environment_spec_digest": digest(spec),
                "installed_environment_receipt_digest": digest(installed.receipt_bytes),
                "model_object_set": {
                    "digest": digest(model_object_set),
                    "kind": "model_object_set",
                    "length": len(model_object_set),
                    "subject_id": digest(model_object_set),
                },
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
    overlap_subjects: ClassVar[frozenset[str]] = frozenset()
    overlap_barrier: ClassVar[threading.Barrier | None] = None
    overlapped: ClassVar[set[str]] = set()
    delayed_model_subject: ClassVar[str] = ""
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
                # Keep the model leg observably live after the endpoint wheel is admitted;
                # the Runtime proof must spawn its executor during this window.
                time.sleep(2)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_artifacts(release: Release, certificate: Path, key: Path, port: int) -> None:
    _Origin.objects = dict(release.objects)
    source = next(digest for digest, kind in release.kinds.items() if kind == "wheel")
    models = next(digest for digest, kind in release.kinds.items() if kind == "model_object_set")
    _Origin.overlap_subjects = frozenset({source, models})
    _Origin.overlap_barrier = threading.Barrier(2)
    _Origin.overlapped = set()
    _Origin.delayed_model_subject = models
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Origin)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(certificate), str(key))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def overlap_proven() -> bool:
    return _Origin.overlapped == set(_Origin.overlap_subjects)
