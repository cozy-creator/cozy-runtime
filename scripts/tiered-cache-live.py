#!/usr/bin/env python3
"""Real TensorFS proof for the worker-local store plus optional network cache."""

from __future__ import annotations

import hashlib
import io
import json
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import mock

import tensorfs

from cozy_runtime.internal import provisioned, pull
from cozy_runtime.internal.placement_materialization import (
    LOCAL_ARTIFACT_STORE_ROOT,
    CacheMount,
    _open_cache_store,
    materialize_objects,
)
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker import acquire

# TensorFS's smallest current cozytensors/1 vector: one inline bf16 tensor and its nested
# plain encoding. It has no external body object, so this proof can add ordinary file blobs
# without growing a second model-fixture author.
HEADER = bytes.fromhex(
    "856d636f7a7974656e736f72732f318080818581038081836576616c756581008100808081826b7472"
    "616e73666f726d657281856f626c6f636b732e302e7765696768740381020081846576616c75650381"
    "02440001ff7f"
)


def _ref(raw: bytes) -> provisioned.ArtifactRef:
    return provisioned.ArtifactRef("sha256:" + hashlib.sha256(raw).hexdigest(), len(raw))


def _put(store: Any, ref: provisioned.ArtifactRef, raw: bytes) -> None:
    store.put_reader(io.BytesIO(raw), ref.digest, ref.length)


def _closure(
    name: str, payload: bytes | None = None
) -> tuple[bytes, provisioned.ArtifactRef, tuple[provisioned.ArtifactRef, ...], dict[str, bytes]]:
    header = _ref(HEADER)
    entries: list[dict[str, object]] = [
        {
            "blob": {"length": header.length, "sha256": header.digest[7:]},
            "kind": "cozytensors",
            "path": f"{name}.cozytensors",
        }
    ]
    objects = [header]
    raws = {header.digest: HEADER}
    if payload is not None:
        file_ref = _ref(payload)
        entries.append(
            {
                "blob": {"length": file_ref.length, "sha256": file_ref.digest[7:]},
                "kind": "file",
                "path": f"{name}.bin",
            }
        )
        objects.append(file_ref)
        raws[file_ref.digest] = payload
    entries.sort(key=lambda row: str(row["path"]))
    manifest = json.dumps({"entries": entries}, sort_keys=True, separators=(",", ":")).encode()
    root = _ref(manifest)
    raws[root.digest] = manifest
    return manifest, root, tuple(objects), raws


def _put_closure(
    store: Any,
    manifest: bytes,
    root: provisioned.ArtifactRef,
    objects: tuple[provisioned.ArtifactRef, ...],
    raws: Mapping[str, bytes],
) -> None:
    for ref in objects:
        _put(store, ref, raws[ref.digest])
    store.put_manifest(manifest, root.digest, root.length)


class _Server(ThreadingHTTPServer):
    objects: dict[str, bytes]
    before_get: Callable[[str], None] | None
    barrier: threading.Barrier | None


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = cast(_Server, self.server)
        object_id = "sha256:" + self.path.removeprefix("/")
        if server.before_get is not None:
            server.before_get(object_id)
        barrier = server.barrier
        if barrier is not None:
            barrier.wait(timeout=5)
            server.barrier = None
        raw = server.objects[object_id]
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _cloud_hydrate(
    store: Any,
    root: provisioned.ArtifactRef,
    objects: tuple[provisioned.ArtifactRef, ...],
    cache_store: Any | None,
    server: _Server,
    *,
    session: str,
    streams: int = 1,
    on_object: Callable[[str, int, bool], None] | None = None,
) -> pull.PullResult:
    plan = pull._plan(
        tensorfs,
        store,
        session,
        root.digest[7:],
        root.length,
        [(item.digest, item.length) for item in objects],
    )
    result = pull.PullResult(
        ref=root.digest,
        manifest_id=root.digest[7:],
        store=str(store.root),
        streams=streams,
        declared=len(plan["held"]) + len(plan["wanted"]),
        held=len(plan["held"]),
        bytes_held=int(plan["held_bytes"]),
        bytes_total=int(plan["held_bytes"]) + int(plan["wanted_bytes"]),
        presence=dict(plan["presence"]),
        sources={str(object_id): "local" for object_id, _ in plan["held"]},
    )
    base = f"http://127.0.0.1:{server.server_port}/"
    pull._execute_plan(
        tensorfs,
        store,
        plan,
        result,
        urls={ref.digest: base + ref.digest[7:] for ref in (root, *objects)},
        in_hand={},
        deadline=pull.Deadline(),
        allow_local=True,
        streams=streams,
        on_object=on_object,
        no_proxy=True,
        cache_store=cache_store,
    )
    return result


def main() -> int:
    server = _Server(("127.0.0.1", 0), _Handler)
    server.objects, server.before_get, server.barrier = {}, None, None
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        with TemporaryDirectory(prefix="cozy-tiered-cache-") as raw_root:
            root = Path(raw_root)
            cache_mount = root / "network-volume"
            cache_mount.mkdir()
            cache_root = cache_mount / "tensorfs"
            cache_store = tensorfs.Store.init(str(cache_root))

            # Two typed Manifests share one header blob and use no network.
            first, second = _closure("first"), _closure("second")
            for closure in (first, second):
                _put_closure(cache_store, *closure)
            _, first_root, first_objects, first_raws = first
            _, second_root, second_objects, _ = second
            object_set = provisioned.ArtifactObjectSet(
                tuple(
                    sorted(
                        (
                            provisioned.ModelRoot(first_root, first_objects),
                            provisioned.ModelRoot(second_root, second_objects),
                        ),
                        key=lambda group: group.root.digest,
                    )
                )
            )
            unique = {ref.digest: ref for ref in (first_root, second_root, *first_objects)}
            warm = materialize_objects(
                {digest: "https://blocked.invalid" for digest in unique},
                object_set,
                cache=CacheMount(cache_mount, "cache-live", "provider-volume-live"),
                deadline=None,
                store_root=root / "warm-local",
            )
            observation = acquire.AcquisitionObservation()
            observation.start("model")
            acquire.observe_model_sources(observation, warm)
            observation.finish("model")
            cached_bytes = sum(ref.length for ref in unique.values())
            assert warm.sources == {
                "cloud": {"bytes": 0, "objects": 0},
                "local": {"bytes": 0, "objects": 0},
                "network_cache": {"bytes": cached_bytes, "objects": len(unique)},
            }
            assert observation.snapshot().model.reused_bytes == cached_bytes
            warm_store = tensorfs.Store.open(str(root / "warm-local"))
            warm_store.manifest(first_root.digest)
            warm_store.manifest(second_root.digest)
            assert warm_store.verify(first_objects[0].digest)["verdict"] == "verified"

            # Exercise the production wrapper with no cache; only this proof rewrites its
            # closed HTTPS grants to the local byte server.
            server.objects = dict(first_raws)

            def local_cloud_pull(
                tensorfs_module: Any,
                store: Any,
                manifest_id: str,
                manifest_length: int,
                granted: list[pull.Read],
                **kwargs: Any,
            ) -> pull.PullResult:
                assert tensorfs_module is tensorfs
                assert manifest_id == first_root.digest and manifest_length == first_root.length
                assert {row.object_id for row in granted} == set(first_raws)
                return _cloud_hydrate(
                    store,
                    first_root,
                    first_objects,
                    None,
                    server,
                    session=str(kwargs.get("session") or "cloud-only"),
                    streams=int(kwargs.get("streams") or 1),
                    on_object=kwargs.get("on_object"),
                )

            assert Path("/var/lib/cozy/store") == LOCAL_ARTIFACT_STORE_ROOT
            with mock.patch.object(pull, "pull_granted_manifest", side_effect=local_cloud_pull):
                cloud_only = materialize_objects(
                    {
                        ref.digest: f"https://delivery.invalid/{ref.digest[7:]}"
                        for ref in (first_root, *first_objects)
                    },
                    provisioned.ArtifactObjectSet(
                        (provisioned.ModelRoot(first_root, first_objects),)
                    ),
                    cache=None,
                    deadline=None,
                    store_root=root / "cloud-only-local",
                )
            cloud_bytes = sum(ref.length for ref in (first_root, *first_objects))
            assert cloud_only.sources["cloud"] == {"bytes": cloud_bytes, "objects": 2}
            tensorfs.Store.open(str(root / "cloud-only-local")).manifest(first_root.digest)

            # A corrupt cache blob is quarantined, fetched, and backfilled. The typed
            # Manifest and header remain cache hits.
            manifest, corrupt_root, corrupt_objects, corrupt_raws = _closure(
                "corrupt", b"correct-cloud-object"
            )
            _put_closure(cache_store, manifest, corrupt_root, corrupt_objects, corrupt_raws)
            payload_ref = corrupt_objects[1]
            corrupt_path = next(
                path
                for path in (cache_root / "blobs").rglob(payload_ref.digest[7:])
                if path.is_file()
            )
            corrupt_path.chmod(0o644)
            corrupt_path.write_bytes(b"X" * payload_ref.length)
            corrupt_path.chmod(0o444)
            local = tensorfs.Store.init(str(root / "corrupt-local"))
            server.objects = dict(corrupt_raws)

            def not_published_early(object_id: str) -> None:
                if object_id == payload_ref.digest:
                    assert not local.contains(payload_ref.digest)

            server.before_get = not_published_early
            repaired = _cloud_hydrate(
                local,
                corrupt_root,
                corrupt_objects,
                cache_store,
                server,
                session="corrupt-fallback",
            )
            server.before_get = None
            assert repaired.sources[payload_ref.digest] == "cloud"
            assert "corrupt" in repaired.cache_degradations
            assert cache_store.document(payload_ref.digest) == corrupt_raws[payload_ref.digest]
            assert list((cache_root / ".tensorfs" / "quarantine").iterdir())

            # One cold fill backfills a fresh cache; another store then stays cloud-free.
            _, cold_root, cold_objects, cold_raws = _closure("cold", b"cold-cloud-object")
            server.objects = dict(cold_raws)
            cold_cache = tensorfs.Store.init(str(root / "cold-cache"))
            cold_result = _cloud_hydrate(
                tensorfs.Store.init(str(root / "cold-local")),
                cold_root,
                cold_objects,
                cold_cache,
                server,
                session="cold-cloud",
            )
            assert set(cold_result.sources.values()) == {"cloud"}
            warm_result = _cloud_hydrate(
                tensorfs.Store.init(str(root / "fresh-local")),
                cold_root,
                cold_objects,
                cold_cache,
                server,
                session="warm-cloud-blocked",
            )
            assert set(warm_result.sources.values()) == {"network_cache"}

            # Concurrent first writers converge at the exact typed keys.
            _, concurrent_root, concurrent_objects, concurrent_raws = _closure(
                "concurrent", b"concurrent-cloud-object"
            )
            server.objects = dict(concurrent_raws)
            concurrent_cache = tensorfs.Store.init(str(root / "concurrent-cache"))
            server.barrier = threading.Barrier(2)

            def fill(index: int) -> pull.PullResult:
                return _cloud_hydrate(
                    tensorfs.Store.init(str(root / f"concurrent-local-{index}")),
                    concurrent_root,
                    concurrent_objects,
                    concurrent_cache,
                    server,
                    session=f"concurrent-{index}",
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                concurrent_results = list(pool.map(fill, range(2)))
            concurrent_cache.manifest(concurrent_root.digest)
            assert all(result.fetched == 3 for result in concurrent_results)

            # Cache failure only loses acceleration; local admission remains complete.
            readonly_root = root / "readonly-cache"
            readonly_cache = tensorfs.Store.init(str(readonly_root))
            for path in sorted(readonly_root.rglob("*"), reverse=True):
                path.chmod(0o500 if path.is_dir() else 0o400)
            readonly_root.chmod(0o500)
            _, readonly_manifest, readonly_objects, readonly_raws = _closure(
                "readonly", b"readonly-cloud-object"
            )
            server.objects = dict(readonly_raws)
            readonly_store = tensorfs.Store.init(str(root / "readonly-local"))
            result = _cloud_hydrate(
                readonly_store,
                readonly_manifest,
                readonly_objects,
                readonly_cache,
                server,
                session="readonly-cache",
            )
            readonly_root.chmod(0o700)
            for path in readonly_root.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
            readonly_store.manifest(readonly_manifest.digest)
            assert "backfill_unavailable" in result.cache_degradations

            unavailable_mount = root / "unavailable-cache"
            unavailable_mount.write_bytes(b"not-a-directory")
            degradations: set[str] = set()
            unavailable = _open_cache_store(
                tensorfs,
                unavailable_mount,
                unavailable_mount / "tensorfs",
                degradations,
            )
            assert unavailable is None and degradations == {"unavailable"}

            # A cache/local alias refuses before either store is created.
            alias_mount = root / "alias"
            alias_mount.mkdir()
            alias_store = alias_mount / "tensorfs"
            try:
                materialize_objects(
                    {},
                    provisioned.ArtifactObjectSet(()),
                    cache=CacheMount(alias_mount, "cache-alias", "provider-volume-alias"),
                    deadline=None,
                    store_root=alias_store,
                )
            except LaunchRefusal as exc:
                assert exc.code == "artifact_store_alias"
            else:
                raise AssertionError("network cache was accepted as the local store")
            assert not alias_store.exists()
            assert (cache_root / "tensorfs.sqlite").is_file()
            assert {path.name for path in cache_root.iterdir()} <= {
                ".tensorfs",
                "blobs",
                "checkouts",
                "manifests",
                "repos",
                "tensorfs.sqlite",
            }
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    print("tiered-cache live: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
