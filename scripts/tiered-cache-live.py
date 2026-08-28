#!/usr/bin/env python3
"""Real-TensorFS proof for the worker-local CAS plus optional network cache."""

from __future__ import annotations

import hashlib
import io
import shutil
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
from cozy_runtime.internal.materialize_launch import (
    LOCAL_ARTIFACT_STORE_ROOT,
    CacheMount,
    LaunchRefusal,
    _open_cache_store,
    materialize_objects,
)


def _ref(raw: bytes) -> provisioned.ArtifactRef:
    return provisioned.ArtifactRef("sha256:" + hashlib.sha256(raw).hexdigest(), len(raw))


def _put(store: Any, ref: provisioned.ArtifactRef, raw: bytes) -> None:
    store.put_reader(io.BytesIO(raw), ref.digest, ref.length)


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
        if server.barrier is not None:
            server.barrier.wait(timeout=5)
        raw = server.objects[object_id]
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _cloud_hydrate(
    store: Any,
    ref: provisioned.ArtifactRef,
    cache_store: Any | None,
    url: str,
    *,
    session: str,
    objects: tuple[provisioned.ArtifactRef, ...] = (),
    object_urls: Mapping[str, str] | None = None,
    streams: int = 1,
    on_object: Callable[[str, int, bool], None] | None = None,
) -> pull.PullResult:
    root = ref.digest.removeprefix("sha256:") if objects else ref.digest
    declared = objects or (ref,)
    plan = pull._plan(
        tensorfs,
        store,
        session,
        root,
        ref.length,
        [(item.digest, item.length) for item in declared],
    )
    result = pull.PullResult(
        ref=ref.digest,
        snapshot=ref.digest.removeprefix("sha256:"),
        store=str(store.root),
        streams=streams,
        declared=len(plan["held"]) + len(plan["wanted"]),
        held=len(plan["held"]),
        bytes_held=int(plan["held_bytes"]),
        bytes_total=int(plan["held_bytes"]) + int(plan["wanted_bytes"]),
        presence=dict(plan["presence"]),
        sources={str(object_id): "local" for object_id, _ in plan["held"]},
    )
    pull._execute_plan(
        tensorfs,
        store,
        plan,
        result,
        urls={ref.digest: url, **(object_urls or {})},
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
    server.objects = {}
    server.before_get = None
    server.barrier = None
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        with TemporaryDirectory(prefix="cozy-tiered-cache-") as raw_root:
            root = Path(raw_root)
            cache_mount = root / "network-volume"
            cache_mount.mkdir()
            cache_root = cache_mount / "tensorfs"
            cache_store = tensorfs.Store.init(str(cache_root))

            # A complete warm model closure uses no cloud and becomes an independent local CAS.
            manifest = b'{"format":"fixture.snapshot/1"}'
            second_manifest = b'{"format":"fixture.snapshot/1","root":2}'
            segment = b"warm-network-cache-segment"
            manifest_ref, second_manifest_ref, segment_ref = (
                _ref(manifest),
                _ref(second_manifest),
                _ref(segment),
            )
            _put(cache_store, manifest_ref, manifest)
            _put(cache_store, second_manifest_ref, second_manifest)
            _put(cache_store, segment_ref, segment)
            object_set = provisioned.ArtifactObjectSet(
                tuple(
                    sorted(
                        (
                            provisioned.ModelRoot(manifest_ref, (segment_ref,)),
                            provisioned.ModelRoot(second_manifest_ref, (segment_ref,)),
                        ),
                        key=lambda group: group.root.digest,
                    )
                )
            )
            grants = {
                ref.digest: "https://blocked.invalid"
                for ref in (manifest_ref, second_manifest_ref, segment_ref)
            }
            warm_local = root / "warm-local"
            warm = materialize_objects(
                grants,
                object_set,
                cache=CacheMount(cache_mount, "cache-live", "provider-volume-live"),
                deadline=None,
                store_root=warm_local,
            )
            assert warm.store == warm_local and warm.store != cache_root
            assert warm.sources == {
                "cloud": {"bytes": 0, "objects": 0},
                "local": {"bytes": 0, "objects": 0},
                "network_cache": {
                    "bytes": manifest_ref.length + second_manifest_ref.length + segment_ref.length,
                    "objects": 3,
                },
            }
            shutil.rmtree(cache_root)
            warm_store = tensorfs.Store.open(str(warm_local))
            assert (
                warm_store.verify(manifest_ref.digest.removeprefix("sha256:"))["verdict"]
                == "verified"
            )
            assert (
                warm_store.verify(segment_ref.digest.removeprefix("sha256:"))["verdict"]
                == "verified"
            )
            assert (
                warm_store.verify(second_manifest_ref.digest.removeprefix("sha256:"))["verdict"]
                == "verified"
            )
            cache_store = tensorfs.Store.init(str(cache_root))

            # No attached cache means no cache path operation: the exact cloud closure
            # lands in the same worker-local store used by production, with no degradation.
            assert Path("/var/lib/cozy/store") == LOCAL_ARTIFACT_STORE_ROOT
            cloud_only_set = provisioned.ArtifactObjectSet(
                (provisioned.ModelRoot(manifest_ref, (segment_ref,)),)
            )
            cloud_only_grants = {
                ref.digest: f"https://delivery.invalid/{ref.digest[7:]}"
                for ref in (manifest_ref, segment_ref)
            }
            server.objects = {
                manifest_ref.digest: manifest,
                segment_ref.digest: segment,
            }

            def local_cloud_pull(
                tensorfs_module: Any,
                store: Any,
                snapshot: str,
                snapshot_length: int,
                granted: list[pull.Read],
                *,
                timeout: float | None = None,
                session: str = "",
                streams: int = pull.STREAMS,
                on_object: Callable[[str, int, bool], None] | None = None,
                cache_store: Any | None = None,
                allow_local: bool = False,
            ) -> pull.PullResult:
                assert tensorfs_module is tensorfs
                assert timeout is None and cache_store is None and not allow_local
                rows = list(granted)
                assert snapshot == manifest_ref.digest and snapshot_length == manifest_ref.length
                assert {row.object_id for row in rows} == {
                    manifest_ref.digest,
                    segment_ref.digest,
                }
                return _cloud_hydrate(
                    store,
                    manifest_ref,
                    None,
                    f"http://127.0.0.1:{server.server_port}/{manifest_ref.digest[7:]}",
                    session=session,
                    objects=(segment_ref,),
                    object_urls={
                        segment_ref.digest: (
                            f"http://127.0.0.1:{server.server_port}/{segment_ref.digest[7:]}"
                        )
                    },
                    streams=streams,
                    on_object=on_object,
                )

            cloud_only_local = root / "cloud-only-local"
            with mock.patch.object(pull, "pull_granted_snapshot", side_effect=local_cloud_pull):
                cloud_only = materialize_objects(
                    cloud_only_grants,
                    cloud_only_set,
                    cache=None,
                    deadline=None,
                    store_root=cloud_only_local,
                )
            cloud_only_bytes = manifest_ref.length + segment_ref.length
            assert cloud_only.sources == {
                "cloud": {"bytes": cloud_only_bytes, "objects": 2},
                "local": {"bytes": 0, "objects": 0},
                "network_cache": {"bytes": 0, "objects": 0},
            }
            assert not cloud_only.network_cache_degradations
            cloud_only_store = tensorfs.Store.open(str(cloud_only_local))
            assert all(
                cloud_only_store.verify(ref.digest[7:])["verdict"] == "verified"
                for ref in (manifest_ref, segment_ref)
            )

            # A corrupt cache body commits nothing locally before the cloud fallback runs.
            corrupt_raw = b"correct-cloud-object"
            corrupt_ref = _ref(corrupt_raw)
            _put(cache_store, corrupt_ref, corrupt_raw)
            corrupt_path = next(
                path
                for path in cache_root.rglob(corrupt_ref.digest.removeprefix("sha256:"))
                if path.read_bytes() == corrupt_raw
            )
            corrupt_path.chmod(0o644)
            corrupt_path.write_bytes(b"X" * corrupt_ref.length)
            corrupt_path.chmod(0o444)
            corrupt_store = tensorfs.Store.init(str(root / "corrupt-local"))
            server.objects = {corrupt_ref.digest: corrupt_raw}

            def corrupt_not_local(object_id: str) -> None:
                assert not corrupt_store.contains(object_id.removeprefix("sha256:")), (
                    "corrupt cache published locally"
                )

            server.before_get = corrupt_not_local
            corrupt = _cloud_hydrate(
                corrupt_store,
                corrupt_ref,
                cache_store,
                f"http://127.0.0.1:{server.server_port}/{corrupt_ref.digest[7:]}",
                session="corrupt-fallback",
            )
            assert corrupt.sources == {corrupt_ref.digest: "cloud"}
            assert "corrupt" in corrupt.cache_degradations
            assert cache_store.document(corrupt_ref.digest) == corrupt_raw
            quarantined = list((cache_root / "quarantine").iterdir())
            assert (
                len(quarantined) == 1 and quarantined[0].read_bytes() == b"X" * corrupt_ref.length
            )

            # A cloud miss backfills one exact key; a fresh local CAS then runs cloud-blocked.
            cloud_raw = b"cold-cloud-object"
            cloud_ref = _ref(cloud_raw)
            server.objects = {cloud_ref.digest: cloud_raw}
            server.before_get = None
            cold_store = tensorfs.Store.init(str(root / "cold-local"))
            cold = _cloud_hydrate(
                cold_store,
                cloud_ref,
                cache_store,
                f"http://127.0.0.1:{server.server_port}/{cloud_ref.digest[7:]}",
                session="cold-cloud",
            )
            assert cold.sources == {cloud_ref.digest: "cloud"}
            assert cache_store.document(cloud_ref.digest) == cloud_raw
            fresh_store = tensorfs.Store.init(str(root / "fresh-local"))
            fresh = _cloud_hydrate(
                fresh_store,
                cloud_ref,
                cache_store,
                "http://127.0.0.1:1/cloud-is-blocked",
                session="warm-cloud-blocked",
            )
            assert fresh.sources == {cloud_ref.digest: "network_cache"}

            # Concurrent honest fills use create-if-absent publication of the same exact key.
            concurrent_raw = b"concurrent-cloud-object"
            concurrent_ref = _ref(concurrent_raw)
            server.objects = {concurrent_ref.digest: concurrent_raw}
            server.barrier = threading.Barrier(2)

            def concurrent(index: int) -> pull.PullResult:
                store = tensorfs.Store.init(str(root / f"concurrent-local-{index}"))
                return _cloud_hydrate(
                    store,
                    concurrent_ref,
                    cache_store,
                    f"http://127.0.0.1:{server.server_port}/{concurrent_ref.digest[7:]}",
                    session=f"concurrent-{index}",
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                concurrent_results = list(pool.map(concurrent, range(2)))
            server.barrier = None
            assert cache_store.document(concurrent_ref.digest) == concurrent_raw
            assert all(
                result.sources[concurrent_ref.digest] == "cloud" for result in concurrent_results
            )

            # Read-only and unavailable cache paths only lose acceleration.
            readonly_root = root / "readonly-cache"
            readonly_cache = tensorfs.Store.init(str(readonly_root))
            for path in sorted(readonly_root.rglob("*"), reverse=True):
                path.chmod(0o500 if path.is_dir() else 0o400)
            readonly_root.chmod(0o500)
            readonly_raw = b"readonly-cache-cloud-object"
            readonly_ref = _ref(readonly_raw)
            server.objects = {readonly_ref.digest: readonly_raw}
            readonly_store = tensorfs.Store.init(str(root / "readonly-local"))
            readonly = _cloud_hydrate(
                readonly_store,
                readonly_ref,
                readonly_cache,
                f"http://127.0.0.1:{server.server_port}/{readonly_ref.digest[7:]}",
                session="readonly-cache",
            )
            readonly_root.chmod(0o700)
            for path in readonly_root.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
            assert readonly_store.contains(readonly_ref.digest[7:])
            assert "backfill_unavailable" in readonly.cache_degradations

            unavailable_mount = root / "unavailable-cache"
            unavailable_mount.write_bytes(b"not-a-directory")
            unavailable_degradations: set[str] = set()
            unavailable_cache = _open_cache_store(
                tensorfs,
                unavailable_mount,
                unavailable_mount / "tensorfs",
                unavailable_degradations,
            )
            assert unavailable_cache is None and unavailable_degradations == {"unavailable"}
            unavailable_raw = b"unavailable-cache-cloud-object"
            unavailable_ref = _ref(unavailable_raw)
            server.objects = {unavailable_ref.digest: unavailable_raw}
            unavailable_store = tensorfs.Store.init(str(root / "unavailable-local"))
            unavailable = _cloud_hydrate(
                unavailable_store,
                unavailable_ref,
                unavailable_cache,
                f"http://127.0.0.1:{server.server_port}/{unavailable_ref.digest[7:]}",
                session="unavailable-cache",
            )
            assert unavailable_store.contains(unavailable_ref.digest[7:])
            assert not unavailable.cache_degradations

            # An alias refuses before creating either side, including the weightless case.
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
                raise AssertionError("network cache was accepted as the local CAS")
            assert not alias_store.exists()
            assert (cache_root / "store.json").is_file()
            assert {path.name for path in cache_root.iterdir()} <= {
                "objects",
                "quarantine",
                "store.json",
                "meta",
                "tmp",
                "verified",
            }
            assert not any(path.is_file() for path in (cache_root / "meta").rglob("*"))
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    print("tiered-cache live: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
