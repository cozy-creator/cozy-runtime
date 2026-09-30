"""Verify and materialize native child results on the parent's worker."""

from __future__ import annotations

import hashlib
from pathlib import Path

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._call_results import ByteResult, FileResult, result_at
from cozy_runtime.author._executor_requests import ByteGrant
from cozy_runtime.author._media import KIND_MEDIA, SNIFF_BYTES, admits, normalize, reconcile, sniff
from cozy_runtime.internal.worker import byte_inputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def validate(result: pb.ChildCallResult) -> None:
    if len(result.byte_result_grants) > 32:
        raise WorkspaceRefusal("native child result inventory exceeds its bound")
    seen: set[str] = set()
    for grant in result.byte_result_grants:
        if grant.output_id in seen or len(grant.output_id.encode()) > 1024:
            raise WorkspaceRefusal("native child result repeats or exceeds an output ID")
        seen.add(grant.output_id)
        byte_inputs.validate_source(grant.source, grant.retention_id)
    captured = result.HasField("observation") and result.observation.HasField("capture")
    if ("runtime.capture" in seen) != captured:
        raise WorkspaceRefusal("capture observation and native grant disagree")
    if result.byte_result_grants and result.state != pb.CHILD_CALL_STATE_SUCCEEDED:
        raise WorkspaceRefusal("a non-successful child result cannot grant outputs")


def materialize(
    workspace: Workspace,
    owner: str,
    result: pb.ChildCallResult,
    spool: Path,
    *,
    max_bytes: int = byte_inputs.MAX_BYTES,
) -> list[ByteGrant]:
    validate(result)
    if sum(grant.source.content_bytes for grant in result.byte_result_grants) > max_bytes:
        raise WorkspaceRefusal("native child results exceed the parent's byte bound")
    value = canonical_json.decode(result.result_canonical_bytes)
    rows: list[ByteGrant] = []
    for index, grant in enumerate(result.byte_result_grants):
        capture = grant.output_id == "runtime.capture"
        node = None if capture else result_at(value, grant.output_id)
        if capture:
            if grant.source.content_bytes > 128 << 20:
                raise WorkspaceRefusal("capture exceeds its byte bound")
            kind, media_type = "tree", byte_inputs.TREE_MIME
            digest = documents.spell(grant.source.manifest.digest)
            length = grant.source.manifest.length
        else:
            if not isinstance(node, dict) or node.get("kind") not in (*KIND_MEDIA, "tree"):
                raise WorkspaceRefusal("native grant no longer names a typed byte result")
            try:
                declared = msgspec.convert(node, ByteResult, strict=True)
            except msgspec.ValidationError as exc:
                raise WorkspaceRefusal("native child result changed its byte identity") from exc
            kind = declared.kind
            digest, length = declared.digest, declared.size_bytes
            if length < 0 or declared.asset_ref != digest or length != grant.source.content_bytes:
                raise WorkspaceRefusal("native child result changed its byte identity")
            if kind == "tree":
                if digest != documents.spell(grant.source.manifest.digest):
                    raise WorkspaceRefusal("native child tree changed its manifest")
                length = grant.source.manifest.length
                media_type = byte_inputs.TREE_MIME
            else:
                try:
                    media_type = msgspec.convert(node, FileResult, strict=True).media_type
                except msgspec.ValidationError as exc:
                    raise WorkspaceRefusal("native child file omitted its media type") from exc
                if not admits(kind, normalize(media_type)):
                    raise WorkspaceRefusal("native child file omitted its media type")
        destination = spool / "child-results" / str(result.call_index) / str(index)
        held = pb.NativeByteRetentionRequest(source=grant.source, retention_id=grant.retention_id)
        members = byte_inputs.copy_retained(
            workspace,
            owner,
            held,
            destination,
            max_bytes=max_bytes,
            expected_blob=(digest, length) if kind != "tree" else None,
        )
        if kind != "tree":
            with (destination / "payload").open("rb") as stream:
                actual = sniff(stream.read(SNIFF_BYTES))
            reconciled = reconcile(media_type, actual)
            if reconciled is None:
                raise WorkspaceRefusal("native child media type differs from its encoded bytes")
            media_type = reconciled
        if capture:
            if {member.path for member in members} != {"capture.json", "sketches.f32"}:
                raise WorkspaceRefusal("capture tree does not contain exactly its two files")
            semantic = hashlib.sha256()
            for name in ("capture.json", "sketches.f32"):
                with (destination / name).open("rb") as stream:
                    while block := stream.read(1 << 20):
                        semantic.update(block)
            if semantic.digest() != result.observation.capture.content_digest:
                raise WorkspaceRefusal("capture content differs from its semantic identity")
        rows.append(
            {
                "output_id": grant.output_id,
                "kind": kind,
                "digest": digest,
                "length": length,
                "content_bytes": grant.source.content_bytes,
                "media_type": media_type,
                "local": str(destination / "payload" if kind != "tree" else destination),
            }
        )
    return rows
