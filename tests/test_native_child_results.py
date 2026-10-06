"""Current SDK decodes verified native file/tree grants without a worker fixture."""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import msgspec
import pytest
from PIL import Image
from cozy_runtime import canonical_json
from cozy_runtime.author import FileAsset, ImageAsset, Tree
from cozy_runtime.author._call_results import decode
from cozy_runtime.author._executor_requests import CallState


class Result(msgspec.Struct):
    report: FileAsset
    bundle: Tree


class ImageResult(msgspec.Struct):
    image: ImageAsset


def png_result(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    path = root / "image.png"
    Image.new("RGB", (8, 8), (12, 34, 56)).save(path, format="PNG")
    data = path.read_bytes()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    value = {
        "image": {
            "asset_ref": digest,
            "kind": "image",
            "digest": digest,
            "size_bytes": len(data),
            "media_type": "image/png",
        }
    }
    rows = [
        {
            "output_id": "image",
            "kind": "image",
            "digest": digest,
            "length": len(data),
            "content_bytes": len(data),
            "local": str(path),
            "media_type": "image/png",
        }
    ]
    return value, rows, data


def retained(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    data = b"native file and tree bytes"
    path = root / "report"
    path.write_bytes(data)
    tree = root / "bundle"
    tree.mkdir()
    (tree / "payload").write_bytes(data)
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    manifest = canonical_json.encode(
        {
            "entries": [
                {
                    "path": "payload",
                    "kind": "file",
                    "blob": {"sha256": digest[7:], "length": len(data)},
                }
            ]
        }
    )
    tree_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    value = {
        "report": {
            "asset_ref": digest,
            "kind": "file",
            "digest": digest,
            "size_bytes": len(data),
            "media_type": "application/octet-stream",
        },
        "bundle": {
            "asset_ref": tree_digest,
            "kind": "tree",
            "digest": tree_digest,
            "size_bytes": len(data),
        },
    }
    rows = [
        {
            "output_id": "report",
            "kind": "file",
            "digest": digest,
            "length": len(data),
            "content_bytes": len(data),
            "local": str(path),
            "media_type": "application/octet-stream",
        },
        {
            "output_id": "bundle",
            "kind": "tree",
            "digest": tree_digest,
            "length": len(manifest),
            "content_bytes": len(data),
            "local": str(tree),
            "media_type": "application/vnd.cozy.tree-manifest",
        },
    ]
    return value, rows, data


def test_author_request_decoder_still_refuses_hydration_records() -> None:
    from cozy_runtime.author._assets import asset_dec_hook

    for target in (FileAsset, Tree):
        with pytest.raises(TypeError, match="reference string"):
            asset_dec_hook(target, {"asset_ref": "sha256:" + "00" * 32, "local": "/tmp"})


def test_encoded_png_result_hydrates_as_existing_image_asset(tmp_path: Path) -> None:
    value, grants, data = png_result(tmp_path)
    rows = grants
    assert rows[0]["media_type"] == "image/png"
    result, _ = decode(
        value,
        ImageResult,
        rows,
        request_id="parent",
        guard=lambda: None,
        observation=None,
    )
    assert isinstance(result.image, ImageAsset)
    assert result.image.media_type == "image/png"
    assert result.image.read_bytes() == data


def test_native_grants_and_result_leaves_tolerate_producer_additions(tmp_path: Path) -> None:
    value, grants, data = png_result(tmp_path)

    value["image"]["future"] = {"label": "newer producer"}

    answer = msgspec.to_builtins(CallState(ok=True, state="succeeded", byte_grants=tuple(grants)))
    answer["byte_grants"][0]["future"] = ["newer worker"]
    received = msgspec.convert(answer, CallState, strict=True)
    assert received.byte_grants == ({**grants[0], "future": ["newer worker"]},)
    result, _ = decode(
        value,
        ImageResult,
        received.byte_grants,
        request_id="parent",
        guard=lambda: None,
        observation=None,
    )
    assert result.image.read_bytes() == data


@pytest.mark.parametrize(("index", "field"), [(0, "length"), (1, "content_bytes")])
def test_native_grant_reader_rejects_boolean_byte_counts(
    tmp_path: Path, index: int, field: str
) -> None:
    value, grants, _ = retained(tmp_path)

    answer = msgspec.to_builtins(CallState(ok=True, state="succeeded", byte_grants=tuple(grants)))
    answer["byte_grants"][index][field] = True
    received = msgspec.convert(answer, CallState, strict=True)
    with pytest.raises(msgspec.ValidationError, match="got `bool`"):
        decode(
            value,
            Result,
            received.byte_grants,
            request_id="parent",
            guard=lambda: None,
            observation=None,
        )


def test_file_grant_without_tree_byte_count_still_hydrates(tmp_path: Path) -> None:
    value, grants, data = png_result(tmp_path)

    answer = msgspec.to_builtins(CallState(ok=True, state="succeeded", byte_grants=tuple(grants)))
    del answer["byte_grants"][0]["content_bytes"]
    received = msgspec.convert(answer, CallState, strict=True)
    result, _ = decode(
        value,
        ImageResult,
        received.byte_grants,
        request_id="parent",
        guard=lambda: None,
        observation=None,
    )
    assert result.image.read_bytes() == data


def test_encoded_png_mislabeled_as_jpeg_refuses_at_result_decode(tmp_path: Path) -> None:
    value, grants, data = png_result(tmp_path)

    value["image"]["media_type"] = "image/jpeg"

    rows = grants
    # Materialization corrects same-family MIME claims from the bytes. A native
    # result must still agree with its verified grant when the SDK hydrates it.
    assert rows[0]["media_type"] == "image/png"
    assert Path(rows[0]["local"]).read_bytes() == data
    with pytest.raises(ValueError, match="differs from its verified host grant"):
        decode(value, ImageResult, rows, request_id="parent", guard=lambda: None, observation=None)


def test_native_grant_reader_ignores_fields_unused_by_its_kind(tmp_path: Path) -> None:
    value, grants, data = retained(tmp_path)

    answer = msgspec.to_builtins(CallState(ok=True, state="succeeded", byte_grants=tuple(grants)))
    answer["byte_grants"][0]["content_bytes"] = {"newer": "unused on file"}
    answer["byte_grants"][1]["media_type"] = ["unused on tree"]
    answer["byte_grants"][1]["length"] = {"newer": "unused on tree"}

    value["bundle"]["media_type"] = {"newer": "unused on tree result"}
    received = msgspec.convert(answer, CallState, strict=True)
    result, _ = decode(
        value,
        Result,
        received.byte_grants,
        request_id="parent",
        guard=lambda: None,
        observation=None,
    )
    assert result.report.read_bytes() == (result.bundle.path / "payload").read_bytes() == data
