#!/usr/bin/env python3
"""Exact release/lane-only public model read route fence, with no live server."""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Mapping
from typing import Any

from cozy_runtime.internal import pull

MANIFEST = b'{"entries":[]}'
MANIFEST_ID = "sha256:" + hashlib.sha256(MANIFEST).hexdigest()


class Response:
    headers: Mapping[str, str] = {"Content-Length": str(len(MANIFEST))}

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return MANIFEST if size < 0 else MANIFEST[:size]


def main() -> None:
    seen: list[tuple[str, str]] = []

    def fake_json(
        base: str,
        path: str,
        deadline: pull.Deadline,
        *,
        token: str = "",
        body: Any = None,
    ) -> Any:
        del base, deadline, token
        seen.append(("POST" if body is not None else "GET", path))
        if path.startswith("/v1/models/resolve?"):
            return {
                "model": "proof/model",
                "release": "h3 final",
                "lane": "fp8+x",
                "manifest_id": MANIFEST_ID,
                "stamps": {},
            }
        return {
            "reads": [
                {
                    "object_id": MANIFEST_ID,
                    "length": len(MANIFEST),
                    "url": "https://objects.invalid/manifest",
                    "expires_at": "2099-01-01T00:00:00Z",
                }
            ]
        }

    original_json, original_open = pull._json, pull._open
    pull._json = fake_json

    def fake_open(
        req: urllib.request.Request,
        deadline: pull.Deadline,
        *,
        no_proxy: bool = False,
    ) -> Any:
        del deadline, no_proxy
        seen.append(("GET", str(req.full_url).removeprefix("http://127.0.0.1:1")))
        return Response()

    pull._open = fake_open
    try:
        base = "http://127.0.0.1:1"
        deadline = pull.Deadline.after(5)
        resolved = pull.resolve(base, "proof/model@h3 final", deadline, lane="fp8+x")
        assert pull.manifest(base, resolved, deadline, 1 << 20) == MANIFEST
        assert len(pull.reads(base, resolved, deadline)) == 1
    finally:
        pull._json, pull._open = original_json, original_open

    assert not any("/manifests/" in path for _, path in seen), seen
    assert seen == [
        ("GET", "/v1/models/resolve?ref=proof%2Fmodel%40h3+final&lane=fp8%2Bx"),
        ("GET", "/v1/models/proof/model/releases/h3%20final/lanes/fp8%2Bx/manifest"),
        ("POST", "/v1/models/proof/model/releases/h3%20final/lanes/fp8%2Bx/reads"),
    ], seen
    print("model release routes: exact release/lane manifest and reads only")


if __name__ == "__main__":
    main()
