"""CPU storage proof for an incompressible ChromeTrace larger than the old1GiB limit.

This exercises the real authored attempt and Outputs.save_tree, not native admission.
The separate ordinary-CLI CPU proof must qualify the worker's output-capacity grant.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import os
import resource
import time
from pathlib import Path
from typing import Annotated

import msgspec

from cozy_runtime.author import App, AssetBound, Invocation, Outputs, Tree, attempt


class Request(msgspec.Struct):
    pass


class Result(msgspec.Struct):
    trace: Annotated[Tree, AssetBound(max_bytes=2 << 30)]


def verify(directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=False)
    module_path = Path(__file__).parent / "h3_request_profile/instrument.py"
    spec = importlib.util.spec_from_file_location("trace_recorder_storage", module_path)
    assert spec and spec.loader
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)
    raw = directory / "trace.json"
    # Repeat distances exceed the gzip window. One bounded random block avoids a
    # large random-memory allocation while retaining >1GiB after compression.
    payload = base64.b64encode(os.urandom(1 << 20))
    digest = hashlib.sha256()
    started = time.perf_counter()
    with raw.open("xb") as target:

        def write(data: bytes) -> None:
            target.write(data)
            digest.update(data)

        write(b'{"traceEvents":[')
        for index in range(1100):
            if index:
                write(b",")
            write(
                b'{"name":"storage-proof","ph":"i","ts":'
                + str(index).encode()
                + b',"args":{"payload":"'
            )
            write(payload)
            write(b'"}}')
        write(b"]}")
    app = App()
    sizes = {}

    @app.job
    def retain(payload: Request, out: Outputs) -> Result:
        nonlocal sizes
        tree, sizes = recorder.retain_trace_file(raw, out.save_tree)
        return Result(tree)

    result, outcome, _ = attempt(
        app.get("retain"),
        {},
        Invocation(
            "large-trace-storage",
            directory / "spool",
            time.monotonic() + 900,
            max_output_bytes=2 << 30,
        ),
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None
    retained = result.result.trace.path / "trace.json.gz"
    assert retained.stat().st_size == sizes["gzip_bytes"] > 1 << 30
    recovered = hashlib.sha256()
    with gzip.open(retained, "rb") as source:
        while chunk := source.read(1 << 20):
            recovered.update(chunk)
    assert recovered.digest() == digest.digest()
    with retained.open("rb") as source:
        retained_digest = hashlib.file_digest(source, "sha256").hexdigest()
    receipt = {
        "scope": "real authored CPU attempt, file-backed gzip and Outputs.save_tree custody",
        "native_admission_qualified": False,
        "sizes": sizes,
        "retained_path": str(retained),
        "retained_sha256": retained_digest,
        "decoded_sha256": recovered.hexdigest(),
        "decoded_matches_complete_input": True,
        "wall_seconds": time.perf_counter() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
