"""Private fault plants below the public typed-input CLI acceptance proof.

These checks deliberately exercise the local RecordOwner and worker internals. They are not
evidence about the public CLI surface; `proofs/cli/run.sh` owns that contract.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.author import ImageAsset
from cozy_runtime.cli import payload as payload_grammar
from cozy_runtime.cli.io import CliError
from cozy_runtime.internal import descriptor
from cozy_runtime.internal.config import read_config
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import (
    LocalInput,
    LocalRequest,
    development_source_digest,
    run_slice,
)
from cozy_runtime.internal.worker import grants
from cozy_runtime.internal.worker.session import WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb


class ProofFailure(RuntimeError):
    """Private acceptance evidence did not match its contract."""


class _NestedAsset(msgspec.Struct):
    b: ImageAsset


class _AmbiguousAssets(msgspec.Struct):
    aliased: ImageAsset = msgspec.field(name="a.b")
    a: _NestedAsset


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ProofFailure(detail)


def _input_slice(
    project: Path,
    config: Any,
    workspace: Path,
    payload: dict[str, Any],
    inputs: dict[str, LocalInput],
    *,
    request_id: str,
    tamper: Path | None = None,
    accepted_placement: str = "",
) -> Any:
    found = discover(project)
    body = descriptor.build(found)
    descriptor_id = descriptor.job_descriptor_id(body, "inspect")
    release_id = development_source_digest(project)
    workspace.mkdir(parents=True, exist_ok=True)
    descriptor_path = workspace / "package-descriptor.json"
    descriptor_path.write_bytes(descriptor.canonical_bytes(body))
    overlay_root = project / "src"

    def plant(owner: Any, worker: Any) -> None:
        del worker
        if tamper is not None:
            mint = owner._grant

            def mutate_after_mint(document: bytes, invocation_digest: bytes) -> Any:
                grant = mint(document, invocation_digest)
                changed = bytearray(tamper.read_bytes())
                changed[-1] ^= 0x5A
                tamper.write_bytes(changed)
                return grant

            owner._grant = mutate_after_mint
        if accepted_placement:
            retain = owner._on_accepted

            def corrupt_acceptance(message: Any) -> None:
                changed = pb.AttemptAccepted()
                changed.CopyFrom(message)
                changed.placement_id = accepted_placement
                retain(changed)

            owner._on_accepted = corrupt_acceptance

    return run_slice(
        config,
        LocalRequest(
            entrypoint="inspect",
            payload=payload,
            outputs=(),
            request_id=request_id,
            kind="job",
            job_descriptor_id=descriptor_id,
            inputs=inputs,
        ),
        [],
        job_records=[
            {
                "job_descriptor_id": descriptor_id,
                "build_id": release_id,
                "application": str(body["application"]),
                "package_descriptor": str(descriptor_path),
                "overlay": str(overlay_root if overlay_root.is_dir() else project),
                "overlay_content_digest": release_id,
                "job": "inspect",
                "gpu_count": 0,
                "publishes": False,
                "emits_media": False,
                "gpu_rate_micro_usd_per_hour": 0,
                "cap_micro_usd": 0,
                "reclaim_on_terminal": True,
            }
        ],
        workspace=workspace,
        package_revision_digest=release_id,
        options=WorkerOptions(root=workspace / "worker", release_id=release_id, no_warm=True),
        record_owner_hook=plant,
    )


def _serving_slice(
    project: Path,
    config: Any,
    workspace: Path,
    payload: dict[str, Any],
    inputs: dict[str, LocalInput],
    *,
    request_id: str,
    accepted_mutation: Any = None,
) -> Any:
    body = descriptor.build(discover(project))
    release_id = str(body["application"])

    def plant(owner: Any, worker: Any) -> None:
        del worker
        retain = owner._on_accepted

        def corrupt_acceptance(message: Any) -> None:
            changed = pb.AttemptAccepted()
            changed.CopyFrom(message)
            accepted_mutation(changed)
            retain(changed)

        owner._on_accepted = corrupt_acceptance

    with contextlib.chdir(project):
        return run_slice(
            config,
            LocalRequest(
                entrypoint="echo",
                payload=payload,
                outputs=(),
                request_id=request_id,
                inputs=inputs,
            ),
            workspace=workspace,
            package_revision_digest=release_id,
            package_descriptor=body,
            options=WorkerOptions(root=workspace / "worker", release_id=release_id, no_warm=True),
            record_owner_hook=plant if accepted_mutation is not None else None,
        )


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    project = repo / "proofs" / "fixtures" / "cli-project"
    with tempfile.TemporaryDirectory(prefix="cozy-local-input-proof-") as temporary:
        root = Path(temporary)
        media = root / "media"
        media.mkdir()
        image = media / "same.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"same-subject-bytes")

        ambiguous = {"a.b": f"@{image}", "a": {"b": f"@{image}"}}
        original = json.loads(json.dumps(ambiguous))
        try:
            payload_grammar.extract_local_assets(ambiguous, _AmbiguousAssets)
            ambiguous_refused = False
        except CliError as exc:
            ambiguous_refused = "field path segments" in str(exc)
        _require(ambiguous_refused and ambiguous == original, "dotted path mutated payload")

        grant_root = root / "grant-root"
        grant_root.mkdir()
        outside = root / "outside.bin"
        outside.write_bytes(b"outside")
        (grant_root / "escape").symlink_to(outside)
        authorizer = grants.Authorizer(roots=(str(grant_root),))
        refused = 0
        for url in (f"file://{grant_root}/../outside.bin", f"file://{grant_root}/escape"):
            try:
                authorizer.authorize("grant_input_read", "GET", url, {"input_id": "hero"})
            except grants.GrantRefusal as exc:
                refused += exc.code == "grant_path_outside_roots"
        _require(refused == 2, "grant traversal or symlink escape admitted")
        inside = grant_root / "inside.bin"
        inside.write_bytes(b"inside")
        authorizer.authorize("grant_input_read", "GET", f"file://{inside}", {"input_id": "hero"})

        config = read_config(
            {
                "PATH": os.environ["PATH"],
                "HOME": os.environ["HOME"],
                "COZY_HOME": str(root / "home"),
            }
        )
        job_payload = {
            "prompt": "fault arm",
            "hero": "input:hero",
            "references": [{"type": "image", "image": "input:references.0.image"}],
        }

        tampered = media / "tampered.png"
        tampered.write_bytes(image.read_bytes())
        outcome = _input_slice(
            project,
            config,
            root / "tamper",
            dict(job_payload),
            {"hero": LocalInput(str(tampered), 0), "references.0.image": LocalInput(str(image), 0)},
            request_id="input-tamper",
            tamper=tampered,
        )
        narrative = json.dumps(dataclasses.asdict(outcome), default=str)
        _require("input_digest_mismatch" in narrative, "post-mint mutation did not refuse")
        leftovers = [
            path
            for path in (root / "tamper" / "worker" / "spool").rglob("inputs/*")
            if path.is_file()
        ]
        _require(not leftovers, f"refused transfer left files: {leftovers}")

        outcome = _input_slice(
            project,
            config,
            root / "order",
            dict(job_payload),
            {"hero": LocalInput(str(image), 0), "references.0.image": LocalInput(str(image), 7)},
            request_id="input-order",
        )
        _require(
            "asset_order" in json.dumps(dataclasses.asdict(outcome), default=str),
            "bad order admitted",
        )

        hostile = "../../outside-authority"
        outcome = _input_slice(
            project,
            config,
            root / "hostile",
            dict(job_payload),
            {"hero": LocalInput(str(image), 0), "references.0.image": LocalInput(str(image), 0)},
            request_id=hostile,
        )
        _require(outcome.ended == "terminal", f"opaque request failed: {outcome.ended}")
        _require(not list(root.rglob("outside-authority*")), "request id became path authority")

        outcome = _input_slice(
            project,
            config,
            root / "job-echo",
            dict(job_payload),
            {"hero": LocalInput(str(image), 0), "references.0.image": LocalInput(str(image), 0)},
            request_id="input-echo",
            accepted_placement="plc-not-offered",
        )
        _require(
            outcome.ended == "record_owner_error" and "serving placement" in outcome.error,
            "changed job acceptance placement did not converge to owner refusal",
        )

        serving_payload = {"text": "acceptance", "count": 1}
        serving = _serving_slice(
            project,
            config,
            root / "serving",
            serving_payload,
            {},
            request_id="serving",
        )
        _require(
            serving.ended == "terminal"
            and serving.accepted.get("request_id") == "serving"
            and "reserved_host_bytes" in serving.accepted,
            f"unmodified serving acceptance failed: {serving.ended}",
        )

        mutations = (
            (
                "request",
                lambda message: setattr(message, "request_id", "wrong"),
                "accepted request",
            ),
            (
                "ordinal",
                lambda message: setattr(message, "attempt_ordinal", message.attempt_ordinal + 1),
                "accepted ordinal",
            ),
            (
                "invocation",
                lambda message: setattr(message, "invocation_spec_digest", b"\0" * 32),
                "accepted invocation digest",
            ),
            (
                "construction",
                lambda message: setattr(message, "model_construction_digest", b"\1" * 32),
                "invents model construction",
            ),
            (
                "plan",
                lambda message: setattr(message.plan, "delivery", "float"),
                "model-bearing plan",
            ),
        )
        for name, mutation, expected in mutations:
            outcome = _serving_slice(
                project,
                config,
                root / f"serving-{name}",
                serving_payload,
                {},
                request_id=f"serving-{name}",
                accepted_mutation=mutation,
            )
            _require(
                outcome.ended == "record_owner_error" and expected in outcome.error,
                f"{name} acceptance mutation did not refuse: {outcome.ended}: {outcome.error}",
            )

    print(
        "private local input acceptance: path/grant fences, mutation/order/request-id plants, "
        "and exact job/serving acceptance receipts passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
