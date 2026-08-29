"""Prepare and verify the installed Runtime CLI acceptance proof."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, cast

from proof_data import ProofFailure, canonical, read_json, require


def prepare(root: Path) -> None:
    media = root / "media"
    media.mkdir(parents=True)
    image = media / "same.png"
    video = media / "motion.mp4"
    audio = media / "voice.wav"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"same-subject-bytes")
    video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"video-reference")
    audio.write_bytes(b"RIFF\x10\x00\x00\x00WAVE" + b"audio-reference")
    prompt = "@/this/is/a/long/prompt/not/a/path"
    references = [
        {"type": "image", "image": f"@{image}"},
        {"type": "video", "video": f"@{video}"},
        {"type": "audio", "audio": f"@{audio}"},
        {"type": "image", "image": f"@{image}"},
    ]
    (root / "request.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "hero": "overridden-by-the-explicit-cli-term",
                "references": references,
            }
        )
    )
    (root / "reordered.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "hero": "overridden-by-the-explicit-cli-term",
                "references": [references[2], references[0], references[1], references[3]],
            }
        )
    )


def tree_digest(root: Path) -> str:
    measured = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        measured.update(path.relative_to(root).as_posix().encode())
        measured.update(b"\0")
        measured.update(path.read_bytes())
        measured.update(b"\0")
    return measured.hexdigest()


def check_wheel(path: Path) -> None:
    offenders: list[str] = []
    attempt_callers: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith("cozy_runtime/cli/") and name.endswith(".py")
        ]
        require(bool(names), "Runtime wheel contains no CLI modules")
        for name in names:
            raw = archive.read(name)
            tree = ast.parse(raw, filename=name)
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "input"
                for node in ast.walk(tree)
            ):
                offenders.append(name)
            if b"run_slice" in raw:
                attempt_callers.append(name)
    require(not offenders, f"installed CLI calls interactive input(): {offenders}")
    require(
        attempt_callers == ["cozy_runtime/cli/run.py"],
        f"CLI attempt-kernel ownership drifted: {attempt_callers}",
    )


def _document(path: Path) -> dict[str, Any]:
    value = read_json(path)
    require(isinstance(value, dict), f"{path.name} is not an object")
    return cast("dict[str, Any]", value)


def verify(root: Path) -> None:
    echo = _document(root / "echo.json")
    require((echo.get("result") or {}).get("text") == "hublesshubless", "offline echo drifted")

    keys = _document(root / "keys.json")
    key_result = keys.get("result") or {}
    require(key_result.get("canonical_refusal") == "topology_extra_keys", "canonical tier")
    require(key_result.get("ignored_extras") == ["core.left_over_weight"], "local tier")
    require(key_result.get("strict_refusal") == "strict_extra_keys", "strict tier")
    require(
        key_result.get("transcribed_markers") == ["core.v_pred->objective=v_prediction"],
        "marker transcription",
    )
    require(key_result.get("marker_conflict") == "marker_conflict", "marker conflict")
    require(key_result.get("clean") is True, "clean artifact confessed a key")

    first = _document(root / "inputs.json")
    invocation = first.get("invocation") or {}
    invocation_document = invocation.get("document") or {}
    rows = {
        row["input_id"]: row
        for row in invocation_document.get("inputs", ())
        if row.get("input_id") != "payload"
    }
    expected = {
        "hero",
        "references.0.image",
        "references.1.video",
        "references.2.audio",
        "references.3.image",
    }
    require(set(rows) == expected, f"typed input ids drifted: {sorted(rows)}")
    require(
        [rows[name].get("order", 0) for name in sorted(rows)] == [0, 0, 1, 2, 3],
        "mixed-media order drifted",
    )
    image_rows = [rows["hero"], rows["references.0.image"], rows["references.3.image"]]
    require(len({row["digest"] for row in image_rows}) == 1, "equal subjects deduplicated")
    require(
        (first.get("result") or {}).get("prompt") == "@/this/is/a/long/prompt/not/a/path",
        "non-asset @/ prompt changed",
    )
    expected_digest = "sha256:" + hashlib.sha256(canonical(invocation_document)).hexdigest()
    require(invocation.get("digest") == expected_digest, "invocation digest is not canonical")
    plan = first.get("plan") or {}
    require(plan.get("invocation_spec_digest") == expected_digest, "accepted invocation drift")
    require(plan.get("request_id") == "cli-inspect", "accepted request id drift")
    require(plan.get("attempt_ordinal") == 1, "accepted ordinal drift")
    require(int(plan.get("executor_generation", 0)) > 0, "executor generation absent")
    receipt = json.dumps({"invocation": invocation, "plan": plan}, sort_keys=True)
    source_paths = [root / "media" / name for name in ("same.png", "motion.mp4", "voice.wav")]
    require(all(str(path) not in receipt for path in source_paths), "source path entered receipt")
    require("credential" not in receipt, "grant credential entered receipt")

    reordered = _document(root / "reordered-result.json")
    reordered_digest = (reordered.get("invocation") or {}).get("digest")
    require(reordered_digest != expected_digest, "reference order did not change invocation")


def main() -> int:
    if len(sys.argv) < 3:
        raise ProofFailure("usage: cli.py <prepare|tree|wheel|verify> <path>")
    command, path = sys.argv[1], Path(sys.argv[2])
    if command == "prepare":
        prepare(path)
    elif command == "tree":
        print(tree_digest(path))
    elif command == "wheel":
        check_wheel(path)
    elif command == "verify":
        verify(path)
    else:
        raise ProofFailure(f"unknown command {command!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
