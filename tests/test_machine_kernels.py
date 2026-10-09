"""The machine's kernel boot through its real entry: real builder processes, a real source
registry and the sealed store an executor reads. The card is the one thing stated: the box
running this has none to compile for.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import (
    accel,
    kernel_cache,
    kernel_compile,
    kernel_sources,
    machine_kernels,
)
from test_kernel_compile import _registry


def _boot(capsys: pytest.CaptureFixture[str]) -> tuple[int, list[dict[str, Any]]]:
    code = machine_kernels.main()
    return code, [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_the_boot_compiles_what_the_cards_admit_into_the_store_its_executors_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _registry(tmp_path / "sources")
    scope = tmp_path / "run"
    scope.mkdir()
    monkeypatch.setattr(kernel_sources, "job", partial(kernel_sources.job, root=root))
    monkeypatch.setattr(accel, "device_capability", lambda entry: {"GPU-a": 120}.get(entry, 0))
    monkeypatch.setattr(kernel_compile, "waiting", None)
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [None])
    monkeypatch.setenv("TMPDIR", str(scope))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setenv(kernel_cache.OWN_ENV, str(tmp_path / "kernels" / "u0"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-unknown")
    monkeypatch.setattr(sys, "path", list(sys.path))  # the boot puts Sol's sources on it

    code, lines = _boot(capsys)
    # Sol's sources are the registry's one artifact here; SageAttention2 has no source in it,
    # FA3 serves no sm120, and Sol's own object needs a DSL this tree does not carry.
    job = kernel_sources.job(kernel_sources.PYTHON, 120)
    built = [row for row in lines if row.get("state") == "ready"]
    assert [(row["kernel"], row["key"]) for row in built] == [
        (kernel_sources.PYTHON, job.key.digest[:16])
    ]
    assert built[0]["compile_ms"] > 0
    last = lines[-1]
    assert code == 0 and last["done"] is True and last["cards"] == [120], last
    assert (last["kernels"], last["failed"]) == (1, 0) and last["driver"] == accel.driver_version()

    # An executor of this generation reads the same store by the same key.
    executor = kernel_cache.from_sealed({kernel_cache.OWN_ENV: str(tmp_path / "kernels" / "u0")})
    assert executor is not None and last["store"] == str(executor.own)
    served = kernel_compile.submit(executor, job)
    assert served.state == "ready" and Path(served.path, kernel_cache.SITE, "tiny_kernel").is_dir()

    # The next machine run finds every key built: it starts no builder.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a")
    started = dict(kernel_compile._STARTED)
    code, lines = _boot(capsys)
    assert code == 0 and started == kernel_compile._STARTED
    assert next(row["state"] for row in lines if "state" in row) == "ready"


def test_a_machine_without_a_card_or_a_store_compiles_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(accel, "device_capability", lambda entry: 0)
    monkeypatch.setenv(kernel_cache.OWN_ENV, str(tmp_path / "u0"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    code, lines = _boot(capsys)
    assert code == 0 and lines == [
        {
            "done": True,
            "cards": [],
            "driver": accel.driver_version(),
            "store": str(tmp_path / "u0"),
            "kernels": 0,
            "failed": 0,
            "ms": lines[0]["ms"],
        }
    ]
    assert not (tmp_path / "u0").exists()


#: Upstream's `setup.py build` as each recipe runs it: it records the jobs it was handed,
#: runs one compiler job as torch would (`PYTORCH_NVCC`), and finishes only once the other
#: wide build runs too, meeting it at a FIFO. A boot that ran them in turn would leave the
#: first one still, and ended as wedged.
_SETUP = """\
import os, subprocess, sys
from pathlib import Path
# The text each recipe rewrites: {anchors!r}
args = sys.argv[1:]
lib, egg = (Path(args[args.index(flag) + 1]) for flag in ("--build-lib", "--egg-base"))
meet = Path({meet!r})
(meet / "{name}").write_text(os.environ["MAX_JOBS"])
subprocess.run([os.environ["PYTORCH_NVCC"], str(meet / "{name}.nvcc")], check=True)
(meet / "{name}.py").write_text(Path(__file__).read_text())
print("[1/2] {name}", flush=True)
os.close(os.open(meet / "fifo", os.O_RDONLY if "{name}" == "sageattention" else os.O_WRONLY))
(lib / "{name}_ext").mkdir(parents=True)
(lib / "{name}_ext" / "__init__.py").write_text("")
(egg / "{name}.egg-info").mkdir()
(egg / "{name}.egg-info" / "PKG-INFO").write_text("Name: {name}\\nVersion: 0\\n")
print("[2/2] {name}", flush=True)
"""


def _tree(root: Path, name: str, files: dict[str, str]) -> dict[str, str]:
    archive = root / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for relative, text in files.items():
            data = text.encode()
            member = tarfile.TarInfo(f"{name}-src/{relative}")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return {
        "version": "0",
        "file": archive.name,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }


def test_the_boot_starts_every_build_at_once_through_the_compile_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On an H100 the request's first kernel (FA3) and H3's (SageAttention2) build side by
    side, every compiler job admitted by the machine's ledger: neither waits for the other."""
    root = _registry(tmp_path / "sources")
    meet = tmp_path / "meet"
    meet.mkdir()
    os.mkfifo(meet / "fifo")
    toolkit = tmp_path / "toolkit"
    toolkit.mkdir()
    (toolkit / "nvcc").write_text('#!/bin/sh\necho compiled > "$1"\n')
    (toolkit / "nvcc").chmod(0o755)
    monkeypatch.setenv("PATH", f"{toolkit}{os.pathsep}{os.environ['PATH']}")

    def files(name: str, setup_py: str) -> dict[str, str]:
        """Every file the recipe rewrites, holding the text it rewrites; `setup_py` runs."""
        recipe = kernel_sources.RECIPES[name]
        texts: dict[str, list[str]] = {}
        for path, old, _ in (*recipe.replace, *recipe.replace_on.get(90, ())):
            texts.setdefault(path, []).append(old)
        tree = {path: "\n".join(anchors) + "\n" for path, anchors in texts.items()}
        return tree | {setup_py: _SETUP.format(anchors=texts[setup_py], meet=str(meet), name=name)}

    index = json.loads((root / "index.json").read_text())
    index["sources"] |= {
        "sageattention": _tree(root, "sageattention", files("sageattention", "setup.py")),
        "flash-attention": _tree(root, "flash-attention", files("flash-attn3", "hopper/setup.py")),
        "flash-attention-cutlass": _tree(root, "cutlass", {"README.md": "cutlass\n"}),
    }
    (root / "index.json").write_text(json.dumps(index))
    kernel_sources.registry.cache_clear()
    monkeypatch.setattr(kernel_sources, "job", partial(kernel_sources.job, root=root))
    monkeypatch.setattr(kernel_sources, "torch_build", lambda: ("2.14.0", "13.0"))
    monkeypatch.setattr(kernel_sources, "compiler", lambda: "toolkit")
    monkeypatch.setattr(accel, "device_capability", lambda entry: {"GPU-h100": 90}.get(entry, 0))
    monkeypatch.setattr(kernel_compile, "waiting", None)
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [None])
    scope = tmp_path / "run"
    scope.mkdir()
    monkeypatch.setenv("TMPDIR", str(scope))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setenv(kernel_cache.OWN_ENV, str(tmp_path / "kernels" / "u0"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-h100")
    monkeypatch.setattr(sys, "path", list(sys.path))

    code, lines = _boot(capsys)
    ready = {row["kernel"] for row in lines if row.get("state") == "ready"}
    assert code == 0 and ready == {kernel_sources.PYTHON, "sageattention", "flash-attn3"}, lines
    for name in ("sageattention", "flash-attn3"):
        recipe = kernel_sources.RECIPES[name]
        rewritten = (meet / f"{name}.py").read_text()
        for path, old, new in (*recipe.replace, *recipe.replace_on.get(90, ())):
            if path.endswith("setup.py"):
                assert new in rewritten and old not in rewritten, (name, old)
        assert int((meet / name).read_text()) == kernel_compile.cpus(), name
        assert (meet / f"{name}.nvcc").read_text() == "compiled\n", name
    book = json.loads((tmp_path / "kernels" / "u0" / ".compile.json").read_text())
    assert book["jobs"] == {} or not kernel_compile.compilers(tmp_path / "kernels" / "u0")
