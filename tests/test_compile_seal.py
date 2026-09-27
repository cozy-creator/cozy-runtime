"""cr-086 arm 3: what `torch.compile` may and may not do to a sealed executor.

Two mechanisms have been conflated in the record as one refusal ("it renamed TensorFS
destinations OR mutated the sealed allocator environment after CUDA init"). They are
different, they fire in different places, and only one of them is still real:

* **Destinations.** `torch.compile(module)` returns an `OptimizedModule` and every
  `state_dict` key gains an `_orig_mod.` prefix, so the derived tensor requirements match
  no checkpoint. That is why construction refuses (`internal/derive.py:_refuse_compile`).
  It is a fact about ONE spelling: `nn.Module.compile` and diffusers' regional
  `compile_repeated_blocks` leave keys, parameters, buffers and tensor identity untouched.
* **The env seal.** PyTorch lazily writes `TORCHINDUCTOR_CACHE_DIR` on its first inductor
  codegen. Before `b6c65d7` an executor with no environment content digest was spawned
  with the JIT cache names ABSENT, so that post-CUDA write moved a sealed name and the
  attempt died `env_seal_broken`. `worker/child.py:_prepare_jit_cache` now imposes all of
  `child_env.JIT_CACHE_ENV` unconditionally, as absolute paths, before Torch is imported —
  which makes torch's own write a no-op. The seal is no longer a reason compilation cannot
  run, and the red arm below is what keeps that true.

`checks/env_seal.py` covered the imposition and a hand-mutated allocator variable, never a
real compile, and was deleted with the rest of the verification machinery (`a96e71d`).
These arms run the real thing.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import child_env, jit_cache


def _torch_here() -> str:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return f"torch must be importable in the test interpreter: {exc}"
    return ""


NO_TORCH = _torch_here()
pytestmark = pytest.mark.skipif(bool(NO_TORCH), reason=NO_TORCH or "")

JIT_NAMES = (*(name for name, _directory in child_env.JIT_CACHE_ENV), "TMPDIR")


@pytest.fixture
def restore_env() -> Iterator[None]:
    """Every arm here moves process environment and the executor's module-level seal."""
    from cozy_runtime.internal import executor

    saved_env = {name: os.environ.get(name) for name in child_env.ALLOWLIST}
    saved_seal = dict(executor._SEALED)
    try:
        yield
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        executor._SEALED.clear()
        executor._SEALED.update(saved_seal)


def _compile_something() -> None:
    """One real inductor compilation on the CPU backend, then the write under test.

    The write is `torch/_inductor/runtime/cache_dir_utils.py:cache_dir`: the CUDA/triton
    path reaches it on every compile (`triton_cache_dir`), but Torch 2.14's CPU path no
    longer does, so it is called here exactly as that path calls it.
    """
    import torch
    import torch._dynamo as dynamo
    from torch import nn
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    dynamo.reset()
    tiny = nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 16)).eval()
    torch.compile(tiny, fullgraph=True)(torch.randn(4, 16))
    cache_dir()


def _seal_under(imposed: dict[str, str]) -> dict[str, object]:
    """Snapshot the seal as the executor does, compile, and read its own verdict back."""
    from cozy_runtime.internal import executor

    for name in child_env.ALLOWLIST:
        os.environ.pop(name, None)
    os.environ.update(imposed)
    executor._SEALED.clear()
    executor._capture_seal()
    assert executor._env_document() == {"env_intact": True, "env_changed": []}
    _compile_something()
    return executor._env_document()


def test_compile_leaves_the_seal_intact_under_the_imposed_jit_scope(
    tmp_path: Path, restore_env: None
) -> None:
    """ARM: the environment a worker actually seals today survives a real compile."""
    scope = jit_cache.scope(
        tmp_path,
        "local-cr-086-arm",
        pod_scope="cr-086-arm",
    )
    assert set(scope.environment) == set(JIT_NAMES)
    assert all(Path(value).is_absolute() for value in scope.environment.values())

    verdict = _seal_under(dict(scope.environment))

    assert verdict == {"env_intact": True, "env_changed": []}


def test_compile_moves_the_seal_when_the_jit_scope_is_not_imposed(restore_env: None) -> None:
    """RED ARM: plant the pre-`b6c65d7` condition and watch the same compile move the seal.

    This is the defect the record remembers as "mutated the sealed allocator environment".
    It was never the allocator: `PYTORCH_CUDA_ALLOC_CONF` does not move here, and naming it
    is what let the whole of compilation be written off. Since #692 a moved name is restored
    and reported, never refused. Delete the imposition in `_prepare_jit_cache` and the ARM
    above reports this name too.
    """
    verdict = _seal_under({})

    assert verdict == {"env_intact": True, "env_changed": ["TORCHINDUCTOR_CACHE_DIR"]}
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ, "the executor restores its seal"


def test_only_the_wrapping_spelling_renames_destinations() -> None:
    """The other half of the "or": which spellings move a checkpoint key, and which do not."""
    import torch
    from torch import nn

    def build() -> nn.Module:
        with torch.device("meta"):
            block = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8))
            return nn.Sequential(block, nn.Linear(8, 8))

    baseline = sorted(build().state_dict())
    assert baseline

    # The spelling construction refuses: every destination gains `_orig_mod.`.
    wrapped: Any = torch.compile(build())
    assert sorted(wrapped.state_dict()) != baseline
    assert all(key.startswith("_orig_mod.") for key in wrapped.state_dict())

    # The in-place spelling, which is what regional compilation uses: nothing moves.
    in_place = build()
    in_place.compile()
    assert sorted(in_place.state_dict()) == baseline

    # Regional: compile the repeated sub-module only. Keys, parameters, buffers and the
    # tensor OBJECTS backing them are the ones fill materialized.
    regional: Any = build()
    before = {name: id(p) for name, p in regional.named_parameters()}
    regional[0].compile()
    assert sorted(regional.state_dict()) == baseline
    assert {name: id(p) for name, p in regional.named_parameters()} == before


def test_construction_refuses_compile_and_names_itself() -> None:
    """The construction fence still holds — for BOTH spellings, and that is deliberate.

    Derive is derive-by-execution, so a lazily compiled callable would trace under fake
    tensors and launch real kernels on fake pointers (pgw#1659). The refusal is scoped to
    construction; nothing patches `torch.compile` on the invocation path, which is where a
    package that wants regional compilation puts it.
    """
    import torch
    from torch import nn

    from cozy_runtime.internal.derive import (
        ConstructionFault,
        Observations,
        serving_substrate,
    )

    with serving_substrate("sm89", Observations()):
        with pytest.raises(ConstructionFault) as wrapping:
            torch.compile(nn.Linear(4, 4))
        with pytest.raises(ConstructionFault) as in_place:
            nn.Linear(4, 4).compile()

    assert wrapping.value.construct == "torch.compile"
    assert in_place.value.construct == "torch.compile"
    # ...and the patch is undone on the way out, so invocation is unfenced.
    torch.compile(lambda x: x)


def test_the_worker_imposes_the_jit_scope_for_every_executor(tmp_path: Path) -> None:
    """The production guarantee the red arm above depends on.

    An UNBOUND executor (no environment installation — a Runtime-only spawn) is the case
    that used to return `{}` and hand torch an absent `TORCHINDUCTOR_CACHE_DIR`. Every
    executor now gets the full set, absolute, so torch's own lazy write is a no-op.
    """
    from cozy_runtime.internal.worker.child import ExecutorSupervision

    supervision = ExecutorSupervision(
        root=tmp_path / "run",
        python="/usr/bin/python3",
        base_env=(),
        cozy_home=tmp_path / "home",
    )
    try:
        assert supervision.environment_installation_id == ""
        imposed = supervision._prepare_jit_cache()
    finally:
        supervision.close()

    assert set(imposed) == set(JIT_NAMES)
    assert all(Path(value).is_absolute() for value in imposed.values())
    # An absolute path is what makes torch's `os.path.abspath` rewrite a no-op; a relative
    # one would be normalized under the executor and move the sealed value.
    assert all(os.path.abspath(value) == value for value in imposed.values())


def test_warm_refuses_only_the_wrapping_spelling() -> None:
    """`Model.warm`'s guard (cr-110): `torch.compile(module)` refuses, naming the site in the
    author's frame; the in-place `module.compile()` runs and moves no key."""
    import torch
    from torch import nn

    from cozy_runtime.internal.derive import ConstructionFault, refuse_compile

    with torch.device("meta"):
        block = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8))
    keys = sorted(block.state_dict())

    with refuse_compile(lazy=False):
        with pytest.raises(ConstructionFault) as refused:
            torch.compile(block)
        block.compile()
    assert refused.value.construct == "torch.compile"
    assert "module.compile()" in refused.value.message
    assert __file__ in refused.value.site
    assert sorted(block.state_dict()) == keys

    # Construction's guard still refuses both, because there the lazy one traces fake tensors.
    with refuse_compile(lazy=True), pytest.raises(ConstructionFault):
        block.compile()
