"""`Model.warm` on the REAL worker (cr-110, model-lifecycle.md #708).

Every arm drives the real trampoline, real executor processes and real weights through the
real fill plane against `corpus/tenant` — the same harness as `test_shared_lane` — with ONE
addition: the tenant's model declares a `warm`. What the arms prove:

* `warm` runs once per construction FILL, inside the prepare, before the placement is
  DISPATCHABLE, under a Context with a device, no bound and no request; a dry step through
  a declared component-use scope works there; its wall time is the `warm` leg of the
  prepare and rides the fill's boot record. Two placements, two fills, two warms.
* Residency moving bytes between requests does not re-run it: the swap arm vacates and
  restores each tenant four times and the warm count stays at one per fill.
* A raising `warm` is a typed construction failure (`warm_failed`, naming the exception);
  the placement never becomes DISPATCHABLE.
* `torch.compile(module)` inside `warm` is refused; `module.compile()` is allowed.

Skips name the host fact that stopped them, exactly as `test_shared_lane` does.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cozy_runtime.internal.worker.plan import UNCALIBRATED_ACTIVATION_BYTES
from test_shared_lane import (
    NO_TENANT,
    Neighbour,
    TenantStore,
    _drive,
    _free,
    _turn,
    _workspace,
    measure_context,
    write_tenant_store,
)

needs_tenant = pytest.mark.skipif(bool(NO_TENANT), reason=NO_TENANT or "")

ROOT = Path(__file__).resolve().parent.parent
TENANT = ROOT / "corpus" / "tenant"

#: The recorder every variant shares: one JSON row per `warm` call, next to the module.
_RECORDER = """

import json as _json
from pathlib import Path as _Path

WARM_LOG = _Path(__file__).with_name("warm.jsonl")


def _record(ctx: Context, **facts: object) -> None:
    row = {
        "request_id": ctx.request_id,
        "device": str(ctx.device),
        "cancelled": ctx.cancelled,
        "unbounded": ctx.deadline == float("inf"),
        **facts,
    }
    with WARM_LOG.open("a") as log:
        log.write(_json.dumps(row) + "\\n")
"""

#: What each variant's `warm` does. `dry` takes one step through the declared scope.
_VARIANTS = {
    "dry": "        _record(ctx, checksum=self.touch(0))\n",
    "raising": '        raise RuntimeError("triton says no")\n',
    "wrapping": ('        import torch\n\n        torch.compile(self.pipe.components["unet"])\n'),
    "inplace": (
        '        self.pipe.components["unet"].compile()\n'
        "        _record(ctx, checksum=self.touch(0))\n"
    ),
}

_ANCHOR = (
    '    @uses_components("text_encoder", "unet", "vae")\n'
    "    def touch(self, seed: int) -> float:\n"
)


def _package(root: Path, variant: str) -> Path:
    """`corpus/tenant` with a `warm` of the named variant, under its own module name."""
    module = f"tenant_warm_{variant}"
    project = root / module
    project.mkdir(parents=True)
    source = (TENANT / "tiny_tenant.py").read_text()
    assert source.count(_ANCHOR) == 1
    method = "    def warm(self, ctx: Context) -> None:\n" + _VARIANTS[variant] + "\n"
    (project / f"{module}.py").write_text(source.replace(_ANCHOR, method + _ANCHOR) + _RECORDER)
    table = (TENANT / "package.toml").read_text().replace("tiny_tenant:app", f"{module}:app")
    (project / "package.toml").write_text(table)
    return project


def _warms(project: Path) -> list[dict[str, object]]:
    log = project / "warm.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def _legs(notes: list[tuple[str, str]]) -> list[tuple[str, dict[str, float]]]:
    """(placement, {leg: ms}) per `prepare legs` note."""
    found = []
    for kind, step in notes:
        match = re.match(r"'([^']+)' prepare legs: (.*)", step)
        if kind == "boot" and match:
            legs = dict(re.findall(r"(\w+) (\d+)ms", match.group(2)))
            found.append((match.group(1), {k: float(v) for k, v in legs.items()}))
    return found


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> TenantStore:
    if NO_TENANT:
        pytest.skip(NO_TENANT)
    return write_tenant_store(tmp_path_factory.mktemp("tenant") / "small", blocks=2, scale=4)


def _assert_warm_row(row: dict[str, object]) -> None:
    assert row["request_id"] == "" and row["device"] == "cuda:0", row
    assert row["cancelled"] is False and row["unbounded"] is True, row
    assert isinstance(row["checksum"], float) and -1.0 <= row["checksum"] <= 1.0, row


# --------------------------------------------------------------------------- the green arms


@needs_tenant
def test_warm_runs_once_per_fill_before_the_placement_is_dispatchable(
    small: TenantStore, tmp_path: Path
) -> None:
    """Two placements of one construction: two fills, two warms, each before its
    placement's DISPATCHABLE edge and each recorded as the prepare's `warm` leg."""
    project = _package(tmp_path, "dry")
    turns = [_turn("plc-p", "p-1", 1), _turn("plc-q", "q-1", 2), _turn("plc-p", "p-2", 3)]
    with _workspace() as root:
        owner, worker, notes = _drive(small, turns, root, package=project)
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    assert [turn.outcome["status"] for turn in owner.turns] == [1, 1, 1]

    rows = _warms(project)
    assert len(rows) == 2, rows
    for row in rows:
        _assert_warm_row(row)

    # The leg rides the fill's boot record, once per placement. (The operator's `prepare
    # legs` note elides its cheapest legs under a cap, cr-141, so a few-ms warm is read
    # here and not there.)
    assert set(dict(_legs(notes))) == {"plc-p", "plc-q"}, notes
    filled = [
        row for row in worker.boot.steps if row["step"] == "load" and row["outcome"] == "filled"
    ]
    assert len(filled) == 2 and all(row["warm_ms"] > 0 for row in filled), filled

    # BEFORE DISPATCHABLE: each placement's `prepare legs` note (the warm leg is on it)
    # precedes its own dispatchable edge — the primary's is the serving transition, a
    # hosted placement's names it.
    steps = [step for _kind, step in notes]
    edges = {"plc-p": "SERVING_STATE_DISPATCHABLE", "plc-q": "'plc-q' dispatchable"}
    for placement, edge in edges.items():
        warmed = next(i for i, s in enumerate(steps) if s.startswith(f"'{placement}' prepare legs"))
        served = next(i for i, s in enumerate(steps) if edge in s)
        assert warmed < served, (placement, steps[warmed], steps[served])


@needs_tenant
def test_warm_does_not_rerun_when_residency_moves_the_bytes(tmp_path: Path) -> None:
    """THE CADENCE. `test_shared_lane`'s swap arm with a warming tenant: beside a neighbour
    that leaves the card room for ONE tenant, each attempt vacates the other and restores
    its own — four moves of every weight byte through the same module objects — and the
    warm count stays at exactly one per fill."""
    store = write_tenant_store(tmp_path / "large", blocks=48)
    weights = store.weight_bytes
    context = measure_context()
    lower = weights + 2 * context + UNCALIBRATED_ACTIVATION_BYTES
    upper = 2 * weights + context
    if lower >= upper:
        pytest.skip(f"the fixture ({weights} B) is not larger than a context plus the bound")
    target = (lower + upper) // 2
    if _free() <= target:
        pytest.skip(f"the card has {_free()} B free, under the {target} B the arm needs")
    project = _package(tmp_path, "dry")
    neighbour = Neighbour(target)
    try:
        turns = [_turn("plc-p", "p-1", 1), _turn("plc-q", "q-1", 2), _turn("plc-p", "p-2", 3)]
        with _workspace() as root:
            owner, _worker, notes = _drive(store, turns, root, package=project)
    finally:
        neighbour.close()
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    assert [turn.outcome["status"] for turn in owner.turns] == [1, 1, 1]
    vacated = [step for kind, step in notes if kind == "residency" and " vacated " in step]
    restored = [step for kind, step in notes if kind == "residency" and " restored " in step]
    assert len(vacated) == 4 and len(restored) >= 2, (vacated, restored)
    rows = _warms(project)
    assert len(rows) == 2, rows
    for row in rows:
        _assert_warm_row(row)


@needs_tenant
def test_the_in_place_compile_spelling_is_allowed(small: TenantStore, tmp_path: Path) -> None:
    project = _package(tmp_path, "inplace")
    with _workspace() as root:
        owner, _worker, _notes = _drive(small, [_turn("plc-p", "p-1", 1)], root, package=project)
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    assert owner.turns[0].outcome["status"] == 1
    rows = _warms(project)
    assert len(rows) == 1, rows
    _assert_warm_row(rows[0])


# --------------------------------------------------------------------------- the red arms


def _refused(project: Path, small: TenantStore) -> tuple[str, list[tuple[str, str]]]:
    with _workspace() as root:
        owner, worker, notes = _drive(small, [_turn("plc-p", "p-1", 1)], root, package=project)
    assert owner.ended == "worker_error", (owner.ended, owner.error)
    assert owner.turns == [] and not owner.dispatched
    assert not any("dispatchable" in step.lower() for _kind, step in notes), notes
    assert worker.latched.startswith("prepare refused"), worker.latched
    return owner.error, notes


@needs_tenant
def test_a_raising_warm_is_a_construction_failure(small: TenantStore, tmp_path: Path) -> None:
    project = _package(tmp_path, "raising")
    error, notes = _refused(project, small)
    assert "warm_failed" in error and "RuntimeError: triton says no" in error, error
    assert any("warm_failed" in step and "triton says no" in step for _k, step in notes), notes
    assert _warms(project) == []


@needs_tenant
def test_the_wrapping_compile_spelling_is_refused_inside_warm(
    small: TenantStore, tmp_path: Path
) -> None:
    project = _package(tmp_path, "wrapping")
    error, _notes = _refused(project, small)
    assert "warm_failed" in error and "torch.compile" in error, error
    assert "module.compile()" in error, error
