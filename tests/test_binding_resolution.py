"""Jobs' model slots are DECLARED slots of the release (se-023).

`package.toml` may carry a default for a job's source (`<job>.models.<param>`, cr-073), so
every resolve/bindings surface must enumerate jobs' slots exactly like entrypoints' — the
0.2.2+ strict resolve otherwise refuses the shipped default as `unknown_binding`, which is
what blocked editable installs of sdxl 2.2.0. A job's slot differs in exactly one way: its
source binds per invocation, so an UNBOUND job slot is legal, never `unbound_slot`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_runtime.cli.io import CliError, Options
from cozy_runtime.cli.run import resolved_bindings
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover

TENANT = Path(__file__).resolve().parent.parent / "corpus" / "tenant"

JOB = '''

class PruneRequest(msgspec.Struct, forbid_unknown_fields=True):
    factor: int = 2


class PruneResult(msgspec.Struct):
    kept: int


@app.job
def prune(ctx: Context, payload: PruneRequest, source: TenantModel, tel: Telemetry) -> PruneResult:
    """A model-bound job: `source` is a declared slot whose artifact binds per invocation."""
    return PruneResult(kept=payload.factor)
'''

BINDING = """
[bindings."prune.models.source"]
model = "cozytest/tenant"
release = "1"
lane = "bf16"
"""


def _package(tmp_path: Path, module: str, *, job_binding: bool, extra_toml: str = "") -> Path:
    """`corpus/tenant` plus one model-bound job, under a per-test module name."""
    project = tmp_path / module
    project.mkdir()
    source = (TENANT / "tiny_tenant.py").read_text() + JOB
    (project / f"{module}.py").write_text(source)
    table = (TENANT / "package.toml").read_text().replace("tiny_tenant:app", f"{module}:app")
    if job_binding:
        table += BINDING
    (project / "package.toml").write_text(table + extra_toml)
    return project


def test_a_bound_job_slot_resolves_beside_the_entrypoints(tmp_path: Path) -> None:
    project = _package(tmp_path, "tenant_bound", job_binding=True)
    records, plans = resolved_bindings(Options(dir=str(project)))
    assert plans == []
    assert [(r["model_binding_path"], r["ref"], r["lane"]) for r in records] == [
        ("prune.models.source", "cozytest/tenant@1", "bf16"),
        ("retouch.models.model", "cozytest/tenant@1", "bf16"),
        ("touch.models.model", "cozytest/tenant@1", "bf16"),
    ]


def test_an_unbound_job_slot_is_legal_and_absent(tmp_path: Path) -> None:
    """The source binds per invocation: no package.toml default is not a refusal."""
    project = _package(tmp_path, "tenant_unbound", job_binding=False)
    records, _ = resolved_bindings(Options(dir=str(project)))
    assert [r["model_binding_path"] for r in records] == [
        "retouch.models.model",
        "touch.models.model",
    ]


def test_a_genuinely_unknown_path_still_refuses(tmp_path: Path) -> None:
    """The strict resolve keeps its teeth: an undeclared path is `unknown_binding`."""
    extra = '\n[bindings."missing.models.source"]\nmodel = "cozytest/tenant"\nrelease = "1"\n'
    project = _package(tmp_path, "tenant_unknown", job_binding=True, extra_toml=extra)
    with pytest.raises(CliError) as refused:
        resolved_bindings(Options(dir=str(project)))
    assert "missing.models.source" in str(refused.value)
    assert "declares no slot" in str(refused.value)


def test_the_interface_arm_reads_the_jobs_array(tmp_path: Path) -> None:
    """`bindings --package-interface` recovers jobs' slots from the frozen PackageInterface/1."""
    project = _package(tmp_path, "tenant_interface", job_binding=True)
    interface = tmp_path / "metadata" / "package-interface.json"
    interface.parent.mkdir()
    interface.write_bytes(
        package_interface.canonical_bytes(package_interface.build(discover(project)))
    )
    records, _ = resolved_bindings(Options(dir=str(project), package_interface=str(interface)))
    assert [(r["model_binding_path"], r["ref"]) for r in records] == [
        ("prune.models.source", "cozytest/tenant@1"),
        ("retouch.models.model", "cozytest/tenant@1"),
        ("touch.models.model", "cozytest/tenant@1"),
    ]


def test_source_binding_resolution_never_imports_the_package(tmp_path: Path) -> None:
    project = _package(tmp_path, "tenant_static_bindings", job_binding=True)
    module = project / "tenant_static_bindings.py"
    module.write_text(
        module.read_text() + "\nraise RuntimeError('bindings imported package code')\n"
    )
    records, plans = resolved_bindings(Options(dir=str(project)))
    assert plans == []
    assert [row["model_binding_path"] for row in records] == [
        "prune.models.source",
        "retouch.models.model",
        "touch.models.model",
    ]
