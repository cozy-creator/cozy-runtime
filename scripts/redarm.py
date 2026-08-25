#!/usr/bin/env python3
"""Fence prover: PLANT each violation for real, observe the refusal, remove the plant.

Not a test — it mutates the working tree and restores it. Static arms run `scripts/fence.py`
and expect it red; live arms execute the real runtime function and expect its typed refusal.
Run it whenever a fence changes: `uv run python scripts/redarm.py`.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "cozy_runtime"


@dataclass(frozen=True, slots=True)
class Arm:
    fence: str
    path: pathlib.Path
    plant: str


STATIC_ARMS = (
    Arm("env-authority", SRC / "internal" / "child_env.py", "import os\nS = os.environ.get('X')\n"),
    Arm("package-imports", SRC / "author" / "_app.py", "import cozy_runtime.internal.config\n"),
    Arm("package-imports", SRC / "internal" / "config.py", "from cozy_runtime.cli import io\n"),
    Arm("forbidden-dependency", SRC / "internal" / "config.py", "import varena\n"),
    Arm("forbidden-dependency", ROOT / "pyproject.toml", 'dependencies = ["tensorhub"]'),
    Arm("single-config", SRC / "cli" / "version.py", "def read_config() -> None: ...\n"),
    Arm("single-config", SRC / "cli" / "io.py", "CONFIG = {'smuggled': True}\n"),
    Arm("no-interactive", SRC / "cli" / "new.py", "ANSWER = input('name? ')\n"),
)

LIVE_ARMS = (
    (
        "child-env-allowlist",
        "from cozy_runtime.internal.child_env import project_child_env\n"
        "project_child_env({'CUDA_LAUNCH_BLOCKING': '1'})\n",
        "not in the reviewed device-child env allowlist",
    ),
    (
        "one-config-authority",
        "from cozy_runtime.internal.config import read_config\n"
        "read_config({}); read_config({})\n",
        "exactly one config authority",
    ),
)

# The author surface's red arms (cr-002a). Each body CONSTRUCTS the violation for real and
# runs it; the arm passes only when the runtime refuses with the expected typed code.
PRELUDE = """
import sys
sys.path.insert(0, 'examples')
from enum import Enum
from typing import Annotated
import msgspec
from cozy_runtime.author import *
from cozy_runtime.author.fakes import invoke_with_fakes

class In(msgspec.Struct):
    prompt: str
    num_inference_steps: ModelDefault[int] = 20

class One(msgspec.Struct):
    file: FileAsset

class Two(msgspec.Struct):
    a: FileAsset
    b: FileAsset

class Meta(msgspec.Struct):
    ok: bool

app = App()

@app.entrypoint
def saver(payload: In, out: Outputs) -> One:
    KEEP.append(out)
    return One(file=out.save_bytes(b'x', media_type='text/plain'))

KEEP = []
"""

AUTHOR_ARMS = (
    (
        "no-outputs-cannot-save",
        """
@app.entrypoint
def thief(payload: In) -> Meta:
    KEEP[0].save_bytes(b'smuggled', media_type='text/plain')
    return Meta(ok=True)

invoke_with_fakes(app.get('saver'), {'prompt': 'p'})
assert app.get('thief').surface.capabilities == frozenset(), 'thief declared a capability'
invoke_with_fakes(app.get('thief'), {'prompt': 'p'})
""",
        "escaped_handle",
    ),
    (
        "foreign-output-handle",
        """
FOREIGN = invoke_with_fakes(app.get('saver'), {'prompt': 'p'}, request_id='attempt-1').result

@app.entrypoint
def borrower(payload: In, out: Outputs) -> One:
    return One(file=FOREIGN.file)

invoke_with_fakes(app.get('borrower'), {'prompt': 'p'}, request_id='attempt-2')
""",
        "foreign_handle",
    ),
    (
        "duplicate-output-handle",
        """
@app.entrypoint
def dup(payload: In, out: Outputs) -> Two:
    f = out.save_bytes(b'x', media_type='text/plain')
    return Two(a=f, b=f)

invoke_with_fakes(app.get('dup'), {'prompt': 'p'})
""",
        "duplicate_handle",
    ),
    (
        "unreturned-output-handle",
        """
@app.entrypoint
def leaky(payload: In, out: Outputs) -> Meta:
    out.save_bytes(b'dropped', media_type='text/plain')
    return Meta(ok=True)

invoke_with_fakes(app.get('leaky'), {'prompt': 'p'})
""",
        "unreturned_handle",
    ),
    (
        "result-schema-mismatch",
        """
@app.entrypoint
def wrong(payload: In) -> Meta:
    return In(prompt='not a Meta')

invoke_with_fakes(app.get('wrong'), {'prompt': 'p'})
""",
        "result_schema",
    ),
    (
        "underivable-axis",
        """
class Aspect(Enum):
    A = 'a'
    B = 'b'

class Gapped(msgspec.Struct):
    aspect: Annotated[Aspect, Shape(pixels={Aspect.A: (64, 64)})] = Aspect.A

@app.entrypoint
def gapped(payload: Gapped) -> Meta:
    return Meta(ok=True)

describe(app)
""",
        "underivable_axis",
    ),
    (
        "preflight-touches-bytes",
        """
class RefIn(msgspec.Struct):
    reference: ImageAsset

class Facts(msgspec.Struct, frozen=True):
    size: int

def peeker(payload: RefIn) -> Facts:
    return Facts(size=len(payload.reference.read_bytes()))

@app.entrypoint(preflight=peeker)
def peek(payload: RefIn, facts: Preflight[Facts]) -> Meta:
    return Meta(ok=True)

describe(app)
""",
        "preflight_bytes",
    ),
    (
        "preflight-requests-a-service",
        """
class Facts(msgspec.Struct, frozen=True):
    ok: bool

def greedy(payload: In, out: Outputs) -> Facts:
    return Facts(ok=True)

@app.entrypoint(preflight=greedy)
def greedy_entry(payload: In, facts: Preflight[Facts]) -> Meta:
    return Meta(ok=True)

describe(app)
""",
        "preflight_capability",
    ),
    (
        "preflight-duplicates-a-feature",
        """
class Facts(msgspec.Struct, frozen=True):
    steps: int

def restating(payload: In) -> Facts:
    return Facts(steps=payload.num_inference_steps)

@app.entrypoint(preflight=restating)
def restated(payload: In, facts: Preflight[Facts]) -> Meta:
    return Meta(ok=True)

describe(app)
""",
        "two_producers",
    ),
    (
        "recommended-does-not-exist",
        "Resources(recommended='h100')\n",
        "unexpected keyword argument 'recommended'",
    ),
    (
        "deleted-surface-declared",
        """
class Batched(msgspec.Struct):
    prompt: str
    batch_size: int = 4

@app.entrypoint
def batched(payload: Batched) -> Meta:
    return Meta(ok=True)

describe(app)
""",
        "deleted_surface",
    ),
    (
        "adapter-recipe-conflict",
        """
@app.entrypoint
def planner(payload: In) -> Meta:
    return Meta(ok=True)

invoke_with_fakes(app.get('planner'), {'prompt': 'p'}, recipes=(
    Recipe('adapter:turbo', 'adapter', {'num_inference_steps': 4}),
    Recipe('adapter:lightning', 'adapter', {'num_inference_steps': 8}),
))
""",
        "recipe_conflict",
    ),
    (
        "recipe-names-an-unmarked-field",
        """
@app.entrypoint
def planner(payload: In) -> Meta:
    return Meta(ok=True)

invoke_with_fakes(app.get('planner'), {'prompt': 'p'}, recipes=(
    Recipe('checkpoint:x', 'checkpoint', {'prompt': 'injected'}),
))
""",
        "unknown_default",
    ),
    (
        "request-cannot-forge-hydration",
        """
class RefIn(msgspec.Struct):
    reference: ImageAsset

@app.entrypoint
def uses_ref(payload: RefIn) -> Meta:
    return Meta(ok=True)

invoke_with_fakes(app.get('uses_ref'),
                  {'reference': {'ref': 'x', 'local': '/etc/passwd'}})
""",
        "hydration state is never wire data",
    ),
    (
        "registration-after-freeze",
        """
@app.entrypoint
def first(payload: In) -> Meta:
    return Meta(ok=True)

describe(app)

@app.entrypoint
def second(payload: In) -> Meta:
    return Meta(ok=True)
""",
        "registry is frozen",
    ),
)

# A deliberately mis-typed handler must fail BOTH readers of the one signature with the
# same verdict class: mypy --strict statically, and describe at build (§1.7).
MISTYPED = '''"""Planted: an unannotated handler parameter — the signature IS the contract."""

import msgspec

from cozy_runtime.author import App, Context


class Out(msgspec.Struct):
    ok: bool


app = App()


@app.entrypoint
def broken(ctx: Context, payload) -> Out:
    return Out(ok=True)
'''


def fence() -> tuple[int, str]:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fence.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    return out.returncode, out.stdout + out.stderr


def run_static(arm: Arm) -> str:
    original = arm.path.read_text()
    try:
        if arm.path.name == "pyproject.toml":
            planted = original.replace('dependencies = ["msgspec>=0.19"]', arm.plant, 1)
        else:
            planted = original + "\n" + arm.plant
        arm.path.write_text(planted)
        code, output = fence()
    finally:
        arm.path.write_text(original)
    if code == 0:
        return f"NOT PROVEN — {arm.fence} stayed green with a planted violation"
    lines = output.splitlines()
    line = next((ln.strip() for ln in lines if arm.path.name in ln), lines[0])
    return f"red — {arm.fence} ({arm.path.relative_to(ROOT)}): {line}"


def run_live(name: str, source: str, expect: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, cwd=ROOT, check=False
    )
    if out.returncode == 0 or expect not in out.stderr:
        return f"NOT PROVEN — {name} did not refuse (exit {out.returncode})"
    detail = [ln for ln in out.stderr.splitlines() if expect in ln][-1]
    return f"red — {name}: {detail.strip()}"


def run_mistyped() -> str:
    """One planted signature, two readers: mypy --strict and describe must BOTH refuse."""
    planted = ROOT / "examples" / "_planted_mistyped.py"
    planted.write_text(MISTYPED)
    try:
        checked = subprocess.run(
            ["mypy", "--strict", str(planted.relative_to(ROOT))],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        described = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, 'examples')\n"
                "import _planted_mistyped as m\n"
                "from cozy_runtime.author import describe\n"
                "describe(m.app)\n",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
    finally:
        planted.unlink()
    if checked.returncode == 0:
        return "NOT PROVEN — mypy --strict accepted the mis-typed handler"
    if described.returncode == 0 or "conformance" not in described.stderr:
        return "NOT PROVEN — describe accepted the mis-typed handler"
    static = next(ln for ln in checked.stdout.splitlines() if "error:" in ln)
    build = next(ln for ln in described.stderr.splitlines() if "conformance" in ln)
    return (
        "red — mistyped-dual-verdict:\n"
        f"      mypy:     {static.split('error: ')[-1].strip()}\n"
        f"      describe: {build.split('conformance: ')[-1].strip()[:110]}"
    )


def main() -> int:
    code, _ = fence()
    if code != 0:
        print("refusing to plant: the tree is already fence-red", file=sys.stderr)
        return 1
    results = [run_static(a) for a in STATIC_ARMS]
    results += [run_live(*a) for a in LIVE_ARMS]
    results += [run_live(name, PRELUDE + body, expect) for name, body, expect in AUTHOR_ARMS]
    results.append(run_mistyped())
    for r in results:
        print(r)
    unproven = [r for r in results if r.startswith("NOT PROVEN")]
    code, _ = fence()
    if code != 0:
        print("PLANT NOT REMOVED — tree left fence-red", file=sys.stderr)
        return 1
    print(f"{len(results) - len(unproven)}/{len(results)} red arms observed; tree restored green")
    return 1 if unproven else 0


if __name__ == "__main__":
    raise SystemExit(main())
