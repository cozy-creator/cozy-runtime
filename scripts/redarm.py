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
    Arm(
        "excluded-keywords",
        SRC / "author" / "_context.py",
        "def place(device: str) -> None: ...\n",
    ),
    Arm(
        "no-identifiers-in-endpoint-code",
        ROOT / "examples" / "tiny_h3.py",
        '_PINNED = "minimaxai/h3@v1"\n',
    ),
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

# cr-002b's model-construction arms. The prelude builds a REAL registry over the real
# fixtures, so every violation below is constructed and executed, never simulated.
MODEL_PRELUDE = """
import sys
sys.path.insert(0, 'examples')
from typing import Annotated, Any
import msgspec
import tiny_artifacts as art
import tiny_diffusers as td
import tiny_h3 as h3
import tiny_sdxl as sd
from cozy_runtime.author import *
from cozy_runtime.author._signature import analyze
from cozy_runtime.author.fakes import fake_context, invoke_with_fakes
from cozy_runtime.internal.residency import StagedBackend

class Meta(msgspec.Struct):
    ok: bool

class In(msgspec.Struct):
    prompt: str

def registry():
    return ModelRegistry(release='r', backend=StagedBackend())

def fl2va():
    r = registry()
    return r, r.acquire('generate.models.model', h3.Fl2VAModel, art.h3(task='fl2va'))
"""

MODEL_ARMS = (
    (
        "loader-outside-load",
        """
KEEP = []

class Escaping(h3._H3Base, task='fl2va'):
    def load(self, loader: Loader) -> None:
        KEEP.append(loader)
        self._load(loader, task='fl2va')
    @uses_components('fl2va_transformer')
    def denoise(self, run, cond, *, frames, on_step=None): ...

registry().acquire('g.models.m', Escaping, art.h3(task='fl2va'))
KEEP[0].cache('smuggled')
""",
        "loader_closed",
    ),
    (
        "unregistered-adapter-refuses",
        """
class Unregistered:
    def state_dict(self): return {}

class Naive(Model[Unregistered]):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.load(Unregistered)

registry().acquire('g.models.m', Naive, art.sdxl())
""",
        "unregistered_adapter",
    ),
    (
        "from_pretrained-never-reached",
        """
import pathlib
SENTINEL = pathlib.Path('redarm-from-pretrained-io')

class Reflective:
    def state_dict(self): return {}
    @classmethod
    def from_pretrained(cls, path):
        SENTINEL.write_text('performed I/O')
        return cls()

class Duck(Model[Reflective]):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.load(Reflective)

try:
    registry().acquire('g.models.m', Duck, art.sdxl())
finally:
    assert not SENTINEL.exists(), 'from_pretrained PERFORMED I/O'
""",
        "unregistered_adapter",
    ),
    (
        "use-site-stamp",
        """
def handler(payload: In, model: Annotated[sd.SdxlModel, 'task=txt2img']) -> Meta:
    return Meta(ok=True)

analyze(handler, name='handler', kind='entrypoint')
""",
        "empty_marker_plane",
    ),
    (
        "slot-object-at-a-use-site",
        """
class Slot:
    def __init__(self, cls, **policy): ...

def handler(payload: In, model: sd.SdxlModel = Slot(sd.SdxlModel, repo='cozy/sdxl')) -> Meta:
    return Meta(ok=True)

analyze(handler, name='handler', kind='entrypoint')
""",
        "injected_default",
    ),
    (
        "empty-component-set",
        "uses_components()\n",
        "empty_component_set",
    ),
    (
        "duplicate-component",
        "uses_components('unet', 'unet')\n",
        "duplicate_component",
    ),
    (
        "uses-components-on-lifecycle",
        "uses_components('unet')(lambda self, loader: None).__name__\n"
        "class Late(Model[object]):\n"
        "    @uses_components('unet')\n"
        "    def load(self, loader: Loader) -> None: ...\n",
        "component_placement",
    ),
    (
        "concurrent-public-scope",
        """
class Nested(h3._H3Base, task='fl2va'):
    def load(self, loader: Loader) -> None:
        self._load(loader, task='fl2va')
    @uses_components('fl2va_transformer')
    def denoise(self, run, cond, *, frames, on_step=None): ...
    @uses_components('visual_vae')
    def composite(self):
        return self.decode_video(h3.H3Latents(1, 2))

registry().acquire('g.models.m', Nested, art.h3(task='fl2va')).composite()
""",
        "concurrent_scope",
    ),
    (
        "undeclared-component-poisons",
        """
r, model = fl2va()
model.pipe.blocks.encode_audio(None, sample_rate=48000)
""",
        "undeclared_component",
    ),
    (
        "lazy-object-escapes-the-lease",
        """
class Lazy(h3._H3Base, task='fl2va'):
    def load(self, loader: Loader) -> None:
        self._load(loader, task='fl2va')
    @uses_components('fl2va_transformer')
    def denoise(self, run, cond, *, frames, on_step=None):
        return (self.pipe.blocks.step('fl2va_transformer', i, None, 0.0, 0.0, 0)
                for i in range(frames))

registry().acquire('g.models.m', Lazy, art.h3(task='fl2va')).denoise(None, None, frames=2)
""",
        "lazy_escape",
    ),
    (
        "dual-artifact-through-one-class",
        """
class BothTransformers(h3._H3Base, task='fl2va'):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(td.MiniMaxH3ModularPipeline, factory=td.build_h3_both)
        self.modulation = h3.LiveAdaLN(loader.cache('m'), self.pipe)
    @uses_components('fl2va_transformer')
    def denoise(self, run, cond, *, frames, on_step=None): ...

registry().acquire('g.models.m', BothTransformers, art.h3(task='fl2va', dual=True))
""",
        "undeclared_construction",
    ),
    (
        "narrow-artifact-missing-keys",
        """
class BothTransformers(h3._H3Base, task='fl2va'):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(td.MiniMaxH3ModularPipeline, factory=td.build_h3_both)
        self.modulation = h3.LiveAdaLN(loader.cache('m'), self.pipe)
    @uses_components('fl2va_transformer', 'ref2va_transformer')
    def denoise(self, run, cond, *, frames, on_step=None): ...

registry().acquire('g.models.m', BothTransformers, art.h3(task='fl2va'))
""",
        "topology_missing_keys",
    ),
    (
        "unknown-component-name",
        """
class Typo(h3._H3Base, task='fl2va'):
    def load(self, loader: Loader) -> None:
        self._load(loader, task='fl2va')
    @uses_components('fl2va_transfomer')
    def denoise(self, run, cond, *, frames, on_step=None): ...

registry().acquire('g.models.m', Typo, art.h3(task='fl2va'))
""",
        "unknown_component",
    ),
    (
        "canonical-lane-surplus-keys",
        """
extra = dict(art.h3(task='fl2va').topology)
extra['fl2va_transformer.blocks.0.attn.k.weight'] = TensorSpec((32, 32))
registry().acquire('g.models.m', h3.Fl2VAModel, Artifact(
    snapshot='cozy/h3-fl2va@v1/full#drift',
    topology=extra,
    config=art.h3(task='fl2va').config,
    stamps={'task': 'fl2va'}))
""",
        "topology_extra_keys",
    ),
    (
        "stamp-fails-closed",
        "registry().acquire('g.models.m', h3.Ref2VAModel, art.h3(task='fl2va'))\n",
        "stamp_mismatch",
    ),
    (
        "persistent-allocation",
        """
r, model = fl2va()
model.warm_table = tuple(range(1024))
""",
        "persistent_allocation",
    ),
    (
        "request-state-escape",
        """
r, model = fl2va()
model.pipe = model.for_request(fake_context(), seed=1)
""",
        "request_state_escape",
    ),
    (
        "cache-outside-a-scope",
        """
r, model = fl2va()
plan = h3.prepare(h3.GenerateInput(prompt='p'), frames=39)
view = model.for_request(fake_context(), seed=1)
run = h3.start_run(view, plan, task='fl2va', steps=4, checkpoint=model.checkpoint_ref)
model.modulation.for_run(run)
""",
        "no_active_scope",
    ),
    (
        "cache-over-its-declared-bound",
        """
class Unbounded(h3._H3Base, task='fl2va'):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(td.MiniMaxH3ModularPipeline,
                                     factory=lambda c: td.build_h3_pipeline(c, task='fl2va'))
        self.modulation = h3.LiveAdaLN(loader.cache('m', max_entries=1), self.pipe)
    @uses_components('fl2va_transformer')
    def denoise(self, run, cond, *, frames, on_step=None):
        return self._denoise(run, cond, 'fl2va_transformer', frames=frames, on_step=on_step)

model = registry().acquire('g.models.m', Unbounded, art.h3(task='fl2va'))
plan = h3.prepare(h3.GenerateInput(prompt='p'), frames=39)
view = model.for_request(fake_context(), seed=1)
cond = h3.pack(plan, (1, 1), (), ())
for steps in (2, 4, 8):
    run = h3.start_run(view, plan, task='fl2va', steps=steps, checkpoint=model.checkpoint_ref)
    model.denoise(run, cond, frames=39)
""",
        "cache_bound",
    ),
    (
        "mutable-cache-entry",
        """
class Mutable(h3._H3Base, task='fl2va'):
    def load(self, loader: Loader) -> None:
        self._load(loader, task='fl2va')
        self.leaky = loader.cache('leaky')
    @uses_components('fl2va_transformer')
    def denoise(self, run, cond, *, frames, on_step=None):
        return self.leaky.get('k', build=lambda: [1, 2, 3])

model = registry().acquire('g.models.m', Mutable, art.h3(task='fl2va'))
model.denoise(None, None, frames=1)
""",
        "cache_mutable",
    ),
    (
        "request-view-subclass",
        "class Wider(RequestView):\n    pipe: object\n",
        "view_subclass",
    ),
    (
        "for_test-names-an-artifact",
        "h3.Fl2VAModel.for_test(checkpoint_ref='minimaxai/h3@v1')\n",
        "identifier_in_code",
    ),
    (
        "config-carries-a-path",
        "Config({'unet_path': '/models/h3/unet.safetensors'})\n",
        "config_path",
    ),
    (
        "engine-tier-is-not-a-stamp",
        "class Hosted(Model[object], self_loading='engine owns the whole device'): ...\n",
        "reserved_keyword",
    ),
    (
        "stamp-restated-at-a-second-class",
        "class Twin(h3.Fl2VAModel, task='ref2va'): ...\n",
        "restamped",
    ),
    (
        "model-adapters-does-not-exist",
        "class WithStack(Model[object]):\n    adapters: tuple[object, ...] = ()\n",
        "deleted_model_member",
    ),
    (
        "excluded-keyword-on-a-model-method",
        """
class Placing(h3._H3Base, task='fl2va'):
    def load(self, loader: Loader) -> None:
        self._load(loader, task='fl2va')
    @uses_components('fl2va_transformer')
    def denoise(self, run, cond, *, frames, device='cuda:0', on_step=None): ...
""",
        "excluded_keyword",
    ),
    (
        "model-without-load",
        """
class Hollow(Model[object]):
    @uses_components('unet')
    def go(self) -> None: ...

def handler(payload: In, model: Hollow) -> Meta:
    return Meta(ok=True)

app = App()
app.entrypoint(handler)
describe(app)
""",
        "no_load",
    ),
    (
        "model-with-constructor-arguments",
        """
class Preloaded(Model[object]):
    def __init__(self, ref: str) -> None:
        self.ref = ref
    def load(self, loader: Loader) -> None: ...
    @uses_components('unet')
    def go(self) -> None: ...

def handler(payload: In, model: Preloaded) -> Meta:
    return Meta(ok=True)

app = App()
app.entrypoint(handler)
describe(app)
""",
        "model_init",
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


# The Loader/Context separation, planted BOTH WAYS in one file: a `load()` that takes the
# request Context, and request code that holds a Loader. Both are type errors, not
# footnotes — mypy refuses the first, and describe refuses the second (§1.3).
SEPARATION = '''"""Planted: the load-time/request-time separation, in both directions."""

import msgspec

from cozy_runtime.author import App, Context, Loader, Model, Outputs, uses_components


class Out(msgspec.Struct):
    ok: bool


class Pipe:
    def state_dict(self) -> dict[str, object]:
        return {}


class Leaky(Model[Pipe]):
    def load(self, ctx: Context) -> None:
        """(1) load-time code reaching for the REQUEST Context."""

    @uses_components("unet")
    def go(self, out: Outputs) -> None:
        """A request service on a model method."""


app = App()


@app.entrypoint
def handler(ctx: Context, payload: Out, model: Leaky, loader: Loader) -> Out:
    """(2) request code holding a Loader."""
    model.load(loader)
    return Out(ok=True)
'''


def run_separation() -> str:
    """Type-level Loader/Context separation, proven red in BOTH directions."""
    planted = ROOT / "examples" / "_planted_separation.py"
    planted.write_text(SEPARATION)
    try:
        checked = subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", str(planted.relative_to(ROOT))],
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
                "import _planted_separation as m\n"
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
    errors = [ln for ln in checked.stdout.splitlines() if "error:" in ln]
    if not errors:
        return "NOT PROVEN — mypy --strict accepted load(ctx: Context)"
    if described.returncode == 0 or "conformance" not in described.stderr:
        return "NOT PROVEN — describe accepted a Loader in a handler signature"
    build = next(ln for ln in described.stderr.splitlines() if "conformance" in ln)
    return (
        "red — loader-context-separation (both ways):\n"
        f"      mypy:     {errors[0].split('error: ')[-1].strip()[:98]}\n"
        f"      describe: {build.split('conformance: ')[-1].strip()[:98]}"
    )


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
            [sys.executable, "-m", "mypy", "--strict", str(planted.relative_to(ROOT))],
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
    results += [run_live(name, MODEL_PRELUDE + body, expect) for name, body, expect in MODEL_ARMS]
    results.append(run_mistyped())
    results.append(run_separation())
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
