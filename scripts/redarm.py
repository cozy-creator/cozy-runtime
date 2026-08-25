#!/usr/bin/env python3
"""Fence prover: PLANT each violation for real, observe the refusal, remove the plant.

Not a test — it mutates the working tree and restores it. Static arms run `scripts/fence.py`
and expect it red; live arms execute the real runtime function and expect its typed refusal.
Run it whenever a fence changes: `uv run python scripts/redarm.py`.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
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
    Arm("forbidden-dependency", ROOT / "pyproject.toml", 'dependencies = ["tensorhub", '),
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
    # §8's one-kernel rule, planted as the exact thing it forbids: a SECOND direct-call
    # path into the author invocation kernel, and a SECOND attempt engine beside the
    # supervisor's. Both are the v1 failure mode where the developer door and the
    # production door ran different lifecycle code.
    Arm(
        "one-attempt-kernel",
        SRC / "cli" / "version.py",
        "from cozy_runtime.author import attempt\n\n\n"
        "def run_direct(registration: object, payload: dict[str, object]) -> None:\n"
        "    attempt(registration, payload, None)\n",
    ),
    # cr-011's §1.3 rule, planted as the two ways it could break: the observation plane
    # IMPORTING a deciding module, and naming a residency operation without importing one.
    # cr-008b's done-when as a plant: an endpoint that can move bytes is a second policy.
    Arm(
        "endpoint-owns-no-residency",
        ROOT / "corpus" / "pipeline" / "sdxl_txt2img.py",
        "def _reclaim() -> None:\n    import torch\n\n    torch.cuda.empty_cache()\n",
    ),
    Arm(
        "endpoint-owns-no-residency",
        ROOT / "corpus" / "sdxl_pipeline.py",
        "def _offload(pipe: object) -> None:\n    pipe.enable_model_cpu_offload()\n",
    ),
    Arm(
        "timing-decides-nothing",
        SRC / "author" / "_observations.py",
        "from cozy_runtime.internal.fill import ResourceLedger\n",
    ),
    Arm(
        "timing-decides-nothing",
        SRC / "internal" / "worker" / "observe.py",
        "def _shrink(model: object) -> None:\n    model._cozy_active = None\n",
    ),
    Arm(
        "one-attempt-kernel",
        SRC / "internal" / "local.py",
        "from cozy_runtime.internal.worker.attempts import AttemptEngine as _E\n\n\n"
        "def second_engine() -> object:\n"
        "    return AttemptEngine()\n",
    ),
)

#: A committed bundle, edited on disk afterwards, must be a DETECTED corruption rather than
#: a plausible story — the journaled write receipt is what makes that possible.
TAMPER = """
import json, pathlib, shutil, tempfile
from cozy_runtime.internal.journal import Journal
from cozy_runtime.internal.worker import triage

root = pathlib.Path(tempfile.mkdtemp(prefix="cozy-triage-arm"))
journal = Journal(root / "journal.log")
store = triage.BundleStore(root / "triage", journal)
receipt = store.commit(triage.Assembly(triage.mint_subject(), attempt={"request_id": "r"}))
path = root / "triage" / (receipt.subject_id + ".json")
document = json.loads(path.read_text())
document["terminal"] = {"status": "TERMINAL_STATUS_SUCCEEDED"}
path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
try:
    triage.BundleStore.read(root / "triage", receipt.subject_id, root / "journal.log")
finally:
    shutil.rmtree(root, ignore_errors=True)
"""

#: A log line cannot declare terminal success. The bundle has no field that can carry a
#: terminal claim, and the ONE terminal fact it holds is a COPY of the journaled terminal.
LOG_AUTHORITY = """
import pathlib, shutil, tempfile
from cozy_runtime.internal.journal import Journal
from cozy_runtime.internal.worker import triage

root = pathlib.Path(tempfile.mkdtemp(prefix="cozy-authority-arm"))
journal = Journal(root / "journal.log")
store = triage.BundleStore(root / "triage", journal)
assembly = triage.Assembly(triage.mint_subject(), terminal={"status": "TERMINAL_STATUS_FAILED"})
assembly.events.append({"kind": "log", "name": "SUCCEEDED", "value": "the handler is done"})
receipt = store.commit(assembly)
document = triage.BundleStore.read(root / "triage", receipt.subject_id, root / "journal.log")
shutil.rmtree(root, ignore_errors=True)
claims = [row for row in document["events"] if row.get("kind") not in ("log", "metric",
          "stage", "confession", "boot", "liveness", "fault")]
assert not claims, claims
assert document["terminal"]["status"] == "TERMINAL_STATUS_FAILED", document["terminal"]
raise SystemExit(
    "observation only: the bundle's " + str(len(document["events"])) +
    " event row(s) are non-authoritative and the terminal it carries still reads "
    + document["terminal"]["status"]
)
"""

#: The ladder's own prelude: a real `PlanChooser` over a real `Ledger`, no device needed.
#: Every number below is one the ledger would have MEASURED; the arms move them and observe
#: which rung the walk picks and what it refuses.
LADDER = """
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import BindingPlan, PlanChooser

FEATURES = {"entrypoint": "generate", "scalars": {"steps": 20}, "fields": [], "payload_bytes": 64}
ledger = Ledger(host_budget=8 << 30, pinned_budget=512 << 20)
ledger.resident = {"unet": 5_134_927_368, "vae": 167_307_726}
ledger.device_free = 878_733_824
ledger.device_total = 8_186_822_656
ledger.device_allocated = 5_302_235_094
ledger.device_reserved = 5_400_000_000
chooser = PlanChooser(ledger)


def binding(**over):
    row = dict(
        plan_id="p", project="x", model_class="M", binding_path="b", param="m",
        component="unet", store="s", config="c", snapshot="sha256:" + "0" * 64,
        release="r", variant="sm89", vram_bytes=7 << 30, host_bytes=1 << 30,
        pinned_bytes=256 << 20, entrypoint="generate",
    )
    row.update(over)
    return BindingPlan(**row)
"""

LIVE_ARMS = (
    (
        "child-env-allowlist",
        "from cozy_runtime.internal.child_env import project_child_env\n"
        "project_child_env({'CUDA_LAUNCH_BLOCKING': '1'})\n",
        "not in the reviewed device-child env allowlist",
    ),
    (
        "one-config-authority",
        "from cozy_runtime.internal.config import read_config\nread_config({}); read_config({})\n",
        "exactly one config authority",
    ),
    # ------------------------------------------------------------------ cr-011
    (
        "observation-credential-field",
        "from cozy_runtime.author._observations import EventRing\n"
        "EventRing().emit('log', 'fetching', 'info', auth_token='ya29.a0Af')\n",
        "credential_field",
    ),
    (
        "observation-credential-value",
        "from cozy_runtime.author._observations import EventRing\n"
        "EventRing().emit('log', 'signed the url', "
        "'https://s3/x?X-Amz-Signature=deadbeefcafe')\n",
        "credential_value",
    ),
    (
        "observation-malformed-kind",
        "from cozy_runtime.author._observations import EventRing\n"
        "EventRing().emit('progress', 'step')\n",
        "not one of the closed observation kinds",
    ),
    (
        "observation-oversize",
        "from cozy_runtime.author._observations import EventRing\n"
        "EventRing().emit('log', 'x' * 4096)\n",
        "observation_too_large",
    ),
    (
        "triage-bundle-tamper",
        TAMPER,
        "bundle_corrupt",
    ),
    (
        "triage-log-is-not-terminal",
        LOG_AUTHORITY,
        "observation only",
    ),
    # ------------------------------------------------------------------ cr-008b
    (
        "coefficient-unsourced-measured",
        "from cozy_runtime.internal.worker.plan import Coefficient\n"
        "Coefficient('activations', 1 << 30, 'measured')\n",
        "coefficient_unsourced",
    ),
    (
        "uncalibrated-cannot-refuse",
        LADDER + "plan = chooser.choose(binding(tenancy='local'), FEATURES)\n"
        "assert plan.placement == 'component_staged', plan.placement\n"
        "raise SystemExit('an uncalibrated coefficient DEEPENED to ' + plan.placement +\n"
        "    ' and refused nothing: ' + plan.quantified_choice[-140:])\n",
        "an uncalibrated coefficient DEEPENED",
    ),
    (
        "shared-tenancy-reschedules",
        LADDER + "from cozy_runtime.internal.worker.plan import PlanRefusal\n"
        "ledger.activations_peak = 8 << 30\n"
        "try:\n"
        "    chooser.choose(binding(tenancy='shared'), FEATURES)\n"
        "except PlanRefusal as exc:\n"
        "    assert exc.code == 'capacity_shortfall', exc.code\n"
        "    assert exc.shortfall['scope'] == 'worker', exc.shortfall\n"
        "    raise SystemExit(exc.detail[:200])\n",
        "not a structural incompatibility",
    ),
    (
        "local-floor-exits-quantified",
        LADDER + "from cozy_runtime.internal.worker.plan import PlanRefusal\n"
        "ledger.activations_peak = 8 << 30\n"
        "try:\n"
        "    chooser.choose(binding(tenancy='local', vram_floor_bytes=64 << 30), FEATURES)\n"
        "except PlanRefusal as exc:\n"
        "    assert exc.code == 'capacity_floor', exc.code\n"
        "    raise SystemExit(exc.detail[:220])\n",
        "below the PHYSICAL FLOOR",
    ),
    (
        "local-capacity-never-refuses-above-the-floor",
        LADDER + "ledger.activations_peak = 8 << 30\n"
        "plan = chooser.choose(binding(tenancy='local', vram_floor_bytes=1 << 20), FEATURES)\n"
        "raise SystemExit('DEGRADED rather than refused: ' + plan.quantified_choice[-150:])\n",
        "DEGRADED",
    ),
    # The two arms cl-003's second-1024px defect earned. Both are pure arithmetic over a
    # ledger whose numbers are the ones this box actually measured on the four-component
    # pipeline, so the fence is provable without a card.
    (
        "chooser-refuses-an-unreconciled-ledger",
        LADDER + "from cozy_runtime.internal.worker.plan import PlanRefusal\n"
        "ledger.unreconciled = 'device_allocated: +67108864 B unaccounted across the attempt'\n"
        "try:\n"
        "    chooser.choose(binding(tenancy='local'), FEATURES)\n"
        "except PlanRefusal as exc:\n"
        "    assert exc.code == 'ledger_unreconciled', exc.code\n"
        "    raise SystemExit(exc.detail[:220])\n",
        "no rung may be priced against it",
    ),
    (
        "absent-components-are-priced-back",
        # THE EXACT SHAPE cl-003 CAUGHT: a staged attempt left the generation holding the VAE
        # alone, so a ladder that reads only the resident map sees 167 MB of weights, nearly
        # the whole card allocatable, and calls `all_resident` a fit — with the 6.3 GiB it
        # would have to re-stage charged to nobody. Measured numbers, from the real run.
        LADDER + "ledger.resident = {'vae': 167_307_726}\n"
        "ledger.evicted = {'unet': 5_134_927_368, 'text_encoder_2': 1_389_319_680,\n"
        "                  'text_encoder': 246_120_960}\n"
        "ledger.device_free = 7_424_000_000\n"
        "ledger.device_allocated = 175_839_744\n"
        "ledger.device_reserved = 200_000_000\n"
        "plan = chooser.choose(binding(tenancy='local'), FEATURES)\n"
        "row = next(r for r in chooser.walks[-1]['rungs'] if r['rung'] == 'all_resident')\n"
        "assert row['absent_bytes'] == 6_770_368_008, row\n"
        "assert row['verdict'] == 'short', row\n"
        "assert plan.placement == 'component_staged', plan.quantified_choice\n"
        "assert plan.reserved_vram_bytes == 6_937_675_734, plan.reserved_vram_bytes\n"
        "raise SystemExit('priced the absent weights: all_resident needs ' +\n"
        "    str(row['needs_bytes']) + ' B (of which ' + str(row['absent_bytes']) +\n"
        "    ' B is weight it must bring BACK) against ' + str(row['capacity_bytes']) +\n"
        "    ' B capacity -> ' + row['verdict'] + '; chose ' + plan.placement)\n",
        "priced the absent weights",
    ),
    (
        "host-move-doomed-refuses-before-the-oom-killer",
        "from cozy_runtime.internal.worker.ledger import Ledger, read_host_available\n"
        "Ledger(host_budget=1 << 60, pinned_budget=1 << 30).grant(\n"
        "    'doomed', read_host_available() + (4 << 30), 0, 'a move nobody can satisfy')\n",
        "host_move_doomed",
    ),
    (
        "witness-boot-record-on-ram",
        "import pathlib\nfrom cozy_runtime.internal.witness import require_durable\n"
        "require_durable(pathlib.Path('/dev/shm/cozy-witness-arm'))\n",
        "boot_record_on_ram",
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
from cozy_runtime.author.fakes import StagedBackend

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
            # Plant into the FIRST dependency list, whatever it currently holds. The old
            # plant matched one exact literal, so the day cr-007 added protobuf and grpcio
            # it silently became a no-op and the arm read green while proving nothing —
            # which is the failure mode this whole script exists to prevent.
            planted = original.replace("dependencies = [", arm.plant, 1)
            if planted == original:
                return "NOT PROVEN — the pyproject plant matched nothing to replace"
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


# ---------------------------------------------------------------- cr-003: the boundary
# Each arm builds a THROWAWAY endpoint project and runs the real `describe` verb over it.
# The refusal observed is the CLI's, exit code included — this is the build's own gate.

_VALID_MODULE = """
import msgspec
from cozy_runtime.author import App, Context, ImageAsset, ImageFrame, Outputs

app = App()


class In(msgspec.Struct):
    prompt: str


class Out(msgspec.Struct):
    image: ImageAsset


@app.entrypoint
def generate(ctx: Context, payload: In, out: Outputs) -> Out:
    return Out(image=out.save_image(ImageFrame(1, 1, b"rgb"), format="png"))
"""

BOUNDARY_ARMS: tuple[tuple[str, str, str, str, int], ...] = (
    (
        "zero_application",
        '[application]\nobject = "ep:app"\n',
        "import msgspec\nAPP = 1\n",
        "exports no 'app' and no App at all",
        3,
    ),
    (
        "no_application_named",
        "# an endpoint.toml that names nothing\n",
        _VALID_MODULE,
        "names no [application] object",
        3,
    ),
    (
        "not_an_app",
        '[application]\nobject = "ep:app"\n',
        "app = object()\n",
        "is a object, not an App",
        3,
    ),
    (
        "missing_dependency",
        '[application]\nobject = "ep:app"\n',
        "import a_dependency_this_release_forgot_to_lock\n" + _VALID_MODULE,
        "missing_dependency",
        3,
    ),
    (
        "broken_submodule_real_traceback",
        '[application]\nobject = "ep:app"\n',
        _VALID_MODULE + "\n_ = 1 / 0\n",
        "ZeroDivisionError",
        3,
    ),
    (
        "endpoint_job_name_collision",
        '[application]\nobject = "ep:app"\n',
        _VALID_MODULE
        + """

@app.job(name="generate")
def also_generate(payload: In) -> Out:
    raise NotImplementedError
""",
        "duplicate registration 'generate'",
        3,
    ),
    (
        "device_field_on_the_request",
        '[application]\nobject = "ep:app"\n',
        _VALID_MODULE.replace("    prompt: str", "    prompt: str\n    device: str = 'cuda:0'"),
        "excluded_keyword",
        3,
    ),
    (
        "author_declared_retryability",
        '[application]\nobject = "ep:app"\n',
        _VALID_MODULE.replace(
            "    image: ImageAsset", "    image: ImageAsset\n    retryable: bool = True"
        ),
        "deleted_surface",
        3,
    ),
    (
        "recursive_discovery",
        '[application]\nobject = "ep:app"\n',
        _VALID_MODULE,
        "",  # asserted by COUNT, below: the sibling's registrations must not appear
        0,
    ),
)

#: A sibling module that registers on its own `App` and is imported by nobody. A harvester
#: would find it; the one configured export is the complete algorithm, so describe cannot.
_SIBLING = """
import msgspec
from cozy_runtime.author import App

sibling_app = App()


class SideIn(msgspec.Struct):
    x: int


class SideOut(msgspec.Struct):
    y: int


@sibling_app.entrypoint
def harvested_if_recursive(payload: SideIn) -> SideOut:
    return SideOut(payload.x)
"""


def _project(tmp: pathlib.Path, endpoint_toml: str, module: str) -> pathlib.Path:
    project = tmp / "endpoint"
    (project / "src" / "ep").mkdir(parents=True, exist_ok=True)
    (project / "endpoint.toml").write_text(endpoint_toml)
    (project / "src" / "ep" / "__init__.py").write_text(module)
    return project


def _describe(
    project: pathlib.Path, *args: str, env: dict[str, str] | None = None
) -> tuple[int, str]:
    out = subprocess.run(
        [sys.executable, "-m", "cozy_runtime.cli.main", "describe", "--dir", str(project), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env or {})},
        check=False,
    )
    return out.returncode, out.stdout + out.stderr


def run_boundary(name: str, toml: str, module: str, expect: str, code: int) -> str:
    with tempfile.TemporaryDirectory(prefix=f"cozy-arm-{name}-") as tmp:
        project = _project(pathlib.Path(tmp), toml, module)
        if name == "recursive_discovery":
            (project / "src" / "ep" / "sibling.py").write_text(_SIBLING)
            got, output = _describe(project)
            if got != 0:
                return f"NOT PROVEN — {name}: describe failed ({output.strip()[:120]})"
            if "harvested_if_recursive" in output:
                return f"NOT PROVEN — {name}: a sibling module's registration was harvested"
            skip = ("counts", "surface_digest", "next")
            names = [
                ln.split(":")[0]
                for ln in output.splitlines()
                if " · " in ln and not ln.startswith(skip)
            ]
            return (
                f"red — {name}: the sibling registers on import and describe reports "
                f"{names} — no recursive discovery, no harvester"
            )
        got, output = _describe(project)
        if got != code or expect not in output:
            return f"NOT PROVEN — {name}: exit {got} (wanted {code}), output {output.strip()[:160]}"
        line = next(ln for ln in output.splitlines() if expect in ln)
        return f"red — {name} (exit {got}): {line.strip()[:150]}"


def run_stale_descriptor() -> list[str]:
    """The committed descriptor is a CHECKED CACHE: hand-edit it, and both --check and any
    build step that runs --check refuse."""
    results = []
    with tempfile.TemporaryDirectory(prefix="cozy-arm-stale-") as tmp:
        project = _project(pathlib.Path(tmp), '[application]\nobject = "ep:app"\n', _VALID_MODULE)
        code, _ = _describe(project, "--write-descriptor")
        assert code == 0, "the fixture must describe cleanly before it can be made stale"
        target = project / "endpoint.descriptor.json"
        good = target.read_text()

        document = json.loads(good)
        document["entrypoints"][0]["capabilities"] = ["save", "egress"]
        target.write_text(json.dumps(document, indent=2, sort_keys=True))
        code, output = _describe(project, "--check")
        results.append(
            f"red — stale_descriptor (exit {code}): {_first(output, 'different documents')}"
            if code == 13
            else f"NOT PROVEN — a hand-edited descriptor passed --check (exit {code})"
        )

        committed = good.split('"surface_digest": "')[1][:71]
        target.write_text(good.replace(committed, "sha256:" + "0" * 64))
        code, output = _describe(project, "--check")
        results.append(
            f"red — digest_line_edited (exit {code}): {_first(output, 'digests to')}"
            if code == 13
            else f"NOT PROVEN — an edited digest line passed --check (exit {code})"
        )

        target.write_text(good.replace('"schema":', '"smuggled": 1,\n  "schema":'))
        code, output = _describe(project, "--check")
        results.append(
            f"red — unknown_top_level_field (exit {code}): {_first(output, 'unknown top-level')}"
            if code == 13
            else f"NOT PROVEN — an unknown top-level field passed --check (exit {code})"
        )

        target.unlink()
        code, output = _describe(project, "--check")
        results.append(
            f"red — missing_descriptor (exit {code}): {_first(output, 'not committed')}"
            if code == 13
            else f"NOT PROVEN — a missing descriptor passed --check (exit {code})"
        )
    return results


def run_environment_drift() -> str:
    """Two environments claiming to be one release must derive one document (§1.0). Plant a
    DIFFERENT cozy_runtime ahead of the installed one and prove --check refuses."""
    with tempfile.TemporaryDirectory(prefix="cozy-arm-drift-") as tmp:
        drift = pathlib.Path(tmp) / "src"
        shutil.copytree(ROOT / "src", drift)
        target = drift / "cozy_runtime" / "internal" / "schema.py"
        target.write_text(
            target.read_text().replace(
                'DEFAULT_SOURCES = ("code_fallback"',
                'DEFAULT_SOURCES = ("drifted", "code_fallback"',
            )
        )
        code, output = _describe(ROOT / "examples", "--check", env={"PYTHONPATH": str(drift)})
    if code != 13:
        return f"NOT PROVEN — a drifted environment passed --check (exit {code})"
    return f"red — environment_drift (exit {code}): {_first(output, 'different documents')}"


def run_fixed_point() -> list[str]:
    """The descriptor's fixed point, planted four ways over the REAL built document."""
    source = (
        "import sys, json\n"
        "sys.path.insert(0, 'src')\n"
        "from cozy_runtime.internal import descriptor\n"
        "from cozy_runtime.internal.discovery import discover\n"
        "found = discover(__import__('pathlib').Path('examples'))\n"
        "body = descriptor.build(found)\n"
        "body{where} = {value}\n"
        "descriptor._fixed_point(body, descriptor._binding_values(found.bindings))\n"
        "print('NOT PROVEN')\n"
    )
    plants = (
        ("release_digest", "['entrypoints'][0]['release']", '"sha256:" + "ab" * 32'),
        ("build_timestamp", "['entrypoints'][0]['kind']", '"2026-08-24T11:02:00Z"'),
        ("machine_path", "['entrypoints'][0]['kind']", '"/home/fidika/cozy_v2/cozy-runtime"'),
        ("mutable_binding", "['entrypoints'][0]['kind']", '"cozy/h3-dual@v1"'),
    )
    out = []
    for name, where, value in plants:
        run = subprocess.run(
            [sys.executable, "-c", source.format(where=where, value=value)],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        if run.returncode == 0 or "fixed_point" not in run.stderr:
            out.append(f"NOT PROVEN — fixed_point/{name} was accepted into the descriptor")
            continue
        line = [ln for ln in run.stderr.splitlines() if "fixed_point" in ln][-1]
        out.append(f"red — fixed_point/{name}: {line.split('fixed_point: ')[-1].strip()[:120]}")
    return out


def _first(output: str, needle: str) -> str:
    line = next((ln for ln in output.splitlines() if needle in ln), output.splitlines()[0])
    return line.strip()[:140]


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
    results += [run_boundary(*a) for a in BOUNDARY_ARMS]
    results += run_stale_descriptor()
    results += run_fixed_point()
    results.append(run_environment_drift())
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
