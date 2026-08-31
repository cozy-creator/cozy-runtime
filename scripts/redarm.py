#!/usr/bin/env python3
"""Fence prover: PLANT each violation for real, observe the refusal, remove the plant.

Not a test — it mutates the working tree and restores it. Static arms run
`checks/architecture.py`
and expect it red; live arms execute the real runtime function and expect its typed refusal.
Run it whenever a fence changes: `uv run python scripts/redarm.py`.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, cast

from cozy_runtime.internal import descriptor as descriptor_document
from cozy_runtime.internal.discovery import discover

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "cozy_runtime"


@dataclass(frozen=True, slots=True)
class Arm:
    fence: str
    path: pathlib.Path
    plant: str


@dataclass(frozen=True, slots=True)
class OmissionArm:
    fence: str
    path: pathlib.Path
    remove: str


STATIC_ARMS = (
    Arm(
        "prelaunch-version",
        SRC / "__init__.py",
        '__version__ = "1.0.0"\n',
    ),
    Arm(
        "author-import-boundary",
        ROOT / "examples" / "marco-polo" / "marco_polo_package" / "__init__.py",
        "from cozy_runtime.author._app import App\n",
    ),
    Arm("env-authority", SRC / "internal" / "child_env.py", "import os\nS = os.environ.get('X')\n"),
    Arm(
        "construction-fill-order",
        SRC / "internal" / "fill.py",
        "\n\ndef _planted_header_order(rows: list[object]) -> None:\n"
        "    self.order = [row.key for row in rows]\n",
    ),
    Arm(
        "construction-fill-order",
        SRC / "internal" / "derive.py",
        "\n\ndef _planted_mcc_order(rich: object) -> object:\n"
        "    return tuple(sorted(rich.sets, key=lambda row: row.component))\n",
    ),
    Arm(
        "delivery-plan-observation",
        SRC / "internal" / "worker" / "session.py",
        "\n\ndef _planted_stale_delivery_reader(delivery: dict[str, object]) -> object:\n"
        "    return delivery['cost_class']\n",
    ),
    Arm(
        "measured-backend-version",
        SRC / "internal" / "hostfacts.py",
        "\n\ndef _cuda_version() -> str:\n"
        "    # Planted old authority: driver maximum 13.0, not installed toolkit 12.6.\n"
        "    subprocess.run(['nvidia-smi', '--display=COMPUTE'])\n"
        "    return '13.0'\n",
    ),
    # cr-042: the third private-address check, planted in the module whose own header says
    # the defect is "there were two downloaders and only one had the check".
    Arm(
        "one-address-predicate",
        SRC / "internal" / "pull.py",
        "\n\ndef _planted_second_ssrf_check(addr: object) -> bool:\n"
        "    return addr.is_private or addr.is_loopback\n",
    ),
    Arm("package-imports", SRC / "author" / "_app.py", "import cozy_runtime.internal.config\n"),
    Arm("package-imports", SRC / "internal" / "config.py", "from cozy_runtime.cli import io\n"),
    Arm(
        "package-imports",
        SRC / "internal" / "readiness.py",
        "from cozy.worker.v1 import worker_pb2\n",
    ),
    Arm("forbidden-dependency", SRC / "internal" / "config.py", "import varena\n"),
    Arm("forbidden-dependency", ROOT / "pyproject.toml", 'dependencies = ["tensorhub", '),
    Arm(
        "document-kinds",
        SRC / "protocol" / "documents.py",
        "\n\ndef doc_format(full_name: str) -> str:\n    return f'{full_name}/2'\n",
    ),
    Arm(
        "typed-model-binding",
        ROOT / "proofs" / "fixtures" / "tiny-models" / "package.toml",
        '\nrepo = "legacy/model"\n',
    ),
    Arm("single-config", SRC / "cli" / "version.py", "def read_config() -> None: ...\n"),
    Arm("single-config", SRC / "cli" / "io.py", "CONFIG = {'smuggled': True}\n"),
    Arm("no-interactive", SRC / "cli" / "new.py", "ANSWER = input('name? ')\n"),
    Arm(
        "artifact-object-sets",
        ROOT / "vectors" / "artifact-object-sets.jsonl",
        '{"name":"planted-digest-drift","document":{"kind":'
        '"tensorhub.resolved_object_set/1","roots":[]},"canonical":'
        '"{\\"kind\\":\\"tensorhub.resolved_object_set/1\\",\\"roots\\":[]}",'
        '"digest":"sha256:' + "0" * 64 + '"}\n',
    ),
    Arm(
        "base-ownership-vectors",
        ROOT / "vectors" / "base-ownership.jsonl",
        '{"document":{},"expect":"ok","name":"planted-invalid-base"}\n',
    ),
    # The exact drift that cost a rented H200: a reviewed spec digest this build declares
    # and the installed registry does not carry. Planted as a well-formed digest of the
    # right shape, because a malformed one would be caught by looking rather than by asking.
    Arm(
        "reviewed-specs-exist",
        SRC / "internal" / "encoding" / "formats.py",
        'SPEC_PLANTED_UNREVIEWED = "sha256:' + "de" * 32 + '"\n',
    ),
    Arm(
        "excluded-keywords",
        SRC / "author" / "_context.py",
        "def place(device: str) -> None: ...\n",
    ),
    Arm(
        "no-identifiers-in-package-code",
        ROOT / "proofs" / "fixtures" / "tiny-models" / "tiny_h3.py",
        '_PINNED = "minimaxai/h3@v1"\n',
    ),
    # §8's one-kernel rule, planted as the exact thing it forbids: a SECOND direct-call
    # path into the author invocation kernel, and a SECOND attempt engine beside the
    # worker's. Both are the v1 failure mode where the developer door and the
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
    # cr-008b's done-when as a plant: a package that can move bytes is a second policy.
    Arm(
        "package-owns-no-residency",
        ROOT / "corpus" / "pipeline" / "sdxl_txt2img.py",
        "def _reclaim() -> None:\n    import torch\n\n    torch.cuda.empty_cache()\n",
    ),
    Arm(
        "package-owns-no-residency",
        ROOT / "corpus" / "sdxl_pipeline.py",
        "def _offload(pipe: object) -> None:\n    pipe.enable_model_cpu_offload()\n",
    ),
    Arm(
        "package-owns-no-media-decode",
        ROOT / "proofs" / "fixtures" / "tiny-models" / "tiny_h3.py",
        "def _second_decoder() -> None:\n    import av\n\n    av.open('request.mp4')\n",
    ),
    Arm(
        "package-owns-no-media-decode",
        ROOT / "proofs" / "fixtures" / "tiny-models" / "tiny_h3.py",
        "def _steal_asset_path(asset: object) -> object:\n    return asset._local\n",
    ),
    Arm(
        "package-owns-no-media-decode",
        ROOT / "proofs" / "fixtures" / "tiny-models" / "tiny_h3.py",
        "def _library_path_decode(reference: object) -> object:\n"
        "    return reference.from_file('request.mp4')\n",
    ),
    Arm(
        "package-owns-no-media-decode",
        ROOT / "proofs" / "fixtures" / "tiny-models" / "tiny_h3.py",
        "def _packet_decode(container: object) -> object:\n    return container.demux()\n",
    ),
    Arm(
        "timing-decides-nothing",
        SRC / "author" / "_observations.py",
        "from cozy_runtime.internal.fill import StreamingFillBackend\n",
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

OMISSION_ARMS = (
    OmissionArm(
        "python-abi",
        SRC / "_interpreter.py",
        '    or sysconfig.get_config_var("Py_GIL_DISABLED") == 1\n',
    ),
    OmissionArm(
        "dynamic-pods-skip-boot-warm",
        SRC / "cli" / "runtime_worker.py",
        "        no_warm=True,\n",
    ),
    OmissionArm(
        "pod-media-grant-root",
        SRC / "cli" / "runtime_worker.py",
        "        grant_roots=(str(host_paths.MEDIA_ROOT),),\n",
    ),
    OmissionArm(
        "measured-backend-version",
        SRC / "internal" / "worker" / "session.py",
        "            backend_version=facts.cuda_version,\n",
    ),
)

#: A committed bundle, edited on disk afterwards, must be a DETECTED corruption rather than
#: a plausible story — the recorded write receipt is what makes that possible.
TAMPER = """
import json, pathlib, shutil, tempfile
from cozy_runtime.internal.worker import triage
from cozy_runtime.internal.worker.records import WorkerRecords

root = pathlib.Path(tempfile.mkdtemp(prefix="cozy-triage-arm"))
records = WorkerRecords()
store = triage.BundleStore(root / "triage", records)
receipt = store.commit(triage.Assembly(triage.mint_subject(), attempt={"request_id": "r"}))
path = root / "triage" / (receipt.subject_id + ".json")
document = json.loads(path.read_text())
document["terminal"] = {"status": "OUTCOME_STATUS_SUCCEEDED"}
path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
try:
    triage.BundleStore.read(root / "triage", receipt.subject_id, records)
finally:
    shutil.rmtree(root, ignore_errors=True)
"""

#: A log line cannot declare terminal success. The bundle has no field that can carry a
#: terminal claim, and the ONE terminal fact it holds is a copy of the recorded terminal.
LOG_AUTHORITY = """
import pathlib, shutil, tempfile
from cozy_runtime.internal.worker import triage
from cozy_runtime.internal.worker.records import WorkerRecords

root = pathlib.Path(tempfile.mkdtemp(prefix="cozy-authority-arm"))
records = WorkerRecords()
store = triage.BundleStore(root / "triage", records)
assembly = triage.Assembly(triage.mint_subject(), terminal={"status": "OUTCOME_STATUS_FAILED"})
assembly.events.append({"kind": "log", "name": "SUCCEEDED", "value": "the handler is done"})
receipt = store.commit(assembly)
document = triage.BundleStore.read(root / "triage", receipt.subject_id, records)
shutil.rmtree(root, ignore_errors=True)
claims = [row for row in document["events"] if row.get("kind") not in ("log", "metric",
          "stage", "confession", "boot", "liveness", "fault")]
assert not claims, claims
assert document["terminal"]["status"] == "OUTCOME_STATUS_FAILED", document["terminal"]
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
import dataclasses

from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    PlanChooser,
    PreparedModel,
    PreparedRequest,
)

REQUEST = PreparedRequest.unresolved("generate", {"steps": 20})
GENERATION = PreparedModel()
ledger = Ledger(host_budget=8 << 30, pinned_budget=512 << 20)
ledger.resident = {"unet": 5_134_927_368, "vae": 167_307_726}
ledger.device_free = 878_733_824
ledger.device_total = 8_186_822_656
ledger.device_allocated = 5_302_235_094
ledger.device_reserved = 5_400_000_000
chooser = PlanChooser(ledger)


BASE = DeclaredBinding(
    entrypoint_binding_plan_id="sha256:" + "0" * 64,
    entrypoint="generate", model_class="M", model_binding_path="b",
    model_parameter_name="m", release="r", store="s", config="c",
    components=("unet",), snapshots={"unet": "sha256:" + "0" * 64},
    reference_snapshot="sha256:" + "0" * 64, variant="sm89",
    model_weight_bytes=6 << 30, host_bytes=1 << 30, pinned_bytes=256 << 20,
)


def binding(**over):
    return dataclasses.replace(BASE, **over)


def choose(**over):
    return chooser.choose(binding(**over), GENERATION, REQUEST)
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
        # cr-051: the value guard inspects bytes and is the real defence; the NAME guard's
        # residual value is a credential in an unrecognised format. It must not eat the most
        # obvious fields an inference package emits — `tokens_per_second` and
        # `output_tokens` — while still refusing a field named for a credential.
        "observation-metric-names-are-not-credentials",
        "from cozy_runtime.author._observations import EventRing, ObservationRefusal\n"
        "ring = EventRing()\n"
        "admitted = ('tokens_per_second', 'output_tokens', 'input_tokens', 'total_tokens',\n"
        "            'token_count', 'eos_token', 'kernel_signature', 'signatures')\n"
        "for name in admitted:\n"
        "    ring.emit('metric', name, 1.0)\n"
        "    ring.emit('log', 'decoded', 'info', **{name: 1})\n"
        "assert ring.caps()['refused'] == 0, ring.caps()\n"
        "refused = ('token', 'signature', 'access_token', 'api_token', 'refresh_token',\n"
        "           'session_token', 'oauth_token', 'api_signature', 'auth_token',\n"
        "           'api_key', 'password', 'private_key', 'cookie')\n"
        "for name in refused:\n"
        "    try:\n"
        "        ring.emit('log', 'fetching', 'info', **{name: 'ya29.a0Af'})\n"
        "    except ObservationRefusal as exc:\n"
        "        assert exc.code == 'credential_field', (name, exc.code)\n"
        "    else:\n"
        "        raise SystemExit('ADMITTED a credential-named field: ' + name)\n"
        "assert ring.caps()['refused'] == len(refused), ring.caps()\n"
        "raise SystemExit('admitted %d descriptive names, refused %d credential names'\n"
        "    % (len(admitted), len(refused)))\n",
        "admitted 8 descriptive names, refused 13 credential names",
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
        # A slot that declares a component the artifact does not carry keeps the NAME so
        # the fill refuses it by name; narrowing it away silently would serve a graph the
        # class never declared.
        "declared-component-the-artifact-lacks-is-not-narrowed-away",
        LADDER + "b = binding(components=['unet', 'refiner'])\n"
        "assert 'refiner' in b.components, b.components\n"
        "raise SystemExit('kept the undeliverable component: ' + repr(b.components))\n",
        "kept the undeliverable component",
    ),
    (
        "uncalibrated-cannot-refuse",
        LADDER + "plan = choose(tenancy='local')\n"
        "assert plan.placement == 'component_staged', plan.placement\n"
        "raise SystemExit('an uncalibrated coefficient DEEPENED to ' + plan.placement +\n"
        "    ' and refused nothing: ' + plan.decision_explanation[-140:])\n",
        "an uncalibrated coefficient DEEPENED",
    ),
    (
        "shared-tenancy-reschedules",
        LADDER + "from cozy_runtime.internal.worker.plan import PlanRefusal\n"
        "ledger.activations_by_cell = {REQUEST.cell(): 8 << 30}\n"
        "try:\n"
        "    choose(tenancy='shared')\n"
        "except PlanRefusal as exc:\n"
        "    assert exc.code == 'capacity_shortfall', exc.code\n"
        "    assert exc.shortfall['scope'] == 'worker', exc.shortfall\n"
        "    raise SystemExit(exc.detail[:200])\n",
        "not a structural incompatibility",
    ),
    (
        "local-floor-exits-quantified",
        LADDER + "from cozy_runtime.internal.worker.plan import PlanRefusal\n"
        "ledger.activations_by_cell = {REQUEST.cell(): 8 << 30}\n"
        "try:\n"
        "    choose(tenancy='local', vram_floor_bytes=64 << 30)\n"
        "except PlanRefusal as exc:\n"
        "    assert exc.code == 'capacity_floor', exc.code\n"
        "    raise SystemExit(exc.detail[:220])\n",
        "below the PHYSICAL FLOOR",
    ),
    (
        "local-capacity-never-refuses-above-the-floor",
        LADDER + "ledger.activations_by_cell = {REQUEST.cell(): 8 << 30}\n"
        "plan = choose(tenancy='local', vram_floor_bytes=1 << 20)\n"
        "raise SystemExit('DEGRADED rather than refused: ' + plan.decision_explanation[-150:])\n",
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
        "    choose(tenancy='local')\n"
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
        "plan = choose(tenancy='local')\n"
        "row = next(r for r in chooser.walks[-1]['rungs'] if r['rung'] == 'all_resident')\n"
        "assert row['absent_bytes'] == 6_770_368_008, row\n"
        "assert row['verdict'] == 'short', row\n"
        "assert plan.placement == 'component_staged', plan.decision_explanation\n"
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
    # ------------------------------------------------------------------ #574b
    # The fill plane's device accounting, armed where it is arithmetic. Both bodies carry
    # the REAL H3 numbers — a 92.42 GiB stored / 128.31 GiB logical N-ary artifact whose
    # transformer is fp8-stored — so the fence is provable without a card.
    (
        "decode-scratch-rides-the-device-envelope",
        "from cozy_runtime.internal.fill import price_envelope\n"
        # The decode scratch is DEVICE memory. It used to ride a fill-local counter that
        # the refusal never printed (#574b); since #613 the only arithmetic is admission,
        # and the envelope must carry the scratch beside the destinations or a fill can
        # be admitted at its weights alone.
        "env = price_envelope("
        "[('transformer', 40_225_668_192, 20_958_085_592, 'decoded_float')], 512 << 20)\n"
        "assert env['decode_peak_bytes'] > 0\n"
        "assert env['required_bytes'] == env['destination_bytes'] + env['decode_peak_bytes']\n"
        "raise SystemExit('decode scratch of %d B rides the device envelope: %d B required' "
        "% (env['decode_peak_bytes'], env['required_bytes']))\n",
        "rides the device envelope",
    ),
    (
        "stored-bytes-are-not-resident-bytes",
        "from cozy_runtime.internal.fill import price_envelope\n"
        # transformer: fp8-stored, bf16-resident. text_encoder/video_vae/audio_vae: plain.
        "rows = [\n"
        "  ('transformer', 40_225_668_192, 20_958_085_592, 'decoded_float'),\n"
        "  ('text_encoder', 51_506_191_840, 51_506_191_840, 'verbatim'),\n"
        "  ('video_vae', 5_207_742_160, 5_207_742_160, 'verbatim'),\n"
        "  ('audio_vae', 605_149_280, 605_149_280, 'verbatim'),\n"
        "]\n"
        "env = price_envelope(rows, 512 << 20)\n"
        "stored = sum(r[2] for r in rows)\n"
        "assert env['destination_bytes'] == 97_544_751_472, env['destination_bytes']\n"
        "assert env['destination_bytes'] > stored, 'the fp8 rows did not expand'\n"
        "raise SystemExit(\n"
        "  'destinations %d B against %d B stored: the envelope EXPANDS the fp8 rows by "
        "%d B, which is what the record priced at zero'\n"
        "  % (env['destination_bytes'], stored, env['destination_bytes'] - stored))\n",
        "the envelope EXPANDS the fp8 rows",
    ),
    (
        # THE REPLY THAT COULD NOT BE SENT (#574b's root cause). An oversized PREPARE reply
        # used to arrive as `{"ok": True}` with none of prepare's keys, because the minimal
        # fallback was shaped for INVOKE and copied `ok` through. The worker then read a key
        # that was not there, the control stream died AFTER the fill had 90.84 GiB on the
        # card, and the re-accepted placement filled a SECOND generation into the same
        # allocator. A reply this path has to mutilate refuses instead.
        "an-oversized-reply-refuses-rather-than-inventing-a-success",
        "from cozy_runtime.internal import executor\n"
        "from cozy_runtime.internal.seam import MAX_FRAME, SeamError\n"
        "sent = []\n"
        "class Cap:\n"
        "    def send(self, frame):\n"
        "        import json\n"
        "        if len(json.dumps(frame)) > MAX_FRAME:\n"
        "            raise SeamError('frame_too_large', 'over the control cap')\n"
        "        sent.append(frame)\n"
        # A prepare-shaped reply whose plan census does not fit one control frame.
        "executor._send_reply(Cap(), {\n"
        "    'reply': 'prepare', 'ok': True,\n"
        "    'model_construction_digest': 'sha256:' + '0' * 64,\n"
        "    'constructed_model_digest': 'sha256:' + '1' * 64,\n"
        "    'plan': {'tensors': [{'key': 'k%d' % i, 'route': 'verbatim'} for i in range(4000)]},\n"
        "    'facts': {'filled': 2651},\n"
        "})\n"
        "out = sent[-1]\n"
        "assert out['ok'] is False, out\n"
        "raise SystemExit('%s: %s' % (out['code'], out['detail'][:150]))\n",
        "reply_too_large",
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
sys.path.insert(0, 'proofs/fixtures/tiny-models')
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
sys.path.insert(0, 'proofs/fixtures/tiny-models')
sys.path.insert(0, 'proofs/internal')
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

def handler(payload: In, model: sd.SdxlModel = Slot(sd.SdxlModel, model='cozy/sdxl')) -> Meta:
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
        # #572d: a HIDDEN entrypoint remains a local registry fact but is not published or
        # served. Descriptor construction owns the omission; this arm proves the author flag.
        "hidden-entrypoint-is-local-and-not-served",
        """
from cozy_runtime.author._signature import analyze

def visible(ctx: Context, payload: In) -> Meta: ...
def concealed(ctx: Context, payload: In) -> Meta: ...

a = analyze(visible, name='visible', kind='entrypoint')
b = analyze(concealed, name='concealed', kind='entrypoint', hidden=True)
assert a.hidden is False and b.hidden is True, (a.hidden, b.hidden)
serving = [s.name for s in (a, b) if not s.hidden]
assert serving == ['visible'], serving
raise SystemExit('the serving set is ' + repr(serving) + ' and the hidden surface stayed local')
""",
        "the serving set is ['visible']",
    ),
    (
        # #573: a GENERIC attribute probe over a guarded root refuses. This is the shape
        # that poisoned H3's first successful fill — the runtime's own scheduler scan
        # `hasattr`-probing a component it had just guarded — and it is armed here so the
        # class of bug is caught by a test rather than by a rented card.
        "generic-attribute-probe-over-a-guarded-root",
        """
from cozy_runtime.author._model import GuardedComponent

class M(Model): pass
m = M.__new__(M)
object.__setattr__(m, '_cozy_active', None)
object.__setattr__(m, '_cozy_poison', None)
hasattr(GuardedComponent('transformer', object(), m), 'clone')
""",
        "undeclared_component",
    ),
    (
        "stamp-fails-closed",
        "registry().acquire('g.models.m', h3.Ref2VAModel, art.h3(task='fl2va'))\n",
        "stamp_mismatch",
    ),
    (
        # #567d: MEMBERSHIP is not looseness. The DUAL artifact — the canonical N-ary
        # packaging every multi-task family ships — carries `task=fl2va+ref2va` and admits
        # both role classes; a stamp set that does NOT contain the class's value still
        # refuses, and so does a set that is missing entirely.
        "stamp-membership-still-refuses-a-foreign-task",
        "dual = art.h3(task='fl2va', dual=True)\n"
        "assert dual.stamp_set('task') == ('fl2va', 'ref2va'), dual.stamp_set('task')\n"
        "registry().acquire('a.models.m', h3.Fl2VAModel, dual)\n"
        "registry().acquire('b.models.m', h3.Ref2VAModel, dual)\n"
        "foreign = Artifact(snapshot=dual.snapshot, topology=dual.topology,\n"
        "                   config=dual.config, stamps={'task': ('fl2va', 't2va')})\n"
        "registry().acquire('c.models.m', h3.Ref2VAModel, foreign)\n",
        "stamp_mismatch",
    ),
    (
        # #567d, the other half: the serving path used to pass NO stamps at all, so this
        # was the state of EVERY checkpoint on EVERY binding.
        "unstamped-artifact-fails-every-stamped-class-closed",
        "bare = Artifact(snapshot='cozy/h3@v1#bare',\n"
        "                topology=art.h3(task='fl2va').topology,\n"
        "                config=art.h3(task='fl2va').config)\n"
        "registry().acquire('g.models.m', h3.Fl2VAModel, bare)\n",
        "carries task=nothing",
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
    planted = ROOT / "proofs" / "fixtures" / "tiny-models" / "_planted_separation.py"
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
                "import sys; sys.path.insert(0, 'proofs/fixtures/tiny-models')\n"
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
        [sys.executable, str(ROOT / "checks" / "architecture.py")],
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


def run_omission(arm: OmissionArm) -> str:
    """Remove one required spelling, observe its fence red, then restore exact bytes."""

    original = arm.path.read_text()
    if original.count(arm.remove) != 1:
        return f"NOT PROVEN — {arm.fence} omission matched {original.count(arm.remove)} places"
    try:
        arm.path.write_text(original.replace(arm.remove, "", 1))
        checked = subprocess.run(
            [
                sys.executable,
                str(ROOT / "checks" / "architecture.py"),
                "--only",
                arm.fence,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        code, output = checked.returncode, checked.stdout + checked.stderr
    finally:
        arm.path.write_text(original)
    if code == 0:
        return f"NOT PROVEN — {arm.fence} stayed green with the required argv flag omitted"
    lines = output.splitlines()
    line = next((ln.strip() for ln in lines if arm.path.name in ln), lines[0])
    return f"red — {arm.fence} omission ({arm.path.relative_to(ROOT)}): {line}"


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
    planted = ROOT / "proofs" / "fixtures" / "tiny-models" / "_planted_mistyped.py"
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
                "import sys; sys.path.insert(0, 'proofs/fixtures/tiny-models')\n"
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
# Each arm builds a THROWAWAY package project and runs the real `describe` verb over it.
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
        "# a package.toml that names nothing\n",
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
        "package_job_name_collision",
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


def _project(tmp: pathlib.Path, package_toml: str, module: str) -> pathlib.Path:
    project = tmp / "package"
    (project / "src" / "ep").mkdir(parents=True, exist_ok=True)
    (project / "package.toml").write_text(package_toml)
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
            skip = ("counts", "package_descriptor_digest", "next")
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


def run_retired_descriptor_shapes() -> list[str]:
    """The descriptor/1 hardcut has no reader for its four redundant old spellings."""
    base = descriptor_document.build(discover(ROOT / "proofs" / "fixtures" / "tiny-models"))

    def clone() -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(json.dumps(base)))

    def refused(name: str, document: dict[str, Any]) -> str:
        try:
            descriptor_document.read_bytes(json.dumps(document).encode(), name)
        except descriptor_document.StalePackageDescriptor as exc:
            if exc.code == "malformed_package_descriptor":
                return f"red — retired_descriptor/{name}: {str(exc)[:120]}"
            return f"NOT PROVEN — {name} refused with {exc.code}, not malformed_package_descriptor"
        return f"NOT PROVEN — retired descriptor shape {name} was accepted"

    hidden = clone()
    hidden["entrypoints"][0]["hidden"] = False

    param = clone()
    modeled = next(row for row in param["entrypoints"] if row.get("models"))
    slot = modeled["models"][0]
    slot["param"] = slot["path"].rsplit(".", 1)[-1]

    required_wire = clone()
    field = next(
        field
        for row in required_wire["entrypoints"]
        for field in row["request"]["fields"]
        if "wire" not in field
    )
    field["wire"] = "required"

    old_tag = clone()

    def tagged(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(value, dict):
            union = value.get("union")
            tag_field = value.get("tag_field")
            if isinstance(union, list) and isinstance(tag_field, str):
                member = next(
                    (item for item in union if isinstance(item, dict) and "tag" in item), None
                )
                if isinstance(member, dict):
                    return cast(dict[str, Any], value), cast(dict[str, Any], member)
            for child in value.values():
                try:
                    return tagged(child)
                except LookupError:
                    pass
        elif isinstance(value, list):
            for child in value:
                try:
                    return tagged(child)
                except LookupError:
                    pass
        raise LookupError("the real example has no tagged union")

    union, member = tagged(old_tag)
    member["tag_field"] = union["tag_field"]
    member["fields"].insert(
        0,
        {
            "name": union["tag_field"],
            "type": {"literal": [member["tag"]]},
            "wire": "required",
        },
    )

    return [
        refused("hidden", hidden),
        refused("param", param),
        refused("required-wire", required_wire),
        refused("repeated-tag-field", old_tag),
    ]


def run_fixed_point() -> list[str]:
    """The descriptor's fixed point, planted four ways over the REAL built document."""
    source = (
        "import sys, json\n"
        "sys.path.insert(0, 'src')\n"
        "from cozy_runtime.internal import descriptor\n"
        "from cozy_runtime.internal.discovery import discover\n"
        "found = discover(__import__('pathlib').Path('proofs/fixtures/tiny-models'))\n"
        "body = descriptor.build(found)\n"
        "body{where} = {value}\n"
        "descriptor._fixed_point(body, descriptor._binding_values(found.bindings))\n"
        "print('NOT PROVEN')\n"
    )
    plants = (
        ("release_digest", "['entrypoints'][0]['release']", '"sha256:" + "ab" * 32'),
        ("build_timestamp", "['entrypoints'][0]['name']", '"2026-08-24T11:02:00Z"'),
        ("machine_path", "['entrypoints'][0]['name']", '"/home/fidika/cozy_v2/cozy-runtime"'),
        ("mutable_binding", "['entrypoints'][0]['name']", '"cozy/h3-dual@v1"'),
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
    results += [run_omission(a) for a in OMISSION_ARMS]
    results += [run_live(*a) for a in LIVE_ARMS]
    results += [run_live(name, PRELUDE + body, expect) for name, body, expect in AUTHOR_ARMS]
    results += [run_live(name, MODEL_PRELUDE + body, expect) for name, body, expect in MODEL_ARMS]
    results.append(run_mistyped())
    results.append(run_separation())
    results += [run_boundary(*a) for a in BOUNDARY_ARMS]
    results += run_retired_descriptor_shapes()
    results += run_fixed_point()
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
