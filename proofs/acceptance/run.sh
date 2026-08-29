#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
proof_root=$(mktemp -d /tmp/cozy-acceptance-proof.XXXXXX)

cleanup() {
    chmod -R u+w "$proof_root" 2>/dev/null || true
    rm -rf -- "$proof_root"
}
trap cleanup EXIT HUP INT TERM

mkdir "$proof_root/build"
UV_OFFLINE=1 uv build --offline --wheel --out-dir "$proof_root/build" "$repo"
UV_OFFLINE=1 uv build --offline --wheel --out-dir "$proof_root/build" \
    "$repo/proofs/fixtures/marco-polo-endpoint"
runtime_wheel=$(find "$proof_root/build" -name 'cozy_runtime-*.whl' -print)
project_wheel=$(find "$proof_root/build" -name 'marco_polo_endpoint-*.whl' -print)

UV_OFFLINE=1 uv venv --python "$(command -v python)" "$proof_root/driver"
UV_OFFLINE=1 uv pip install --offline --python "$proof_root/driver/bin/python" \
    "$runtime_wheel" "$project_wheel"
python="$proof_root/driver/bin/python"
proof="$proof_root/driver/bin/cozy-acceptance-proof"
driver="$repo/proofs/internal/acceptance_observation.py"
descriptor="$repo/proofs/fixtures/marco-polo-endpoint/endpoint.descriptor.json"

(cd / && env -u PYTHONPATH PATH="$PATH" HOME="$proof_root/home" \
    "$python" "$driver" setup "$proof_root" "$project_wheel" "$descriptor")
(cd / && env -u PYTHONPATH "$proof" fixture "$proof_root/fixture-request.json" \
    "$proof_root/fixture-payload.json")
(cd / && env -u PYTHONPATH PATH="$PATH" HOME="$proof_root/home" \
    "$python" "$driver" run "$proof_root")
(cd / && env -u PYTHONPATH "$proof" observe "$proof_root/observation-request.json" \
    "$proof_root/execution-observation.json")
(cd / && env -u PYTHONPATH "$python" "$driver" failure "$proof_root")
(cd / && env -u PYTHONPATH "$proof" observe "$proof_root/failed-request.json" \
    "$proof_root/failed-observation.json")
(cd / && env -u PYTHONPATH "$python" "$driver" verify "$proof_root")

# Closed request: a planted extra authority is a typed refusal with no partial evidence.
"$python" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); d["candidate_id"]="hub-owned"; json.dump(d,open(sys.argv[2],"w"))' \
    "$proof_root/observation-request.json" "$proof_root/extra.json"
set +e
(cd / && env -u PYTHONPATH "$proof" observe "$proof_root/extra.json" "$proof_root/red.json") \
    >"$proof_root/red.stdout" 2>"$proof_root/red.stderr"
status=$?
set -e
test "$status" -eq 6
grep -q acceptance_request_invalid "$proof_root/red.stderr"
test ! -e "$proof_root/red.json"

# An exact-looking document outside the declared input root cannot be read through a
# symlinked parent or an absolute host path.
"$python" -c 'import hashlib,json,sys; d=json.load(open(sys.argv[1])); b=open("/etc/passwd","rb").read(); d["endpoint_descriptor"]={"digest":"sha256:"+hashlib.sha256(b).hexdigest(),"length":len(b),"path":"/etc/passwd"}; json.dump(d,open(sys.argv[2],"w"))' \
    "$proof_root/observation-request.json" "$proof_root/escape.json"
set +e
(cd / && env -u PYTHONPATH "$proof" observe "$proof_root/escape.json" "$proof_root/escape-out.json") \
    >"$proof_root/escape.stdout" 2>"$proof_root/escape.stderr"
status=$?
set -e
test "$status" -eq 6
grep -q acceptance_path_outside_input_root "$proof_root/escape.stderr"
test ! -e "$proof_root/escape-out.json"
