#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
fixture="$repo/proofs/fixtures/model-contract-endpoint"
fake_torch="$repo/proofs/fixtures/fake-torch"
support="$repo/proofs/support/model_contract.py"
uv_bin=$(command -v uv)
proof_root=$(mktemp -d /tmp/cozy-model-contract-proof.XXXXXX)
fixture_source="$proof_root/endpoint-source"

cleanup() {
    chmod -R u+w "$proof_root" 2>/dev/null || true
    rm -rf -- "$proof_root"
}
trap cleanup EXIT HUP INT TERM

if grep -R -n -E 'from cozy_runtime\.internal|import cozy_runtime\.internal' \
    "$repo/proofs/model-contract" "$support"; then
    echo "public model-contract proof imports private Runtime modules" >&2
    exit 1
fi

cp -R "$fixture" "$fixture_source"
mkdir "$proof_root/build"
UV_OFFLINE=1 "$uv_bin" build --offline --wheel --out-dir "$proof_root/build" "$repo"
UV_OFFLINE=1 "$uv_bin" build --offline --wheel --out-dir "$proof_root/build" "$fixture_source"
UV_OFFLINE=1 "$uv_bin" build --offline --wheel --out-dir "$proof_root/build" "$fake_torch"

runtime_wheel=$(find "$proof_root/build" -maxdepth 1 -name 'cozy_runtime-*.whl' -print)
endpoint_wheel=$(find "$proof_root/build" -maxdepth 1 -name 'proof_endpoint-*.whl' -print)
torch_wheel=$(find "$proof_root/build" -maxdepth 1 -name 'torch-*.whl' -print)
test -n "$runtime_wheel" && test -n "$endpoint_wheel" && test -n "$torch_wheel"

UV_OFFLINE=1 "$uv_bin" venv --python "$(command -v python)" "$proof_root/driver"
UV_OFFLINE=1 "$uv_bin" pip install --offline --python "$proof_root/driver/bin/python" "$runtime_wheel"
driver_python="$proof_root/driver/bin/python"

(cd / && env -u PYTHONPATH UV_OFFLINE=1 PIP_NO_INDEX=1 \
    HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    "$proof_root/driver/bin/cozy-runtime" --dir "$fixture_source" \
    describe --write-descriptor >/dev/null)
(cd / && env -u PYTHONPATH UV_OFFLINE=1 PIP_NO_INDEX=1 \
    HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    "$proof_root/driver/bin/cozy-runtime" --dir "$fixture_source" describe --check >/dev/null)

(cd / && env -u PYTHONPATH UV_OFFLINE=1 PIP_NO_INDEX=1 \
    "$driver_python" "$support" prepare "$proof_root" \
    "$fixture_source/endpoint.descriptor.json" "$runtime_wheel" "$endpoint_wheel" \
    "$torch_wheel" "$uv_bin" "$driver_python")

for name in cold rebuild; do
    (cd / && env -u PYTHONPATH UV_OFFLINE=1 PIP_NO_INDEX=1 \
        HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
        "$proof_root/driver/bin/cozy-environment-proof" \
        "$proof_root/environment-$name.json" "$proof_root/receipt-$name.json" \
        >"$proof_root/environment-$name.stdout")
    (cd / && env -u PYTHONPATH UV_OFFLINE=1 PIP_NO_INDEX=1 \
        "$driver_python" "$support" model-request "$proof_root" "$name")
    generation=$(cd / && env -u PYTHONPATH "$driver_python" "$support" generation \
        "$proof_root/environment-$name.stdout")
    (cd / && env -u PYTHONPATH UV_OFFLINE=1 PIP_NO_INDEX=1 \
        HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
        "$generation/venv/bin/cozy-model-contract-proof" \
        "$proof_root/model-$name.json" "$proof_root/model-output-$name" \
        >"$proof_root/model-$name.stdout")
done

(cd / && env -u PYTHONPATH "$driver_python" "$support" red-request "$proof_root")
cold_generation=$(cd / && env -u PYTHONPATH "$driver_python" "$support" generation \
    "$proof_root/environment-cold.stdout")
set +e
(cd / && env -u PYTHONPATH UV_OFFLINE=1 PIP_NO_INDEX=1 \
    HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    "$cold_generation/venv/bin/cozy-model-contract-proof" \
    "$proof_root/model-red.json" "$proof_root/model-output-red" \
    >"$proof_root/model-red.stdout" 2>"$proof_root/model-red.stderr")
red_status=$?
set -e
test "$red_status" -eq 6
grep -q model_contract_binding_set_mismatch "$proof_root/model-red.stderr"
test ! -e "$proof_root/model-output-red"

(cd / && env -u PYTHONPATH "$driver_python" "$support" verify "$proof_root")
