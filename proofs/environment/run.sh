#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
support="$repo/proofs/support/environment.py"
import_probe="$repo/proofs/support/environment_imports.py"
shared_support="$repo/proofs/support/proof_data.py"
private_proof="$repo/proofs/internal/environment_materializer.py"
red_fixture="$repo/proofs/fixtures/environment-private-import-red.txt"
uv_bin=$(command -v uv)
proof_root=$(mktemp -d /tmp/cozy-environment-proof.XXXXXX)

cleanup() {
    chmod -R u+w "$proof_root" 2>/dev/null || true
    rm -rf -- "$proof_root"
}
trap cleanup EXIT HUP INT TERM

private_imports() {
    grep -R -n -E 'from cozy_runtime\.internal|import cozy_runtime\.internal' "$@"
}

if private_imports "$repo/proofs/environment" "$support" "$import_probe" "$shared_support"; then
    echo "public environment proof imports private Runtime modules" >&2
    exit 1
fi
if ! private_imports "$red_fixture" >/dev/null; then
    echo "public proof private-import fence did not detect its planted violation" >&2
    exit 1
fi

mkdir "$proof_root/build"
UV_OFFLINE=1 "$uv_bin" build --offline --wheel --out-dir "$proof_root/build" "$repo"
for fixture in environment-project environment-resolved environment-wheelhouse fake-torch; do
    UV_OFFLINE=1 "$uv_bin" build --offline --wheel --out-dir "$proof_root/build" \
        "$repo/proofs/fixtures/$fixture"
done

runtime_wheel=$(find "$proof_root/build" -maxdepth 1 -name 'cozy_runtime-*.whl' -print)
project_wheel=$(find "$proof_root/build" -maxdepth 1 \
    -name 'proof_environment_project-*.whl' -print)
resolved_wheel=$(find "$proof_root/build" -maxdepth 1 \
    -name 'proof_environment_resolved-*.whl' -print)
wheelhouse_wheel=$(find "$proof_root/build" -maxdepth 1 \
    -name 'proof_environment_wheelhouse-*.whl' -print)
torch_wheel=$(find "$proof_root/build" -maxdepth 1 -name 'torch-*.whl' -print)
test -n "$runtime_wheel" && test -n "$project_wheel" && test -n "$resolved_wheel"
test -n "$wheelhouse_wheel" && test -n "$torch_wheel"

UV_OFFLINE=1 "$uv_bin" venv --python "$(command -v python)" "$proof_root/driver"
UV_OFFLINE=1 "$uv_bin" pip install --offline --python "$proof_root/driver/bin/python" \
    "$runtime_wheel"
driver_python="$proof_root/driver/bin/python"
proof="$proof_root/driver/bin/cozy-environment-proof"

(cd / && env -u PYTHONPATH UV_OFFLINE=1 PIP_NO_INDEX=1 \
    HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    ALL_PROXY=http://127.0.0.1:9 \
    "$driver_python" "$support" prepare "$proof_root" "$project_wheel" \
    "$resolved_wheel" "$wheelhouse_wheel" "$torch_wheel" "$uv_bin" "$driver_python")

run_proof() {
    name=$1
    receipt=$2
    (cd / && env -u PYTHONPATH UV_OFFLINE=1 PIP_NO_INDEX=1 \
        HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
        ALL_PROXY=http://127.0.0.1:9 \
        "$proof" "$proof_root/request-$name.json" "$receipt")
}

for name in cold rebuild cuda; do
    run_proof "$name" "$proof_root/receipt-$name.json" >"$proof_root/$name.stdout"
done

(cd / && env -u PYTHONPATH "$driver_python" "$support" verify "$proof_root")
for name in cold rebuild; do
    generation=$(cd / && env -u PYTHONPATH "$driver_python" "$support" generation \
        "$proof_root" "$name")
    imported=$(cd / && env -u PYTHONPATH "$generation/venv/bin/python" -I "$import_probe" cpu)
    test "$imported" = installed
done
cuda_generation=$(cd / && env -u PYTHONPATH "$driver_python" "$support" generation \
    "$proof_root" cuda)
cuda_fact=$(cd / && env -u PYTHONPATH "$cuda_generation/venv/bin/python" -I \
    "$import_probe" cuda)
test "$cuda_fact" = "12.9 False"

run_refusal() {
    name=$1
    expected_status=$2
    expected_code=$3
    root_may_exist=$4
    receipt="$proof_root/receipt-$name.json"
    if test "$name" = inside; then
        receipt="$proof_root/environment-inside/receipt.json"
    fi
    set +e
    run_proof "$name" "$receipt" >"$proof_root/$name.stdout" 2>"$proof_root/$name.stderr"
    status=$?
    set -e
    test "$status" -eq "$expected_status"
    grep -q "$expected_code" "$proof_root/$name.stderr"
    (cd / && env -u PYTHONPATH "$driver_python" "$support" verify-refusal \
        "$proof_root" "$name" "$root_may_exist")
}

run_refusal missing 6 wheel_grant_set_mismatch no
run_refusal extra 6 wheel_grant_set_mismatch no
run_refusal corrupt 6 wheel_digest_mismatch no
run_refusal renamed 6 wheel_filename_mismatch no
run_refusal duplicate 6 wheel_member_duplicate no
run_refusal expanded 6 wheel_expansion_cap no
run_refusal forbidden 3 environment_proof_request_invalid no
run_refusal inside 3 environment_proof_output_invalid no
run_refusal reuse 13 environment_root_not_fresh yes

set +e
run_proof race-first "$proof_root/receipt-race.json" \
    >"$proof_root/race-first.stdout" 2>"$proof_root/race-first.stderr" &
first_pid=$!
run_proof race-second "$proof_root/receipt-race.json" \
    >"$proof_root/race-second.stdout" 2>"$proof_root/race-second.stderr" &
second_pid=$!
wait "$first_pid"
first_status=$?
wait "$second_pid"
second_status=$?
set -e
(cd / && env -u PYTHONPATH "$driver_python" "$support" verify-race \
    "$proof_root" "$first_status" "$second_status")
if test "$first_status" -eq 13; then
    grep -q environment_proof_receipt_conflict "$proof_root/race-first.stderr"
else
    grep -q environment_proof_receipt_conflict "$proof_root/race-second.stderr"
fi

(cd / && env -u PYTHONPATH UV_OFFLINE=1 PIP_NO_INDEX=1 \
    HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    ALL_PROXY=http://127.0.0.1:9 \
    "$driver_python" "$private_proof" "$proof_root")

echo "environment proof: installed public CLI produced byte-identical cold/rebuild receipts; exact digest/length, isolated imports, CUDA ABI without a device, atomic receipt race, no-partial-output refusals, and private generation reuse/tamper evidence passed"
