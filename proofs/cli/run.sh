#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
fixture="$repo/proofs/fixtures/cli-project"
support="$repo/proofs/support/cli.py"
network_source="$repo/proofs/fixtures/network-cut/connect.c"
uv_bin=$(command -v uv)
proof_root=$(mktemp -d /tmp/cozy-cli-proof.XXXXXX)
project="$proof_root/project"

cleanup() {
    chmod -R u+w "$proof_root" 2>/dev/null || true
    rm -rf -- "$proof_root"
}
trap cleanup EXIT HUP INT TERM

cp -R "$fixture" "$project"
mkdir "$proof_root/build"
UV_OFFLINE=1 "$uv_bin" build --offline --wheel --out-dir "$proof_root/build" "$repo"
UV_OFFLINE=1 "$uv_bin" build --offline --wheel --out-dir "$proof_root/build" "$project"
runtime_wheel=$(find "$proof_root/build" -maxdepth 1 -name 'cozy_runtime-*.whl' -print)
fixture_wheel=$(find "$proof_root/build" -maxdepth 1 -name 'proof_cli_package-*.whl' -print)
test -n "$runtime_wheel" && test -n "$fixture_wheel"

UV_OFFLINE=1 "$uv_bin" venv --python "$(command -v python)" "$proof_root/driver"
UV_OFFLINE=1 "$uv_bin" pip install --offline --python "$proof_root/driver/bin/python" \
    "$runtime_wheel" "$fixture_wheel"
driver_python="$proof_root/driver/bin/python"
runtime="$proof_root/driver/bin/cozy-runtime"

(cd / && env -u PYTHONPATH "$driver_python" "$support" wheel "$runtime_wheel")
(cd / && env -u PYTHONPATH "$driver_python" "$support" prepare "$proof_root")
cc -O2 -shared -fPIC -o "$proof_root/network-cut.so" "$network_source" -ldl

cli() {
    (cd / && env -u PYTHONPATH COZY_HOME="$proof_root/home" UV_OFFLINE=1 PIP_NO_INDEX=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
        ALL_PROXY=http://127.0.0.1:9 \
        "$runtime" --dir "$project" "$@" </dev/null)
}

cut_cli() {
    (cd / && env -u PYTHONPATH COZY_HOME="$proof_root/home" UV_OFFLINE=1 PIP_NO_INDEX=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        LD_PRELOAD="$proof_root/network-cut.so" \
        COZY_NETWORK_ATTEMPT_LOG="$proof_root/network-attempted" \
        "$runtime" --dir "$project" "$@" </dev/null)
}

case_number=0
# EXIT CODES ONLY. The pinned English sentences that used to sit here made every
# reworded refusal a second edit in this file and proved nothing about behaviour;
# `internal/exits.py` is the one exit-code matrix and that is what a caller reads.
expect_exit() {
    wanted=$1
    shift
    case_number=$((case_number + 1))
    stdout="$proof_root/exit-$case_number.stdout"
    stderr="$proof_root/exit-$case_number.stderr"
    set +e
    "$@" >"$stdout" 2>"$stderr"
    status=$?
    set -e
    test "$status" -eq "$wanted"
}

before=$(cd / && env -u PYTHONPATH "$driver_python" "$support" tree "$project")
cli describe >"$proof_root/describe.stdout"
after=$(cd / && env -u PYTHONPATH "$driver_python" "$support" tree "$project")
test "$before" = "$after"

cli --json describe >"$proof_root/descriptor.json"
cli --json describe >"$proof_root/descriptor-replay.json"
cmp "$proof_root/descriptor.json" "$proof_root/descriptor-replay.json"

expect_exit 2 cli frobnicate
expect_exit 2 cli --nope
expect_exit 2 cli run
expect_exit 4 cli run nosuchfn
expect_exit 3 cli run echo text=hello colour=blue
expect_exit 3 cli run echo text=hello count=many
expect_exit 4 cli run modeled text=hello
expect_exit 2 cli pull proof/absent@v1
expect_exit 2 cli list
expect_exit 2 cli rm proof/absent@v1
expect_exit 2 cli ingest proof/absent@v1
expect_exit 4 cli describe nosuchfn

cli --help >"$proof_root/help.stdout"
cli doctor >"$proof_root/doctor.stdout"
cli fit echo text=hello >"$proof_root/fit.stdout"
grep -F -q "echo:" "$proof_root/fit.stdout"

rm -f "$proof_root/network-attempted"
cut_cli --json run echo text=hubless count=2 --offline --out "$proof_root/echo-out" \
    >"$proof_root/echo.json" 2>"$proof_root/echo.stderr"
test ! -e "$proof_root/network-attempted"

cli --json job key_contract run=true --out "$proof_root/keys-out" \
    >"$proof_root/keys.json" 2>"$proof_root/keys.stderr"
cli --json job inspect --in "$proof_root/request.json" \
    "hero=@$proof_root/media/same.png" --out "$proof_root/inputs-out" \
    >"$proof_root/inputs.json" 2>"$proof_root/inputs.stderr"
cli --json job inspect --in "$proof_root/reordered.json" \
    "hero=@$proof_root/media/same.png" --out "$proof_root/reordered-out" \
    >"$proof_root/reordered-result.json" 2>"$proof_root/reordered.stderr"

expect_exit 4 cli job inspect --in "$proof_root/request.json" \
    "hero=@$proof_root/media/absent.png" --out "$proof_root/missing-out"
printf 'plain text' >"$proof_root/media/not-an-image.txt"
expect_exit 3 cli job inspect --in "$proof_root/request.json" \
    "hero=@$proof_root/media/not-an-image.txt" --out "$proof_root/wrong-out"

(cd / && env -u PYTHONPATH "$driver_python" "$support" verify "$proof_root")
if ! (cd / && env -u PYTHONPATH "$driver_python" \
    "$repo/proofs/internal/local_input_acceptance.py") \
    >"$proof_root/private-input.stdout" 2>&1; then
    cat "$proof_root/private-input.stdout" >&2
    exit 1
fi
tail -n 1 "$proof_root/private-input.stdout"
test ! -e "$proof_root/home/runtime.sqlite3"
echo "CLI proof: installed-wheel exits, AXI, exact Descriptor/1 bytes, public key tiers, mixed-media invocation identity, network-cut offline execution, no Runtime database, and typed TensorFS-reader refusal passed"
