#!/usr/bin/env bash
# Build immutable public artifacts from private source. This script never publishes.
set -euo pipefail
machine_version= machine_output=
while (($#)); do
  case "$1" in
    --version) machine_version=${2:?version required}; shift 2 ;;
    --output) machine_output=${2:?output required}; shift 2 ;;
    *) printf '%s\n' 'usage: build-release.sh --version VERSION --output DIRECTORY' >&2; exit 2 ;;
  esac
done
[[ $machine_version =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]] || { printf '%s\n' 'version must be the Runtime semantic version' >&2; exit 2; }
[[ -n $machine_output ]] || exit 2
machine_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p -- "$machine_output"
machine_output=$(cd -- "$machine_output" && pwd)
cd -- "$machine_root"
[[ -z $(git status --porcelain -- .) ]] || { printf '%s\n' 'Commit machine-agent changes before building a release' >&2; exit 1; }
machine_commit=$(git rev-parse HEAD)
machine_stamp=github.com/cozy-creator/cozy-runtime/machine-agent/internal/build
machine_manifest=$(jq -n --arg version "$machine_version" --arg commit "$machine_commit" '{name:"cozy-machine",version:$version,source_commit:$commit,platforms:{}}')
for machine_arch in amd64 arm64; do
  machine_name=cozy-machine-linux-$machine_arch
  CGO_ENABLED=0 GOOS=linux GOARCH=$machine_arch GOMAXPROCS=2 go build -p 2 -trimpath \
    -ldflags "-s -w -X $machine_stamp.Version=$machine_version -X $machine_stamp.Commit=$machine_commit" \
    -o "$machine_output/$machine_name" .
  machine_digest=$(sha256sum "$machine_output/$machine_name")
  machine_digest=${machine_digest%% *}
  machine_manifest=$(jq --arg platform "linux-$machine_arch" --arg sha "$machine_digest" \
    '.platforms[$platform]={sha256:$sha}' <<<"$machine_manifest")
done
printf '%s\n' "$machine_manifest" >"$machine_output/machine-agent.json"
printf '%s\n' "$machine_output/machine-agent.json"
