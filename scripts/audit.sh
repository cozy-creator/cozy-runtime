#!/usr/bin/env bash
# Advisories in the Runtime's locked dependencies (xs-196): pip-audit over uv.lock's pins, nothing installed. One not listed in
# pip-audit-known.txt fails.
set -euo pipefail
cd "$(dirname "$0")/.."
locked=$(mktemp)
trap 'rm -f "$locked"' EXIT
uv export --locked --format requirements-txt --no-hashes --no-emit-project --all-extras --quiet | grep -E '^[A-Za-z0-9][A-Za-z0-9._-]*==' | sed 's/ *;.*//' > "$locked"
ignore=$(grep -oE '^[A-Z]+-[A-Za-z0-9-]+' pip-audit-known.txt | sed 's/^/--ignore-vuln /' | tr '\n' ' ')
# shellcheck disable=SC2086
uvx pip-audit@2.9.0 --disable-pip --no-deps --progress-spinner off -r "$locked" $ignore
