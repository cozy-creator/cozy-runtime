#!/usr/bin/env bash
# Advisories in the Runtime's locked dependencies (xs-196): every uv.lock pin (extras and groups), nothing installed, judged
# against OSV by scripts/osv-audit.py (cl-1212). One not listed in advisories-known.txt fails.
set -euo pipefail
cd "$(dirname "$0")/.."
uv export --locked --format requirements-txt --no-hashes --no-emit-project --all-extras --all-groups --quiet \
  | python3 scripts/osv-audit.py advisories-known.txt
