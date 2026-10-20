#!/usr/bin/env bash
# Run a delegated CI service, including its children, for exactly this shell's lifetime.
set -euo pipefail
unit=$1
shift
cleanup() {
  # Normal completion may already have collected the transient unit.
  sudo systemctl stop "$unit" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Background + wait lets Bash act on a signal immediately, rather than waiting for
# a foreground systemd-run whose service lives outside the Actions process tree.
sudo systemd-run --wait --pipe --collect --unit="$unit" \
  --property=TimeoutStopSec=5s --property=KillMode=control-group "$@" &
wait "$!"
