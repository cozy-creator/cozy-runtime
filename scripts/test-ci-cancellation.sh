#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
unit="cozy-ci-cancel-${GITHUB_RUN_ID:-$$}-${GITHUB_RUN_ATTEMPT:-1}"
wrapper=''
cleanup() {
  [ -z "$wrapper" ] || kill -TERM "$wrapper" 2>/dev/null || true
  sudo systemctl stop "$unit" >/dev/null 2>&1 || true
}
trap cleanup EXIT
# The child refuses TERM: the unit's stop bound must still remove it and its sleep.
bash "$root/ci-delegated.sh" "$unit" bash -c 'trap "" TERM; sleep 300' &
wrapper=$!
for _ in {1..100}; do
  state=$(systemctl show "$unit" --property=ActiveState --value 2>/dev/null || true)
  [ "$state" != active ] || break
  sleep .1
done
[ "$state" = active ]
child=$(systemctl show "$unit" --property=MainPID --value)
kill -TERM "$wrapper"
set +e
wait "$wrapper"
status=$?
set -e
wrapper=''
[ "$status" = 143 ]
! kill -0 "$child" 2>/dev/null
[ "$(systemctl show "$unit" --property=ActiveState --value 2>/dev/null || true)" != active ]
echo 'Cancelled delegated service and child are gone.'
