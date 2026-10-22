#!/usr/bin/env bash
# kernel-verdicts.sh <sku> <cap minutes> <native runtime wheel> <tensorfs wheel> <out dir>
# The hand-run offline kernel oracle for one tensor-core family: rents one <sku> card in the
# default ~/.cozy home, runs kernel-verdicts.py on it through `cozy run --on`, and ends the
# rental in the EXIT trap, proving it gone from `cozy machine list` and from RunPod. The rental
# ends at <cap minutes> whatever the run's state: the spend the lead approved. The card's
# shipped files land in <out dir>/*-value/tables/; copy them to
# src/cozy_runtime/internal/qualification_tables/.
set -uo pipefail
SKU=$1 CAP=$2 RUNTIME=$3 TENSORFS=$4 D=$5
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D"
OUT=$D/results.txt
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT"; }
A='' N='' T0=$(date +%s)
state() { cozy machine show "$A" --json 2>/dev/null | jq -r '.state // empty'; }
gone() {
  local l
  l=$(cozy machine list --json 2>/dev/null) || return 1
  jq -e --arg n "$N" --arg a "$A" \
    'all(.machines[]; (.name != $n and .name != $a) or .state == "ended")' <<<"$l" >/dev/null
}
finish() {
  local rc=$1
  [ -n "${WATCH:-}" ] && kill "$WATCH" 2>/dev/null
  [ -n "${RUN:-}" ] && kill "$RUN" 2>/dev/null
  if [ -n "$A" ]; then
    cozy machine show "$A" --json >"$D/show-end.json" 2>&1
    if [ "$(state)" != ended ]; then
      cozy machine end "$A" -y >"$D/end.txt" 2>&1 && say "ended $A ($N)" ||
        say "machine end $A: $(tail -2 "$D/end.txt" | tr '\n' ' ')"
    fi
    until gone; do sleep 15; done
    say "GONE $A ($N): cozy machine list holds it no more (or ended)"
    until runpodctl get pod >"$D/pods-after.txt" 2>&1 && grep -q '^ID' "$D/pods-after.txt" &&
      ! grep -qwE -- "$N|$A" "$D/pods-after.txt"; do sleep 15; done
    say "GONE on RunPod: runpodctl get pod lists no $N"
  fi
  say "RESULT $([ "$rc" = 0 ] && echo PASSED || echo FAILED) elapsed $(($(date +%s) - T0))s"
  exit "$rc"
}
trap 'finish $?' EXIT
trap 'exit 130' INT TERM
fail() { say "FAIL: $*"; exit 1; }
for f in "$RUNTIME" "$TENSORFS" "$HERE/kernel-verdicts.py"; do
  say "input $f sha256 $(sha256sum "$f" | cut -c1-16)"
done
runpodctl get pod >"$D/pods-before.txt" 2>&1
cozy machine rent "$SKU" --gpus 1 --detach --json -y >"$D/rent.json" 2>&1 ||
  { cat "$D/rent.json" >>"$OUT"; fail "rent refused"; }
A=$(jq -r .machine.name "$D/rent.json")
[ -n "$A" ] && [ "$A" != null ] || { A=''; cat "$D/rent.json" >>"$OUT"; fail "rent answered no machine"; }
N=$(cozy machine show "$A" --json 2>/dev/null | jq -r '.name // empty')
N=${N:-$A}
say "RENTED $A ($N, $SKU) quote $(jq -c .quote "$D/rent.json")"
(sleep $((CAP * 60)) && say "CAP: $CAP min reached, ending the rental" && kill -TERM $$) &
WATCH=$!
prev=''
while :; do
  s=$(state)
  [ "$s" != "$prev" ] && say "state $s" && prev=$s
  case $s in ready) break ;; ended | failed) fail "boot ended: $s" ;; esac
  sleep 15
done
cozy machine show "$A" >"$D/show.txt" 2>&1
say "run: cozy run kernel-verdicts.py --on $A --gpus 1"
# In the background so the cap's TERM is handled at once (bash defers traps behind a
# foreground command).
cozy run "$HERE/kernel-verdicts.py" --on "$A" --gpus 1 --asset "runtime=$RUNTIME" \
  --asset "tensorfs=$TENSORFS" --out "$D/out" >"$D/run.txt" 2>&1 &
RUN=$!
wait "$RUN"
rc=$?
RUN=''
L=$(find "$D/out" -name oracle.log | head -1)
[ -n "$L" ] || { tail -30 "$D/run.txt" >>"$OUT"; fail "cozy run exited $rc with no oracle log"; }
grep -E '^\{"table"|Error|Traceback' "$L" | cut -c1-2000 | tee -a "$OUT"
[ $rc = 0 ] || fail "cozy run exited $rc"
exit 0
