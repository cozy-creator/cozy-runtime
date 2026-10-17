#!/usr/bin/env bash
# The memory-torture gate (cr-203): every Runtime release, on its own files, through `cozy run`.
# Pass: every run succeeds and every output lands. Slower is fine; a failure is not.
#
#   gates/memory-torture.sh <hub> <wheel|-> [card...]
#
# <wheel> is the candidate cozy-runtime wheel (a file or https://…#sha256=…), `-` for the Hub's target.
# Cards (default: all): 8 12 24 32 2x8. Each rents the first SKU of its row with capacity, runs every cell,
# confirms each output landed, ends the rental and confirms it is gone. Results: ./memory-torture-<stamp>/.
set -uo pipefail
H=$1 W=$2; shift 2
CARDS=${*:-8 12 24 32 2x8}
declare -A SKUS=([8]="rtx-3070" [12]="rtx-5070 rtx-3080-ti rtx-4070-ti" [24]="rtx-4000-ada rtx-pro-4000 l4"
  [32]="rtx-pro-4500 rtx-5090" [2x8]="rtx-3070 rtx-5060-ti rtx-4060-ti rtx-3080-ti l4")
D=$PWD/memory-torture-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$D"; cd "$D" || exit 1
TAG=$(basename "$D") FAILED=0 M="" FOLLOW=""
say() { echo "$(date -u +%T) $*" | tee -a summary.txt; }
release() {  # end the machine this script rented, once its outputs landed, and confirm it is gone
  [ -z "$M" ] && return
  [ -n "$FOLLOW" ] && kill $FOLLOW 2>/dev/null; FOLLOW=""
  cozy machine end "$M" --tensorhub "$H" > "end-$M.log" 2>&1
  cozy machine show "$M" --tensorhub "$H" > "gone-$M.txt" 2>&1
  say "ended $M: $(grep -E '^(state|status)' "gone-$M.txt" | tr -s ' ' | tr '\n' ' ')"
  M=""
}
trap release EXIT INT TERM

landed() {  # every output of run $1 is in out-$2 under its sha256; gives up only when nothing changes for 3 min
  local n=$1 dir=$2 last="" still=0 state
  while :; do
    cozy run show "$n" --tensorhub "$H" --json > "show-$dir.json" 2>/dev/null
    state=$(python3 -I - "show-$dir.json" "out-$dir" <<'PY'
import glob, hashlib, json, sys
try:
    run = json.load(open(sys.argv[1]))
except Exception:
    print("no record"); sys.exit()
def have(o):
    return any("sha256:" + hashlib.sha256(open(f, "rb").read()).hexdigest() == o.get("sha256")
               for f in glob.glob(f"{sys.argv[2]}/*{o['name']}*") if not f.endswith(".part"))
missing = [o["name"] for o in run.get("output") or [] if not have(o)]
print(run.get("status", "?") + (" missing " + ",".join(missing) if missing else " landed"))
PY
)
    case $state in *" landed") echo "$state"; return 0 ;; esac
    [ "$state" = "$last" ] && still=$((still + 1)) || still=0; last=$state
    [ $still -ge 18 ] && { echo "$state"; return 1; }
    sleep 10
  done
}

record() {  # one line per run: status, wall, and each call's memory rungs
  local n=$1 cell=$2 state=$3
  python3 -I - "show-$cell.json" "$cell" "$n" "$state" <<'PY' | tee -a summary.txt
import json, sys
path, cell, n, state = sys.argv[1:]
try:
    run = json.load(open(path))
except Exception:
    run = {}
calls = run.get("calls") or [run]
modes = {k: v for c in calls for k, v in (((c.get("measurements") or {}).get("memory") or {}).get("modes") or {}).items()}
ooms = sum(((c.get("measurements") or {}).get("memory") or {}).get("oom_retries") or 0 for c in calls)
reason = run.get("reason") or ""
if isinstance(reason, dict):
    reason = f"{reason.get('code')}: {reason.get('message')}"
print(f"{cell} run={n} {state} exec={(run.get('execution_ms') or 0) / 1000:.0f}s oom_retries={ooms} "
      f"modes={json.dumps(modes)} reason={str(reason)[:300]}")
PY
  case $state in completed*landed) ;; *) FAILED=$((FAILED + 1)) ;; esac
}

number() { grep -oE '"number": ?[0-9]+' "run-$1.json" 2>/dev/null | head -1 | grep -oE '[0-9]+$'; }

start() {  # start <cell> <cozy run args…>: a detached run (for pairs); prints its number
  local cell=$1; shift
  cozy run "$@" --on "$M" --tensorhub "$H" --out "out-$cell" --detach --json > "run-$cell.json" 2> "run-$cell.err"
  number "$cell"
}

finish() {  # finish <cell> <run number>: follow it to its end, confirm its outputs, record it
  local cell=$1 n=$2
  [ -z "$n" ] && { say "$cell did not start: $(tail -c 300 "run-$cell.err")"; FAILED=$((FAILED + 1)); return; }
  cozy run watch "$n" --tensorhub "$H" > "watch-$cell.log" 2>&1
  record "$n" "$cell" "$(landed "$n" "$cell")"
}

cell() {  # cell <cell> <cozy run args…>: one run, followed in the foreground until it ends
  local c=$1; shift
  cozy run "$@" --on "$M" --tensorhub "$H" --out "out-$c" --json > "run-$c.json" 2> "run-$c.err"
  finish "$c" "$(number "$c")"
}

image() { ls "$D/out-$1"/*.png "$D/out-$1"/*.webp "$D/out-$1"/*.jpg 2>/dev/null | head -1; }

references() {  # two character images for the LM and H3 cells
  local k=$1 r=$2
  cell "$k-klein-a" fidika/flux-3-klein/generate prompt="portrait of a young lifeguard with long platinum braids, orange swimsuit, $TAG $k a" seed=$((r + 1)) prompt_upsampling=false
  cell "$k-klein-b" fidika/flux-3-klein/generate prompt="portrait of a woman with short curly auburn hair, navy wool coat, $TAG $k b" seed=$((r + 2)) prompt_upsampling=false
}

cells() {  # the matrix on one machine; $1 = card, $2 = H3 segments
  local k=$1 segs=$2 r=$RANDOM
  references "$k" "$r"
  cell "$k-sdxl" fidika/sdxl/generate prompt="a brass doorknob on a blue door in morning sun, $TAG $k" seed=$((r + 3))
  cell "$k-qwen1" fidika/qwen-image-2/generate_image prompt="a lighthouse on a cliff at dusk, $TAG $k" seed=$((r + 4)) megapixels:=1
  cell "$k-qwen4" fidika/qwen-image-2/generate_image prompt="a market street in the rain at night, $TAG $k" seed=$((r + 5)) megapixels:=4
  local a b; a=$(image "$k-klein-a") b=$(image "$k-klein-b")
  cell "$k-lm" fidika/character-describer/describe_character --asset "image=$a"
  # co-resident pairs: two runs on one GPU at once
  local n1 n2
  n1=$(start "$k-pair-sdxl" fidika/sdxl/generate prompt="a red kite over dunes, $TAG $k pair" seed=$((r + 6)))
  n2=$(start "$k-pair-klein" fidika/flux-3-klein/generate prompt="a teapot on a windowsill, $TAG $k pair" seed=$((r + 7)) prompt_upsampling=false)
  finish "$k-pair-sdxl" "$n1"; finish "$k-pair-klein" "$n2"
  n1=$(start "$k-pair-qwen" fidika/qwen-image-2/generate_image prompt="a fox in fresh snow, $TAG $k pair" seed=$((r + 8)) megapixels:=1)
  n2=$(start "$k-pair-lm" fidika/character-describer/describe_character --asset "image=$b")
  finish "$k-pair-qwen" "$n1"; finish "$k-pair-lm" "$n2"
  h3 "$k" "$segs" "$a" "$b" "$r"
}

h3() {  # H3 long_form: <segs> × 15 s; the last segment is guided by a video reference (ref2va)
  local k=$1 segs=$2 a=$3 b=$4 r=$5 gpus=${6:-} i
  ffmpeg -loglevel error -y -loop 1 -i "$a" -t 5 -r 24 -vf "scale=832:480:force_original_aspect_ratio=increase,crop=832:480,zoompan=z='1+0.002*on':d=1:s=832x480" -pix_fmt yuv420p "$D/clip-$k.mp4"
  {
    echo "mode: turbo"; echo "clean_references: false"; echo "describe_references: false"
    echo "references:"
    echo "  - {name: Lena, kind: character, image: $a, description: 'A young lifeguard with long platinum braids.'}"
    echo "  - {name: Mara, kind: character, image: $b, description: 'A woman with short curly auburn hair.'}"
    echo "  - {name: Push, kind: video, video: $D/clip-$k.mp4}"
    echo "segments:"
    for i in $(seq 1 "$segs"); do
      if [ "$i" = "$segs" ]; then
        echo "  - {duration_s: 15, seed: $((r + 20 + i)), prompt: '<Lena> and <Mara> walk the boardwalk as the camera moves like <Push>, $TAG $k $i.'}"
      else
        echo "  - {duration_s: 15, seed: $((r + 20 + i)), prompt: '<Lena> and <Mara> arm-wrestle on a beach boardwalk at golden hour, $TAG $k $i.'}"
      fi
    done
  } > "h3-$k.yaml"
  cell "$k-h3" fidika/minimax-h3/long_form --input "h3-$k.yaml" ${gpus:+--gpus "$gpus"}
}

card() {
  local card=$1 gpus="" sku
  [ "$card" = 2x8 ] && gpus=2
  M=""
  for sku in ${SKUS[$card]}; do
    cozy machine rent "$sku" ${gpus:+--gpus "$gpus"} --disk-gb 160 --tensorhub "$H" --json > "rent-$card.json" 2> "rent-$card.err"
    M=$(python3 -I - "rent-$card.json" <<'PY'
import json, sys
text = open(sys.argv[1]).read()
try:
    docs = [json.loads(text)]
except ValueError:
    docs = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
print(next((d["machine"]["name"] for d in reversed(docs) if (d.get("machine") or {}).get("name")), ""))
PY
)
    [ -n "$M" ] && { say "card $card: rented $sku as $M"; break; }
  done
  [ -z "$M" ] && { say "card $card: no SKU had capacity ($(tail -c 200 "rent-$card.err"))"; FAILED=$((FAILED + 1)); return; }
  if [ "$W" != - ] && ! cozy machine update "$M" --runtime-wheel "$W" --tensorhub "$H" > "update-$card.log" 2>&1; then
    say "card $card: the wheel did not install: $(tail -c 300 "update-$card.log")"; FAILED=$((FAILED + 1)); release; return
  fi
  cozy machine show "$M" --tensorhub "$H" > "machine-$card.txt" 2>&1
  local log  # the machine's own and memory logs, kept here in case the machine is lost mid-run
  for log in tensord memory; do
    cozy machine logs "$M" "$log" -f -n 0 --tensorhub "$H" > "$log-$card.log" 2>&1 &
    FOLLOW="$FOLLOW $!"
  done
  say "card $card: $(grep -iE 'runtime|tensord' "machine-$card.txt" | tr -s ' ' | tr '\n' ' ' | cut -c1-200)"
  case $card in
    2x8) local r=$RANDOM; references 2x8 "$r"; h3 2x8 2 "$(image 2x8-klein-a)" "$(image 2x8-klein-b)" "$r" 2 ;;
    32) cells 32 3 ;;
    *) cells "$card" 2 ;;
  esac
  release
}

for c in $CARDS; do card "$c"; done
say "memory-torture: $FAILED failed"
[ "$FAILED" = 0 ]
