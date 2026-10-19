#!/usr/bin/env bash
# The memory-torture gate (cr-203): every Runtime release, on its own files, through `cozy run`.
# The absolute bar: every cell completes, every output lands, nothing ends poisoned and no idle executor keeps
# `leftover` bytes; each warm cell (24) reports B2's mem.stats(), reads <= 40 GB (H3 5 s: 0 B per denoise step) and
# needs no more OOM retries than its cold pass. The exit status counts the cells and cards that miss it, plus regressions.
#
#   gates/memory-torture.sh <hub> <wheel|-> [--matrix] [--tensorfs-wheel <file|url>] [--baseline <dir>]
#                           [--baseline-wheel <file|url> [--baseline-tensorfs <file|url>]] [--tensorhub <hub>]
#                           [--org <org>] [--image-digest sha256:…] [--after <card>=<cmd>] [card[:cell,…]...]
#
# <wheel> is the candidate cozy-runtime wheel (a file or https://…#sha256=…), `-` for the Hub's target.
# Cards (default: all): 8 12 24 32 2x8; 24 is the RTX 4090, the ComfyUI comparison card. `<card>:<cell>,…` runs only
# those cells and the references (`8:h3ref,h3`; a pair by its first cell, `pair-sdxl`). Each rents the first SKU of its
# row with capacity whose GPUs are no larger than the card, runs every cell, confirms each output landed, ends the
# rental and confirms it is gone. Cards run at once. Results: ./memory-torture-<stamp>/. A card whose SKUs have no
# capacity rents the next larger class (8: 12, 12: 16, 24: 32, 2x8: 2x12) capped by `cozy machine update
# --gpu-memory-limit` to the card's GiB; its lines say `substitute=<class>@<GiB>`.
# --tensorhub: the Hub (overrides <hub>). --org: whose packages run (default fidika). --image-digest: every rental
# boots that worker image, and `machine show` must name it; with `-` as <wheel>, the image is the release under test.
# --after <card>=<cmd>: after the card's cells, `<cmd> <rent-alias> <hub>` runs on its rental; its exit status counts.
# --matrix: the merge matrix for every memory PR (gates/memory-matrix.sh): cards 8 12 24 2x8; Qwen 4 MP, SDXL 4 MP,
# H3 ref2va_turbo 5 s and 15 s, H3 long_form (2 × 5 s; 2 × 15 s at degree 2 on 2x8) and the pair SDXL ∥ Klein; on 24
# each H3 cell runs cold, then warm (`<cell>-warm`, a new prompt and seed), and the pair Qwen 1 MP ∥ LM.
# --baseline <dir>: a 0.27.4 run's results, for relative checks only: a warm cell whose execution time is >10% over
# the baseline's is WORSE (D7). GB read per request and per step are reported beside the baseline's.
# --baseline-wheel: each card first runs its cells on these wheels (0.27.4 + tensorfs 0.10.0) into ./baseline/, then
# switches the same rental to the candidate (H3 downloads once); those lines override --baseline's for their cells.
set -uo pipefail
GATE_RECEIPTS=$(dirname "$(realpath "$0")")/memory_receipts.py
H=$1 W=$2; shift 2
MATRIX="" TFS="" BASE="" CARDS="" BW="" BTFS="" ORG=fidika IMG=""
declare -A AFTER=()
while [ $# -gt 0 ]; do
  case $1 in
    --matrix) MATRIX=1 ;;
    --tensorfs-wheel) TFS=$2; [ -f "$2" ] && TFS=$(realpath "$2"); shift ;;
    --baseline) BASE=$(realpath "$2"); shift ;;
    --baseline-wheel) BW=$2; [ -f "$2" ] && BW=$(realpath "$2"); shift ;;
    --baseline-tensorfs) BTFS=$2; [ -f "$2" ] && BTFS=$(realpath "$2"); shift ;;
    --tensorhub) H=$2; shift ;;
    --org) ORG=$2; shift ;;
    --image-digest) IMG=$2; shift ;;
    --after) AFTER[${2%%=*}]=${2#*=}; shift ;;
    *) CARDS="$CARDS $1" ;;
  esac
  shift
done
[ -f "$W" ] && W=$(realpath "$W")
[ -n "$IMG" ] && [ -n "$BW" ] && { echo "--baseline-wheel would replace the pinned image's Runtime" >&2; exit 2; }
[ -z "$CARDS" ] && { [ -n "$MATRIX" ] && CARDS="8 12 24 2x8" || CARDS="8 12 24 32 2x8"; }
declare -A SKUS=([8]="rtx-3070 rtx-3060-ti" [12]="rtx-4070-super rtx-5070 rtx-3080-ti rtx-4070-ti rtx-3060"
  [16]="rtx-4080-super rtx-4060-ti rtx-5060-ti" [24]="rtx-4090" [32]="rtx-pro-4500 rtx-5090" [2x8]="rtx-3070 rtx-3060-ti"
  [2x12]="rtx-4070-super rtx-3060")
declare -A GIB=([8]=8 [12]=12 [16]=16 [24]=24 [32]=32 [2x8]=8 [2x12]=12)  # no GPU of a card may be larger
declare -A SUB=([8]=12 [12]=16 [24]=32 [2x8]=2x12)  # the next larger class, capped, when a card has no capacity
D=$PWD/memory-torture-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$D"; cd "$D" || exit 1
TAG=$(basename "$D") FAILED=0 M="" FOLLOW="" ONLY="" SUBST=""
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

capped() {  # every GPU of $M is capped at $1 GiB: `machine show`'s limit line, or (any Calcifer) the memory log's ledger
  cozy machine show "$M" --tensorhub "$H" 2>/dev/null | awk -v g="$1" '/^gpu memory limit/ {v = $4} END {exit v + 0 != g}' ||
    cozy machine logs "$M" memory -n 0 --tensorhub "$H" 2>/dev/null | python3 -I -c '
import json, sys
cap = {}
for line in sys.stdin:
    try:
        e = json.loads(line[line.index("{"):])
    except ValueError:
        continue
    if e.get("event") == "ledger":
        cap[e["gpu"]] = e.get("capacity")
sys.exit(not cap or any(c != int(sys.argv[1]) << 30 for c in cap.values()))' "$1"
}

fits() {  # rental $M has $2 GPUs, none over $1 GiB, as `cozy machine show` lists them
  cozy machine show "$M" --tensorhub "$H" 2>/dev/null | sed -nE 's/^GPU [0-9]+ .* ([0-9.]+) GiB .*/\1/p' |
    awk -v most="$1" -v n="$2" '{k++; if ($1 > most) big = 1} END {exit !(k == n && !big)}'
}

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

record() {  # one line per run (status, wall, execution, retries, poisoned, leftover, GB read host to GPU, memory rungs);
            # fails if the run misses the absolute bar, named in `bar=`
  local n=$1 cell=$2 state=$3
  python3 -I "$GATE_RECEIPTS" record "show-$cell.json" "$cell" "$n" "$state" "$SUBST" summary.txt | tee -a summary.txt
}

number() { grep -oE '"number": ?[0-9]+' "run-$1.json" 2>/dev/null | head -1 | grep -oE '[0-9]+$'; }

want() { [ -z "$ONLY" ] || [[ $1 == *-klein-[ab] || ",$ONLY," == *",${1#*-},"* ]]; }  # card:<cell>,… names them

start() {  # start <cell> <cozy run args…>: a detached run (for pairs); prints its number
  local cell=$1; shift
  cozy run "$@" --on "$M" --tensorhub "$H" --out "out-$cell" --detach --json > "run-$cell.json" 2> "run-$cell.err"
  number "$cell"
}

finish() {  # finish <cell> <run number>: follow it to its end, confirm its outputs, record it
  local cell=$1 n=$2
  [ -z "$n" ] && { say "$cell did not start: $(tail -c 300 "run-$cell.err")"; FAILED=$((FAILED + 1)); return; }
  cozy run watch "$n" --tensorhub "$H" > "watch-$cell.log" 2>&1
  record "$n" "$cell" "$(landed "$n" "$cell")" || FAILED=$((FAILED + 1))
}

cell() {  # cell <cell> <cozy run args…>: one run, followed in the foreground until it ends
  local c=$1; shift
  want "$c" || return 0
  cozy run "$@" --on "$M" --tensorhub "$H" --out "out-$c" --json > "run-$c.json" 2> "run-$c.err"
  finish "$c" "$(number "$c")"
}

image() { ls "$PWD/out-$1"/*.png "$PWD/out-$1"/*.webp "$PWD/out-$1"/*.jpg 2>/dev/null | head -1; }

references() {  # two character images for the LM and H3 cells
  local k=$1 r=$2
  cell "$k-klein-a" $ORG/flux-3-klein/generate prompt="portrait of a young lifeguard with long platinum braids, orange swimsuit, $TAG $k a" seed=$((r + 1)) prompt_upsampling=false
  cell "$k-klein-b" $ORG/flux-3-klein/generate prompt="portrait of a woman with short curly auburn hair, navy wool coat, $TAG $k b" seed=$((r + 2)) prompt_upsampling=false
}

cells() {  # the matrix on one machine; $1 = card, $2 = H3 segments
  local k=$1 segs=$2 r=$RANDOM
  references "$k" "$r"
  cell "$k-sdxl" $ORG/sdxl/generate prompt="a brass doorknob on a blue door in morning sun, $TAG $k" seed=$((r + 3))
  cell "$k-qwen1" $ORG/qwen-image-2/generate_image prompt="a lighthouse on a cliff at dusk, $TAG $k" seed=$((r + 4)) megapixels:=1
  cell "$k-qwen4" $ORG/qwen-image-2/generate_image prompt="a market street in the rain at night, $TAG $k" seed=$((r + 5)) megapixels:=4
  local a b; a=$(image "$k-klein-a") b=$(image "$k-klein-b")
  cell "$k-lm" $ORG/character-describer/describe_character --asset "image=$a"
  pair_sdxl "$k" "$r"
  pair_qwen "$k" "$r" "$b"
  h3ref "$k" "$b" "$r"
  h3 "$k-h3" "$segs" "$a" "$b" "$r"
}

pair_sdxl() {  # co-resident pair, two runs on one GPU at once: SDXL ∥ Klein
  local k=$1 r=$2 n1 n2
  want "$k-pair-sdxl" || return 0
  n1=$(start "$k-pair-sdxl" $ORG/sdxl/generate prompt="a red kite over dunes, $TAG $k pair" seed=$((r + 6)))
  n2=$(start "$k-pair-klein" $ORG/flux-3-klein/generate prompt="a teapot on a windowsill, $TAG $k pair" seed=$((r + 7)) prompt_upsampling=false)
  finish "$k-pair-sdxl" "$n1"; finish "$k-pair-klein" "$n2"
}

pair_qwen() {  # co-resident pair: Qwen 1 MP ∥ LM ($3: the LM's image)
  local k=$1 r=$2 n1 n2
  want "$k-pair-qwen" || return 0
  n1=$(start "$k-pair-qwen" $ORG/qwen-image-2/generate_image prompt="a fox in fresh snow, $TAG $k pair" seed=$((r + 8)) megapixels:=1)
  n2=$(start "$k-pair-lm" $ORG/character-describer/describe_character --asset "image=$3")
  finish "$k-pair-qwen" "$n1"; finish "$k-pair-lm" "$n2"
}

h3ref() {  # H3 ref2va_turbo, one clip (15 s unless $4) from one character image; $5 suffixes the cell (-warm)
  local k=$1 a=$2 r=$3 s=${4:-15} c=$1-h3ref
  [ "$s" = 15 ] || c=$k-h3ref$s
  c=$c${5:-}
  printf '{"prompt": "The woman in <Picture 1> walks through a sunlit park in autumn, leaves falling, she picks up a red leaf and smiles, %s %s.\n\nnon_diegetic_music:\nN/A", "seed": %d, "duration_s": %d}\n' "$TAG" "$c" $((r + 40 + s)) "$s" > "$c.json"
  cell "$c" $ORG/minimax-h3/ref2va_turbo --input "$c.json" --asset "$a"
}

matrix() {  # the merge matrix on one machine ($1 = card); on 24 each H3 cell runs again warm
  local k=$1 r=$RANDOM a b
  references "$k" "$r"
  a=$(image "$k-klein-a") b=$(image "$k-klein-b")
  if [ "$k" = 2x8 ]; then h3 "$k-h3" 2 "$a" "$b" "$r" 2; return; fi
  cell "$k-sdxl4" $ORG/sdxl/generate prompt="a brass doorknob on a blue door in morning sun, $TAG $k" seed=$((r + 3)) megapixels:=4
  cell "$k-qwen4" $ORG/qwen-image-2/generate_image prompt="a market street in the rain at night, $TAG $k" seed=$((r + 5)) megapixels:=4
  h3ref "$k" "$b" "$r" 5; [ "$k" = 24 ] && h3ref "$k" "$b" $((r + 100)) 5 -warm
  h3ref "$k" "$b" "$r" 15; [ "$k" = 24 ] && h3ref "$k" "$b" $((r + 100)) 15 -warm
  h3 "$k-h3" 2 "$a" "$b" "$r" "" 5; [ "$k" = 24 ] && h3 "$k-h3-warm" 2 "$a" "$b" $((r + 100)) "" 5
  pair_sdxl "$k" "$r"
  [ "$k" = 24 ] && pair_qwen "$k" "$r" "$b"
}

h3() {  # H3 long_form <cell>: <segs> × <seconds, 15>; the last segment is guided by a video reference (ref2va)
  local c=$1 segs=$2 a=$3 b=$4 r=$5 gpus=${6:-} s=${7:-15} i
  want "$c" || return 0
  ffmpeg -loglevel error -y -loop 1 -i "$a" -t 5 -r 24 -vf "scale=832:480:force_original_aspect_ratio=increase,crop=832:480,zoompan=z='1+0.002*on':d=1:s=832x480" -pix_fmt yuv420p "$D/clip-$c.mp4"
  {
    echo "mode: turbo"; echo "clean_references: false"; echo "describe_references: false"
    echo "references:"
    echo "  - {name: Lena, kind: character, image: $a, description: 'A young lifeguard with long platinum braids.'}"
    echo "  - {name: Mara, kind: character, image: $b, description: 'A woman with short curly auburn hair.'}"
    echo "  - {name: Push, kind: video, video: $D/clip-$c.mp4}"
    echo "segments:"
    for i in $(seq 1 "$segs"); do
      if [ "$i" = "$segs" ]; then
        echo "  - {duration_s: $s, seed: $((r + 20 + i)), prompt: '<Lena> and <Mara> walk the boardwalk as the camera moves like <Push>, $TAG $c $i.'}"
      else
        echo "  - {duration_s: $s, seed: $((r + 20 + i)), prompt: '<Lena> and <Mara> arm-wrestle on a beach boardwalk at golden hour, $TAG $c $i.'}"
      fi
    done
  } > "$c.yaml"
  cell "$c" $ORG/minimax-h3/long_form --input "$c.yaml" ${gpus:+--gpus "$gpus"}
}

card() {  # one card, in its own subshell: rent, install, run its cells, end the rental; exit status = failures
  local card=${1%%:*} gpus="" sku
  ONLY=${1#*:}; [ "$ONLY" = "$1" ] && ONLY=""
  trap release EXIT
  trap 'exit 143' INT TERM  # end the rental and stop: no cell may run after it
  [ "$card" = 2x8 ] && gpus=2
  M=""
  local class
  for class in "$card" ${SUB[$card]:-}; do
    for sku in ${SKUS[$class]}; do
      cozy machine rent "$sku" ${gpus:+--gpus "$gpus"} --disk-gb 160 ${IMG:+--image-digest "$IMG"} --tensorhub "$H" --json > "rent-$card.json" 2> "rent-$card.err"
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
      if [ -n "$M" ]; then  # address it by its rental alias from here on: the Hub reuses names
        local name=$M
        M=$(cozy machine show "$name" --tensorhub "$H" 2>/dev/null | awk '$1 == "alias" {print $2}')
        [ -z "$M" ] && M=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("machine",{}).get("alias",""))' "rent-$card.json" 2>/dev/null)
        [ -z "$M" ] && { say "card $card: rented $sku as $name but found no rent- alias; end it by hand"; FAILED=$((FAILED + 1)); return; }
        say "card $card: rented $sku as $name ($M)"
        if [ -n "$IMG" ] && ! cozy machine show "$M" --tensorhub "$H" 2>/dev/null | awk -v i="$IMG" '$1 == "image" && $2 == i {ok = 1} END {exit !ok}'; then
          say "card $card: $M does not boot $IMG"; FAILED=$((FAILED + 1)); release; return
        fi
        fits "${GIB[$class]}" "${gpus:-1}" && break 2
        say "card $card: $M's GPUs are not ${gpus:-1} of ${GIB[$class]} GiB or less; next SKU"; release
      fi
    done
  done
  [ -z "$M" ] && { say "card $card: no SKU had capacity ($(tail -c 200 "rent-$card.err"))"; FAILED=$((FAILED + 1)); return; }
  [ "$class" = "$card" ] || SUBST=$class@${GIB[$card]}
  if [ -n "$SUBST" ] && ! { cozy machine update "$M" --gpu-memory-limit "${GIB[$card]}GiB" --tensorhub "$H" > "cap-$card.log" 2>&1 &&
      capped "${GIB[$card]}"; }; then
    say "card $card: the ${GIB[$card]} GiB cap on $M is not in force: $(tail -c 300 "cap-$card.log")"; FAILED=$((FAILED + 1)); release; return
  fi
  local log  # the machine's own and memory logs, kept here in case the machine is lost mid-run
  for log in tensord memory; do
    cozy machine logs "$M" "$log" -f -n 0 --tensorhub "$H" > "$log-$card.log" 2>&1 &
    FOLLOW="$FOLLOW $!"
  done
  if [ -n "$BW" ]; then  # the baseline first, in its own directory; its failures are not the candidate's
    mkdir -p baseline
    if update "$BW" "$BTFS" "$card-baseline"; then (cd baseline && TAG=$TAG-baseline && pass "$card")
    else say "card $card: the baseline wheel did not install: $(tail -c 300 "update-$card-baseline.log")"; fi
  fi
  if { [ "$W" != - ] || [ -n "$BW" ]; } && ! update "$W" "$TFS" "$card"; then
    say "card $card: the wheel did not install: $(tail -c 300 "update-$card.log")"; FAILED=$((FAILED + 1)); release; return
  fi
  pass "$card"
  if [ -n "${AFTER[$card]:-}" ]; then
    bash -c "${AFTER[$card]} \"\$@\"" after "$M" "$H" > "after-$card.log" 2>&1
    local rc=$?; say "card $card: after: ${AFTER[$card]} $M $H exited $rc"; FAILED=$((FAILED + rc))
  fi
  release
  return "$FAILED"
}

update() {  # update <runtime wheel|-> <tensorfs wheel> <name>: install the wheels, or (-) the Hub's target, on $M
  local w=$1 t=$2
  [ "$w" = - ] && w="" t=""
  cozy machine update "$M" ${w:+--runtime-wheel "$w"} ${t:+--tensorfs-wheel "$t"} --tensorhub "$H" > "update-$3.log" 2>&1
}

pass() {  # every cell of card $1 on $M, into the current directory
  local card=$1 r=$RANDOM segs=2
  [ "$card" = 32 ] && segs=3
  cozy machine show "$M" --tensorhub "$H" > "machine-$card.txt" 2>&1
  say "card $card${SUBST:+ (substitute $SUBST)}: $(grep -iE '^(runtime|tensorfs|version|image|gpu memory)' "machine-$card.txt" | tr -s ' ' | tr '\n' ' ' | cut -c1-240)"
  if [ -n "$MATRIX" ]; then
    matrix "$card"
  elif [ "$card" = 2x8 ]; then
    references "$card" "$r"
    h3 "$card-h3" "$segs" "$(image "$card-klein-a")" "$(image "$card-klein-b")" "$r" 2
  else
    cells "$card" "$segs"
  fi
}

trap 'kill $PIDS 2>/dev/null' INT TERM
PIDS=""
for c in $CARDS; do (card "$c"; exit "$FAILED") & PIDS="$PIDS $!"; done
for p in $PIDS; do wait "$p"; FAILED=$((FAILED + $?)); done
say "memory-torture: $FAILED below the absolute bar"
if [ -n "$BASE$BW" ]; then  # relative to 0.27.4 (D7): a warm cell >10% slower to execute; GB read reported beside it
  python3 -I "$GATE_RECEIPTS" compare <(cat ${BASE:+"$BASE/summary.txt"} ${BW:+baseline/summary.txt} 2>/dev/null) summary.txt $CARDS | tee -a summary.txt
  FAILED=$((FAILED + $?))
fi
exit $((FAILED > 255 ? 255 : FAILED))
