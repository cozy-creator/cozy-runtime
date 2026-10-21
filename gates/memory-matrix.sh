#!/usr/bin/env bash
# The memory matrix (design varena-slim §5): the shipping line and the candidate on the same rentals, every request
# through `cozy` in the default home. The bar and the receipts are memory_receipts.py's.
#
#   gates/memory-matrix.sh [--shipping <runtime>,<tensorfs>[,<cozy>]] [--candidate <runtime>,<tensorfs>[,<cozy>]]
#                          [--tensorhub <hub>] [--org <org>] [--on wddm=<machine>] [--walls <summary.txt>]... [--plan]
#                          [card[:group,…]...]
#
# A line is its wheels and, if either line names one, the dev `cozy` binary the machine runs (`--tensord`, a local
# file); wheels are files or https://…#sha256=<hex>. One line runs alone (a shakedown). Two alternate per group: the
# rental's even groups run the shipping line first, odd groups the candidate, so neither always meets a warm page cache.
# Cards (default 24 12 8 6 2x8 2x12): 24, 12, 8 and 6 are GPU memory limits on one rental of the cheapest 24 GB SKU
# with stock; 2x8 and 2x12 on one 2 × 24 GB rental; real8, real12 and h100 rent that card; wddm runs on the machine
# --on names, never rented or ended. Rentals run at once; each ends in a trap and is confirmed gone from the list.
# Groups (`card:group,…` picks): on every card h3ref5 (cold, then warm ×3), h3ref15 (cold), qwen1 qwen2 qwen4 klein
# sdxl1 sdxl4 anima seedvr2 birefnet describer; on 2x8 and 2x12 h3long2 (long_form 2 × 5 s at degree 2); the scenarios
# d9 lora beside pair two cotenant twenty pressure kill on 24, and d9 at degree 2 on 2x12.
# --walls: earlier summaries whose measured walls price the plan, printed before anything is rented; --plan stops there.
# Results: ./memory-matrix-<stamp>/ (summary.txt: one receipt per run; every run's show JSON; machine logs). Exit
# status: the line under test's runs below the absolute bar, its cells worse than the shipping line, and rentals,
# installs or limits that failed.
set -uo pipefail
HERE=$(dirname "$(realpath "$0")")
RECEIPTS=$HERE/memory_receipts.py PORTRAIT=$HERE/portrait.jpg
H=tensorhub ORG=fidika PLAN="" CARDS="" WALLS=()
declare -A LINE=() ON=()
while [ $# -gt 0 ]; do
  case $1 in
    --shipping|--candidate) LINE[${1#--}]=$2; shift ;;
    --tensorhub) H=$2; shift ;;
    --org) ORG=$2; shift ;;
    --on) ON[${2%%=*}]=${2#*=}; shift ;;
    --walls) WALLS+=("$(realpath "$2")"); shift ;;
    --plan) PLAN=1 ;;
    -*) sed -n '4,7s/^# \{0,1\}//p' "$0" >&2; exit 2 ;;
    *) CARDS="$CARDS $1" ;;
  esac
  shift
done
abs() { [ -f "$1" ] && realpath "$1" || echo "$1"; }
version() { basename "${1%%#*}" | cut -d- -f2; }  # a wheel's version, from its file name
LINES=() BINARIES=0
for l in shipping candidate; do
  [ -n "${LINE[$l]:-}" ] || continue
  LINES+=("$l")
  IFS=, read -r r t c <<< "${LINE[$l]}"
  [ -n "$t" ] || { echo "$l: <runtime>,<tensorfs>[,<cozy>]" >&2; exit 2; }
  for f in "$r" "$t"; do [[ $f == https://* ]] || [ -f "$f" ] || { echo "$l: no file $f" >&2; exit 2; }; done
  [ -z "$c" ] || [ -x "$c" ] || { echo "$l: the cozy binary must be a local executable (its version is checked)" >&2; exit 2; }
  [ -n "$c" ] && BINARIES=$((BINARIES + 1))
  LINE[$l]=$(abs "$r"),$(abs "$t")${c:+,$(realpath "$c")}
done
[ ${#LINES[@]} -gt 0 ] || { echo "name a line: --shipping and/or --candidate" >&2; exit 2; }
[ "$BINARIES" = 0 ] || [ "$BINARIES" = ${#LINES[@]} ] || { echo "both lines name a cozy binary, or neither" >&2; exit 2; }
CARDS=${CARDS:-24 12 8 6 2x8 2x12}
PACKAGES="h3ref5 h3ref15 qwen1 qwen2 qwen4 klein sdxl1 sdxl4 anima seedvr2 birefnet describer"
groups() {  # the groups card $1 runs unless named
  case $1 in
    24) echo "$PACKAGES d9 lora beside pair two cotenant twenty pressure kill" ;;
    2x8) echo h3long2 ;;
    2x12) echo h3long2 d9 ;;
    *) echo "$PACKAGES" ;;
  esac
}
chosen() { local spec=$1; [[ $spec == *:* ]] && tr , ' ' <<< "${spec#*:}" || groups "$spec"; }
[[ " $CARDS " == *" wddm"[:\ ]* ]] && [ -z "${ON[wddm]:-}" ] && { echo "wddm runs on --on wddm=<machine>" >&2; exit 2; }

D=$PWD/memory-matrix-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$D"; cd "$D" || exit 1
TAG=$(basename "$D")
say() { echo "$(date -u +%T) $*" | tee -a summary.txt; }
cozy machine rent --json --tensorhub "$H" > catalog.json 2> catalog.err || { cat catalog.err >&2; exit 1; }
SPECS=()
for spec in $CARDS; do
  python3 -I "$RECEIPTS" rental catalog.json "${spec%%:*}" > /dev/null || { echo "no card ${spec%%:*}" >&2; exit 2; }
  SPECS+=("${spec%%:*}=$(chosen "$spec" | tr ' ' ,)")
done
say "lines: $(for l in "${LINES[@]}"; do printf '%s=%s ' "$l" "${LINE[$l]}"; done)"
python3 -I "$RECEIPTS" plan catalog.json "$(IFS=,; echo "${LINES[*]}")" "${WALLS[@]}" -- "${SPECS[@]}" | tee -a summary.txt
[ -n "$PLAN" ] && exit 0
for spec in $CARDS; do
  [ "$(python3 -I "$RECEIPTS" rental catalog.json "${spec%%:*}" | cut -d' ' -f3)" = - ] && continue
  cozy machine update --help 2>/dev/null | grep -q -- --gpu-memory-limit && break
  say "$(cozy --version) has no \`machine update --gpu-memory-limit\` (cozy-creator #1600, 0dfbd20ed): cards $CARDS need it"
  exit 2
done

M="" OWN="" NAME="" FOLLOW="" ON_LINE="" K="" L="" CAP="" ASSET="" FAILED=0 REQ=0
fail() { say "FAIL $*"; FAILED=$((FAILED + 1)); }
seed() { echo $((SRANDOM >> 1)); }  # from the OS: new in every subshell

release() {  # end the rental this run bought (never another), once its outputs landed, and confirm it is gone
  [ -n "$FOLLOW" ] && kill $FOLLOW 2>/dev/null
  FOLLOW=""
  [ -z "$OWN" ] && return
  cozy machine end "$OWN" --tensorhub "$H" > "end-$OWN.log" 2>&1
  cozy machine list --once --tensorhub "$H" > "list-after-$OWN.txt" 2>&1
  if awk -v n="$NAME" '$1 == n {found = 1} END {exit !found}' "list-after-$OWN.txt"; then
    fail "$NAME ($OWN) is still listed after its end: end it by hand ($(grep -w "$NAME" "list-after-$OWN.txt"))"
  else
    say "ended $NAME ($OWN); gone from cozy machine list"
  fi
  OWN=""
}

shown() {  # `cozy machine show $M --json` into $1, then the Python check $2 over it (`m` is the record)
  cozy machine show "$M" --json --tensorhub "$H" > "$1" 2>&1 &&
    python3 -I -c "import json, sys; m = json.load(open(sys.argv[1])); sys.exit(not ($2))" "$1" "${@:3}"
}

rent() {  # rent <gpus> <sku>…: the first SKU with stock; M is its rent- alias
  local gpus=$1 sku; shift
  for sku in "$@"; do
    # disk: every matrix model at once (about 150 GB) plus the environments
    cozy machine rent "$sku" --gpus "$gpus" --disk-gb 250 --tensorhub "$H" --json > "rent-$sku.json" 2> "rent-$sku.err"
    local ordered=$?
    NAME=$(python3 -I - "rent-$sku.json" <<'PY'
import json, sys
text = open(sys.argv[1]).read()
try:
    docs = [json.loads(text)]
except ValueError:  # NDJSON while it follows the boot
    docs = [json.loads(line) for line in text.splitlines() if line.startswith("{") and line.rstrip().endswith("}")]
print(next((d["machine"]["name"] for d in reversed(docs) if (d.get("machine") or {}).get("name")), ""))
PY
)
    if [ -z "$NAME" ]; then  # a refused order tries the next SKU; an accepted one this cannot read stops the card
      [ "$ordered" = 0 ] && { fail "$sku: an order whose machine this cannot read: check \`cozy machine list\` by hand"; return 1; }
      say "$sku × $gpus: $(tail -c 200 "rent-$sku.err" | tr '\n' ' ')"
      continue
    fi
    M=$(cozy machine show "$NAME" --once --tensorhub "$H" 2>/dev/null | awk '$1 == "alias" {print $2}')
    [ -z "$M" ] && { fail "rented $sku as $NAME with no rent- alias: end it by hand"; return 1; }
    OWN=$M
    say "rented $sku × $gpus as $NAME ($M)"
    shown "machine-$M.json" "len(m['status']['gpus']) == int(sys.argv[2])" "$gpus" && return 0
    fail "$M does not show $gpus GPUs"; release
  done
  fail "no SKU of [$*] rented with $gpus GPUs"
  return 1
}

limit() {  # limit <off|<n>GiB|->: the card's GPU memory limit, in force on every GPU as the machine shows it
  [ "$1" = - ] && return 0
  cozy machine update "$M" --gpu-memory-limit "$1" --tensorhub "$H" > "limit-$K.log" 2>&1 ||
    { fail "card $K: limit $1: $(tail -c 300 "limit-$K.log")"; return 1; }
  local bytes=0
  [ "$1" = off ] || bytes=$((${1%GiB} << 30))
  shown "machine-$K.json" "all(g.get('memory_limit_bytes', 0) == int(sys.argv[2]) for g in m['status']['gpus'])" "$bytes" ||
    { fail "card $K: $M does not show a limit of $1 on every GPU"; return 1; }
  say "card $K: limit $1 in force on $M"
}

install() {  # install <line>: its wheels (and binary) on $M, then the versions the machine shows checked against them
  local r t c cozy=""
  IFS=, read -r r t c <<< "${LINE[$1]}"
  [ -n "$c" ] && cozy=$("$c" --version | awk '{print $2}')
  cozy machine update "$M" --runtime-wheel "$r" --tensorfs-wheel "$t" ${c:+--tensord "$c"} --tensorhub "$H" \
    > "install-$1-$K.log" 2>&1 || { fail "install $1 on $M: $(tail -c 300 "install-$1-$K.log")"; ON_LINE=""; return 1; }
  shown "machine-$1.json" "m['status']['runtime'] == sys.argv[2] and m['status']['tensorfs'] == sys.argv[3] and \
sys.argv[4] in ('', m['status']['version'])" "$(version "$r")" "$(version "$t")" "$cozy" ||
    { fail "$M does not show line $1 (runtime $(version "$r"), tensorfs $(version "$t")${cozy:+, cozy $cozy})"; ON_LINE=""; return 1; }
  ON_LINE=$1
}

landed() {  # every output of run $1 in out-$2 under its sha256. The CLI has no landed event: a state unchanged for
  # 3 min (18 reads, 10 s apart) is no progress.
  local n=$1 f=$2 last="" still=0 state
  while :; do
    cozy run show "$n" --tensorhub "$H" --json > "show-$f.json" 2>/dev/null
    state=$(python3 -I - "show-$f.json" "out-$f" <<'PY'
import glob, hashlib, json, sys
try:
    run = json.load(open(sys.argv[1]))
except Exception:
    print("no record"); sys.exit()
def have(o):
    return any("sha256:" + hashlib.sha256(open(f, "rb").read()).hexdigest() == o.get("sha256")
               for f in glob.glob(f"{sys.argv[2]}/*{o['name']}*") if not f.endswith(".part"))
missing = [o["name"] for o in run.get("output") or [] if not have(o)]
print(run.get("status", "?") + (" missing:" + ",".join(missing) if missing else " landed"))
PY
)
    case $state in *" landed") echo "$state"; return 0 ;; esac
    [ "$state" = "$last" ] && still=$((still + 1)) || still=0
    last=$state
    [ $still -ge 18 ] && { echo "$state"; return 1; }
    sleep 10
  done
}

start() {  # start <cell> <cozy run args…>: a detached run on $M; prints its number
  local c=$1; shift
  cozy run "$@" --on "$M" --tensorhub "$H" --out "out-$c.$L" --detach --json > "run-$c.$L.json" 2> "run-$c.$L.err"
  grep -oE '"number": ?[0-9]+' "run-$c.$L.json" | head -1 | grep -oE '[0-9]+$'
}

finish() {  # finish <cell> <number>: follow the run to its end, confirm its outputs and record its receipt
  local c=$1 n=$2 state="not started"
  if [ -n "$n" ]; then
    cozy run watch "$n" --tensorhub "$H" > "watch-$c.$L.log" 2>&1
    state=$(landed "$n" "$c.$L")
  else
    say "$c ($L) did not start: $(tail -c 300 "run-$c.$L.err" | tr '\n' ' ')"
  fi
  python3 -I "$RECEIPTS" record "show-$c.$L.json" "$c" "${n:-0}" "$state" "$L" "$CAP" | tee -a summary.txt > /dev/null
}

cell() { local c=$1; shift; finish "$c" "$(start "$c" "$@")"; }

portrait() {  # ASSET: a portrait no earlier request used, the fixture cropped by this request's own margin
  REQ=$((REQ + 1)) ASSET=$D/portrait-$K-$L-$REQ.jpg
  ffmpeg -loglevel error -y -i "$PORTRAIT" -vf "crop=iw-2*$REQ:ih-2*$REQ" "$ASSET" || fail "no portrait $ASSET"
}

clip() {  # ASSET: a 2 s 832×480 clip no earlier request used, panning over the portrait from its own offset
  REQ=$((REQ + 1)) ASSET=$D/clip-$K-$L-$REQ.mp4
  ffmpeg -loglevel error -y -loop 1 -i "$PORTRAIT" -t 2 -r 24 \
    -vf "crop=640:369:$((REQ % 128)):$((REQ / 128 % 399)),scale=832:480,zoompan=z='1+0.002*on':d=1:s=832x480" \
    -pix_fmt yuv420p "$ASSET" || fail "no clip $ASSET"
}

h3ref() {  # h3ref <cell> <seconds> [extra cozy run args…]: ref2va_turbo from a new portrait, prompt and seed
  local c=$1 s=$2; shift 2
  printf '{"prompt": "The woman in <Picture 1> walks through a sunlit park in autumn, leaves falling, %s.\n\nnon_diegetic_music:\nN/A", "seed": %d, "duration_s": %d, "clean_references": false, "describe_references": false}\n' \
    "$TAG $c $L" "$(seed)" "$s" > "$c.$L.json"
  portrait
  cell "$c" "$ORG/minimax-h3/ref2va_turbo" --input "$c.$L.json" --asset "$ASSET" "$@"
}

h3long() {  # h3long <cell> <segments> <gpus>: long_form of 5 s segments at that degree
  local c=$1 i
  portrait
  {
    echo "mode: turbo"; echo "clean_references: false"; echo "describe_references: false"; echo "references:"
    echo "  - {name: Lena, kind: character, image: $ASSET, description: 'A young lifeguard with long platinum braids.'}"
    echo "segments:"
    for i in $(seq 1 "$2"); do
      echo "  - {duration_s: 5, seed: $(seed), prompt: '<Lena> runs along a beach boardwalk at golden hour, $TAG $c $L $i.'}"
    done
  } > "$c.$L.yaml"
  cell "$c" "$ORG/minimax-h3/long_form" --input "$c.$L.yaml" --gpus "$3"
}

image() {  # image <cell> <org/pkg/fn> [inputs…]: one text-to-image request with a new prompt and seed
  local c=$1 f=$2; shift 2
  cell "$c" "$ORG/$f" prompt="a lighthouse on a cliff at dusk, waves below, $TAG $c $L" seed="$(seed)" "$@"
}

restart() {  # restart <pkg>: its executors end and come back; TensorD and what it holds stay (D9)
  cozy package remove "$ORG/$1" --on "$M" --tensorhub "$H" > "remove-$1-$K.$L.log" 2>&1 &&
    cozy package install "$ORG/$1" --on "$M" --tensorhub "$H" > "install-$1-$K.$L.log" 2>&1 ||
    fail "restart $1 on $M: $(tail -c 300 "remove-$1-$K.$L.log" "install-$1-$K.$L.log" 2>/dev/null | tr '\n' ' ')"
}

upgrade() {  # the Runtime restarted by a `machine update` to the line it already runs; the machine's start must move
  local before
  before=$(cozy machine show "$M" --json --tensorhub "$H" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["status"]["started_at_ms"])')
  install "$L" && shown "machine-upgrade-$K.$L.json" "m['status']['started_at_ms'] != int(sys.argv[2])" "${before:-0}" ||
    fail "card $K: a machine update to the same $L line did not restart it (started_at_ms $before)"
}

# LoRA variants: two turbo LoRA checkpoints of H3, so two fill identities.
LORA_A="model.turbo_lora=fidika/minimax-h3-turbo-lora"
LORA_B="model.turbo_lora=fidika/varena-slim-public-20261020-minimax-h3-turbo-lora"

group() {  # group <name>: one group's runs on line $L, card $K
  local k n a b
  case $1 in
    h3ref5) h3ref "$K-h3ref5" 5; for k in 1 2 3; do h3ref "$K-h3ref5-warm.$k" 5; done ;;
    h3ref15) h3ref "$K-h3ref15" 15 ;;
    qwen1|qwen2|qwen4) image "$K-$1" qwen-image-2/generate_image megapixels:="${1#qwen}" ;;
    klein) image "$K-klein" flux-3-klein/generate ;;
    sdxl1|sdxl4) image "$K-$1" sdxl/generate megapixels:="${1#sdxl}" ;;
    anima) image "$K-anima" anima/generate ;;
    seedvr2) clip; cell "$K-seedvr2" "$ORG/seedvr2/upscale" --asset "video=$ASSET" seed="$(seed)" ;;
    birefnet) portrait; cell "$K-birefnet" "$ORG/birefnet/remove_background" --asset "image=$ASSET" ;;
    describer) portrait; cell "$K-describer" "$ORG/character-describer/describe_character" --asset "image=$ASSET" ;;
    h3long2) h3long "$K-h3long2" 2 2 ;;
    d9)  # warm requests across executor restarts, at the card's degree
      if [ "$K" = 2x12 ]; then
        h3long "$K-d9" 1 2; for k in 1 2 3; do restart minimax-h3; h3long "$K-d9-warm.$k" 1 2; done
      else
        h3ref "$K-d9" 5; for k in 1 2 3; do restart minimax-h3; h3ref "$K-d9-warm.$k" 5; done
      fi ;;
    lora)  # two LoRA variants alternating across Runtime restarts
      h3ref "$K-lora-a.1" 5 "$LORA_A"; upgrade; h3ref "$K-lora-b.1" 5 "$LORA_B"; upgrade
      h3ref "$K-lora-a.2" 5 "$LORA_A"; upgrade; h3ref "$K-lora-b.2" 5 "$LORA_B" ;;
    beside) h3ref "$K-beside-h3" 5; image "$K-beside-qwen4" qwen-image-2/generate_image megapixels:=4 ;;
    pair)  # SDXL ∥ H3
      a=$(start "$K-pair-sdxl4" "$ORG/sdxl/generate" prompt="a red kite over dunes, $TAG $K pair $L" seed="$(seed)" megapixels:=4)
      printf '{"prompt": "The woman in <Picture 1> flies a red kite over dunes, %s.\n\nnon_diegetic_music:\nN/A", "seed": %d, "duration_s": 5, "clean_references": false, "describe_references": false}\n' \
        "$TAG $K pair $L" "$(seed)" > "$K-pair-h3.$L.json"
      portrait
      b=$(start "$K-pair-h3" "$ORG/minimax-h3/ref2va_turbo" --input "$K-pair-h3.$L.json" --asset "$ASSET")
      finish "$K-pair-sdxl4" "$a"; finish "$K-pair-h3" "$b" ;;
    two)  # two requests in one executor
      a=$(start "$K-two-a" "$ORG/sdxl/generate" prompt="a teapot on a windowsill, $TAG $K two a $L" seed="$(seed)")
      b=$(start "$K-two-b" "$ORG/sdxl/generate" prompt="a fox in fresh snow, $TAG $K two b $L" seed="$(seed)")
      finish "$K-two-a" "$a"; finish "$K-two-b" "$b" ;;
    cotenant)  # Klein leaves SDXL's GPU and returns
      image "$K-cotenant-sdxl.1" sdxl/generate; image "$K-cotenant-klein.1" flux-3-klein/generate
      cozy package remove "$ORG/flux-3-klein" --on "$M" --tensorhub "$H" > "leave-$K.$L.log" 2>&1 || fail "klein did not leave $M"
      image "$K-cotenant-sdxl.2" sdxl/generate
      cozy package install "$ORG/flux-3-klein" --on "$M" --tensorhub "$H" > "return-$K.$L.log" 2>&1 || fail "klein did not return to $M"
      image "$K-cotenant-klein.2" flux-3-klein/generate; image "$K-cotenant-sdxl.3" sdxl/generate ;;
    twenty)  # 20 varied requests on one executor: every shape SDXL takes, repeated; all after the first are warm
      local shapes=(1:1 16:9 9:16 4:3 3:4) mps=(1 2 4)
      for k in $(seq 1 20); do
        image "$K-twenty$([ "$k" -gt 1 ] && echo -warm).$k" sdxl/generate aspect_ratio="${shapes[k % 5]}" megapixels:="${mps[k % 3]}"
      done ;;
    pressure)  # H3's deposits and Qwen's peak on one host; the CLI shows no host RAM, so the receipt cannot prove pressure
      h3ref "$K-pressure-h3.1" 5; image "$K-pressure-qwen4" qwen-image-2/generate_image megapixels:=4; h3ref "$K-pressure-h3.2" 5 ;;
    kill)  # the matrix's one labeled deliberate failure (D7): canceled mid-denoise; the next request carries the bar
      printf '{"prompt": "DELIBERATE KILL (D7) %s: the woman in <Picture 1> waves.\n\nnon_diegetic_music:\nN/A", "seed": %d, "duration_s": 5, "clean_references": false, "describe_references": false}\n' \
        "$TAG $K $L" "$(seed)" > "$K-kill-victim.$L.json"
      portrait
      n=$(start "$K-kill-victim" "$ORG/minimax-h3/ref2va_turbo" --input "$K-kill-victim.$L.json" --asset "$ASSET")
      [ -n "$n" ] && cozy run watch "$n" --json --tensorhub "$H" 2>/dev/null | python3 -I -c '
import json, sys
for line in sys.stdin:
    try:
        event = json.loads(line)
    except ValueError:
        continue
    p = event.get("progress") or {}
    if event.get("type") == "run.progress" and 0 < p.get("completed", 0) < p.get("total", 0):
        sys.exit(0)
sys.exit(1)' && cozy run cancel "$n" --tensorhub "$H" > "cancel-$K.$L.log" 2>&1
      finish "$K-kill-victim" "$n"; h3ref "$K-kill-after" 5 ;;
    *) fail "card $K: no group $1" ;;
  esac
}

rental() {  # rental <spec>…: one rental's cards in order, each group on every line, alternating which goes first
  local spec cls gpus lim skus g order first=0
  read -r cls gpus _ skus <<< "$(python3 -I "$RECEIPTS" rental catalog.json "${1%%:*}")"
  trap release EXIT
  trap 'exit 143' INT TERM  # end the rental and stop: no request may follow it
  if [ "$cls" = wddm ]; then M=${ON[wddm]}; NAME=$M; else rent "$gpus" $skus || return "$FAILED"; fi
  K=${1%%:*}
  install "${LINES[0]}" || return "$FAILED"  # before any limit: the line's machine is the one that applies it
  for log in tensord memory; do  # kept here in case the machine is lost mid-run
    cozy machine logs "$M" "$log" -f -n 0 --tensorhub "$H" > "$log-$cls.log" 2>&1 &
    FOLLOW="$FOLLOW $!"
  done
  for spec in "$@"; do
    K=${spec%%:*}
    read -r _ _ lim _ <<< "$(python3 -I "$RECEIPTS" rental catalog.json "$K")"
    CAP=${K#2x}; [ "$lim" = - ] && CAP=card
    limit "$lim" || continue
    for g in $(chosen "$spec"); do
      order=("${LINES[@]}")
      [ $((first % 2)) = 1 ] && order=($(printf '%s\n' "${LINES[@]}" | tac))
      first=$((first + 1))
      for L in "${order[@]}"; do
        [ "$ON_LINE" = "$L" ] || install "$L" || continue
        say "card $K: $g on $L"
        group "$g"
      done
    done
  done
  release
  return "$FAILED"
}

declare -A RENTALS=()
for spec in $CARDS; do
  cls=$(python3 -I "$RECEIPTS" rental catalog.json "${spec%%:*}" | cut -d' ' -f1)
  RENTALS[$cls]="${RENTALS[$cls]:-} $spec"
done
PIDS=""
trap '[ -n "$PIDS" ] && kill $PIDS 2>/dev/null' INT TERM
for cls in "${!RENTALS[@]}"; do (rental ${RENTALS[$cls]}) & PIDS="$PIDS $!"; done
for p in $PIDS; do wait "$p"; FAILED=$((FAILED + $?)); done
python3 -I "$RECEIPTS" compare summary.txt "${LINES[@]}" | tee comparison.txt
FAILED=$((FAILED + ${PIPESTATUS[0]}))
say "memory-matrix: $FAILED failures; receipts in $D/summary.txt, comparison in comparison.txt"
exit $((FAILED > 255 ? 255 : FAILED))
