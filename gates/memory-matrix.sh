#!/usr/bin/env bash
# The memory matrix: runs on real cards before every memory PR merges (cr-203, owner rule 2026-10-18).
#
#   gates/memory-matrix.sh <wheel|-> [--tensorfs-wheel <file>] [--baseline <dir>] [card...]
#
# Cards 8 12 24 2x8, at once, on the cheapest SKUs; cells: H3 ref2va_turbo 5 s and 15 s, H3 long_form, Qwen 4 MP and
# SDXL 4 MP. `-` runs the Hub's target: run it once on 0.27.4 and pass that directory as --baseline afterwards. With a
# baseline, the exit status is the number of cells that completed there and not here.
exec "$(dirname "$(realpath "$0")")/memory-torture.sh" tensorhub "$1" --matrix "${@:2}"
