#!/usr/bin/env bash
# The memory matrix: runs on real cards before every memory PR merges (cr-203, xs-046 D7).
#
#   gates/memory-matrix.sh <wheel|-> [--tensorfs-wheel <file|url>] [--baseline <dir>] [card[:cell,…]...]
#
# Cards 8 12 24 (RTX 4090) 2x8, at once; cells: SDXL 4 MP, Qwen 4 MP, H3 ref2va_turbo 5 s and 15 s, H3 long_form and
# the pair SDXL ∥ Klein; on 24 each H3 cell cold then warm, and the pair Qwen 1 MP ∥ LM. With a baseline
# (outputs/memory-matrix-0274/), the exit status is the number of cells worse than it.
exec "$(dirname "$(realpath "$0")")/memory-torture.sh" tensorhub "$1" --matrix "${@:2}"
