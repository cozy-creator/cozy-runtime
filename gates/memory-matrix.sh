#!/usr/bin/env bash
# The memory matrix: runs on real cards before every memory PR merges (cr-203, xs-046 D7).
#
#   gates/memory-matrix.sh <wheel|-> [--tensorfs-wheel <file|url>] [--baseline <dir>]
#                          [--baseline-wheel <0.27.4> --baseline-tensorfs <0.10.0>] [--tensorhub <hub>] [--org <org>]
#                          [--image-digest sha256:…] [--after <card>=<cmd>] [card[:cell,…]...]
#
# Cards 8 12 24 (RTX 4090) 2x8, at once; cells: SDXL 4 MP, Qwen 4 MP, H3 ref2va_turbo 5 s and 15 s, H3 long_form and
# the pair SDXL ∥ Klein; on 24 each H3 cell cold then warm, and the pair Qwen 1 MP ∥ LM. Exit status: the cells that
# do not complete, end poisoned or leave leftover bytes, plus (with --baseline outputs/memory-matrix-0274/) the warm
# cells whose execution time is >10% over 0.27.4's. --baseline-wheel runs 0.27.4's cells first on each rental, then the
# candidate's on the same rental, so H3 downloads once. A card out of stock rents the next larger class, capped to its
# GiB (`substitute=` in its lines). Options as in memory-torture.sh.
exec "$(dirname "$(realpath "$0")")/memory-torture.sh" tensorhub "$1" --matrix "${@:2}"
