#!/usr/bin/env bash
#
# Run the whole study end to end.
#
#   ./run_all.sh              every experiment, in order
#   ./run_all.sh --quick      skip the slowest parts (no SuiteSparse, no web-Stanford)
#
# Output goes to result/ (CSV + this log) and figure/ (PDF for LaTeX, PNG for
# slides).  Experiments 1 and 2 print an explicit PASS/FAIL gate and this
# script stops if either fails, because nothing downstream means anything if
# the implementation or the Google matrix is wrong.

set -uo pipefail
cd "$(dirname "$0")"

PY="./.venv/bin/python -u"          # -u so the log stays live when piped
LOG=result/run.log
QUICK=""
[ "${1:-}" = "--quick" ] && QUICK=1

mkdir -p result figure
: > "$LOG"

run() {
  echo ""                                     | tee -a "$LOG"
  echo "################  $*  ################" | tee -a "$LOG"
  # stderr carries only progress bars and library warnings; keep it out of the log
  $PY "$@" 2>/dev/null | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

gate() {
  run "$@" || { echo "" | tee -a "$LOG"
                echo "GATE FAILED in $1 - stopping." | tee -a "$LOG"; exit 1; }
}

echo "unit tests"                             | tee -a "$LOG"
$PY -m pytest tests/ -q 2>&1 | tail -2        | tee -a "$LOG"

# --- gates: the implementation, then the Google matrix ----------------------
if [ -n "$QUICK" ]; then
  gate experiment/exp1_reproduce.py --quick
else
  gate experiment/exp1_reproduce.py
fi
gate experiment/exp2_pagerank.py

# --- the study --------------------------------------------------------------
run experiment/exp3_damping.py        # the headline: measured vs predicted
run experiment/exp4_ranking.py        # ranking quality vs eigen-residual
run experiment/exp5_spectrum.py       # why it works: the spectrum is real
run experiment/exp6_robustness.py     # starting vector, and wall-clock cost
run experiment/exp7_generality.py     # five undirected graphs, 25 problems

# exp8 loads web-Stanford (2.3M links) and samples its spectrum, which is by far
# the slowest step; skipped by --quick.
if [ -z "$QUICK" ]; then
  run experiment/exp8_limitations.py  # the boundary: a directed graph
else
  echo "" | tee -a "$LOG"
  echo "(--quick: skipping exp8_limitations.py)" | tee -a "$LOG"
fi

echo ""                        | tee -a "$LOG"
echo "ALL EXPERIMENTS COMPLETE" | tee -a "$LOG"
echo "  results -> result/     figures -> figure/" | tee -a "$LOG"
