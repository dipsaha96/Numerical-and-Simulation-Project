#!/usr/bin/env bash
# Run every experiment in order, logging clean output to results/run.log.
set -uo pipefail
PY="./.venv/bin/python -u"   # unbuffered: keeps the log live when piped
LOG=results/run.log
mkdir -p results figures
: > "$LOG"

run() {
  echo "" | tee -a "$LOG"
  echo "################ $* ################" | tee -a "$LOG"
  # stderr carries only progress bars and warnings; keep it out of the log.
  $PY "$@" 2>/dev/null | tee -a "$LOG"
}

$PY -m pytest tests/ -q 2>&1 | tail -2 | tee -a "$LOG"
run experiments/exp1_reproduce.py --runs 30
run experiments/exp2_pagerank.py --graphs wiki-Vote cit-HepPh web-Stanford
run experiments/exp3_damping.py --graphs cit-HepPh web-Stanford wiki-Vote
run experiments/exp4_ranking.py --graph cit-HepPh --dampings 0.85 0.99
run experiments/exp5_spectrum.py --graphs cit-HepPh wiki-Vote
run experiments/exp6_symmetric.py
run experiments/exp7_normality.py
echo "" | tee -a "$LOG"
echo "ALL EXPERIMENTS COMPLETE" | tee -a "$LOG"
