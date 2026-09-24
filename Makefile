# CSE 402 -- Accelerating PageRank with Dynamic Momentum Power Iteration
#
#   make setup   create .venv and install pinned dependencies
#   make test    run the unit tests
#   make all     run every experiment, regenerating results/ and figures/
#   make quick   the offline/synthetic subset, no downloads needed
#   make guide   rebuild PROJECT_GUIDE.pdf, the beginner's walkthrough
#   make clean   remove generated results and figures (keeps downloaded data)

PY := ./.venv/bin/python

.PHONY: setup test all quick clean phase1 phase2 phase3 phase4 phase5 phase6 phase7 guide distclean

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

# Gates run first: a failure here invalidates everything downstream.
phase1:
	$(PY) experiments/exp1_reproduce.py

phase2:
	$(PY) experiments/exp2_pagerank.py

phase3:
	$(PY) experiments/exp3_damping.py

phase4:
	$(PY) experiments/exp4_ranking.py

phase5:
	$(PY) experiments/exp5_spectrum.py

phase6:
	$(PY) experiments/exp6_symmetric.py

phase7:
	$(PY) experiments/exp7_normality.py

# The beginner's guide PDF.  Needs reportlab; figures must exist first.
guide:
	$(PY) -m pip install --quiet reportlab pillow
	$(PY) docs/make_guide.py

all: test phase1 phase2 phase3 phase4 phase5 phase6 phase7
	@echo
	@echo "All experiments complete. See results/ and figures/."

quick:
	$(PY) experiments/exp1_reproduce.py --quick
	$(PY) experiments/exp2_pagerank.py --offline --graphs synthetic
	$(PY) experiments/exp3_damping.py --offline --graphs synthetic
	$(PY) experiments/exp6_symmetric.py --offline
	$(PY) experiments/exp7_normality.py

clean:
	rm -f results/*.csv figures/*.pdf figures/*.png

distclean: clean
	rm -rf data/*.gz data/*.tar.gz data/*/ .venv
