# CSE 402 - Accelerating PageRank with Dynamic Momentum Power Iteration
#
#   make setup   create .venv and install pinned dependencies
#   make test    run the unit tests
#   make all     run every experiment  -> result/ and figure/
#   make quick   the fast subset: no SuiteSparse downloads, no web-Stanford
#   make clean   remove generated results and figures (keeps downloaded data)

PY := ./.venv/bin/python

.PHONY: setup test all quick clean distclean \
        exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

all:
	./run_all.sh

quick:
	./run_all.sh --quick

# ---- individual experiments ------------------------------------------------
exp1:
	$(PY) experiment/exp1_reproduce.py
exp2:
	$(PY) experiment/exp2_pagerank.py
exp3:
	$(PY) experiment/exp3_damping.py
exp4:
	$(PY) experiment/exp4_ranking.py
exp5:
	$(PY) experiment/exp5_spectrum.py
exp6:
	$(PY) experiment/exp6_robustness.py
exp7:
	$(PY) experiment/exp7_generality.py
exp8:
	$(PY) experiment/exp8_limitations.py

clean:
	rm -f result/*.csv result/*.log figure/*.pdf figure/*.png

distclean: clean
	rm -rf data/*.gz data/*.tar.gz data/*/ .venv
