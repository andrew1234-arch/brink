PY ?= python

.PHONY: help install install-all lint format test app demo live space run-all dry-run

help:            ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:         ## core + dashboard + dev tools
	$(PY) -m pip install -e ".[judge,dashboard,dev]"

install-all:     ## everything, including the Lexsi toolkits (SafeTune needs Python >= 3.12)
	$(PY) -m pip install -e ".[judge,dashboard,dev,curatorkit,auditkit,safetune]"
	$(PY) -m pip install --no-deps aligntune
	$(PY) -m pip install "trl==1.7.1" pypdf python-docx

lint:            ## ruff lint + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:          ## apply ruff fixes and formatting
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

test:            ## offline CPU test suite (tiny random models, no downloads)
	HF_HUB_OFFLINE=1 $(PY) -m pytest -q

app:             ## the app: Run Brink live, Gate your data, Experiments (results/)
	$(PY) app/app.py

live:            ## Brink end to end, headless, recorded to results/live (Tiny Aya needs a GPU)
	$(PY) experiments/live.py

demo:            ## dashboard on clearly labelled simulated runs
	$(PY) app/app.py --demo

space:           ## build the Hugging Face Space folder in build/space
	$(PY) scripts/build_space.py --out build/space

dry-run:         ## what run-all would do
	$(PY) experiments/run_all.py --dry-run

run-all:         ## every experiment not finished yet (needs GPUs; ~12 h on a T4 x2)
	$(PY) experiments/run_all.py
