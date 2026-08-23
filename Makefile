# Works from a fresh clone. `make venv` creates .venv here; if you already have an
# interpreter you want to use, override it:  make report PY=/path/to/python
VENV ?= .venv
PY   ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PORT ?= 8078

.PHONY: help venv corpus kyc report report-llm diff gap gate agreement serve test lint clean \
        docker docker-up docker-test docker-sh

help:
	@echo "AgentTrace - coverage and regression analysis for voice agents"
	@echo ""
	@echo "  make venv        create .venv and install dependencies"
	@echo "  make test        run the test suite"
	@echo "  make lint        ruff"
	@echo ""
	@echo "  make report      coverage report, heuristic labeler, no network"
	@echo "  make report-llm  coverage report using Sarvam-105B (needs SARVAM_API_KEY)"
	@echo "  make diff        compare agent v2 -> v3 per cluster, with significance tests"
	@echo "  make gap         generate scenarios for the top uncovered clusters"
	@echo "  make gate        CI gate: exit 1 on a coverage or compliance regression"
	@echo "  make agreement   how far the report depends on the model (Cohen's kappa)"
	@echo "  make kyc         the same pipeline on a second vertical (bank KYC)"
	@echo "  make serve       dashboard on http://127.0.0.1:$(PORT)"
	@echo ""
	@echo "  make corpus      regenerate the synthetic collections corpus (seeded)"
	@echo ""
	@echo "Docker (no local python needed):"
	@echo "  make docker      build the image"
	@echo "  make docker-up   dashboard on http://localhost:$(PORT)"
	@echo "  make docker-test run the test suite in the container"

venv:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -r requirements.txt
	@echo "done -- now run: make test"

corpus:
	$(PY) scripts/gen_corpus.py --domain collections

kyc:
	$(PY) scripts/gen_corpus.py --domain kyc --n 520 --seed 20260823
	$(PY) -m agenttrace.cli report --domain kyc --offline

report:
	$(PY) -m agenttrace.cli report --offline

report-llm:
	$(PY) -m agenttrace.cli report --llm --progress

diff:
	$(PY) -m agenttrace.cli diff --offline

gap:
	$(PY) -m agenttrace.cli close-gap --offline -n 3

gate:
	$(PY) -m agenttrace.cli gate --offline

agreement:
	$(PY) -m agenttrace.cli agreement

serve:
	$(PY) -m uvicorn agenttrace.api:app --port $(PORT) --reload

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

clean:
	rm -rf .cache .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

# ---- Docker ----------------------------------------------------------------
# The path of least resistance on a host where pip wheels or nixpkgs give trouble.

IMAGE ?= agenttrace:local

docker:
	docker build -t $(IMAGE) .

docker-up: docker
	docker run --rm -p $(PORT):8078 $(IMAGE)

docker-test: docker
	docker run --rm $(IMAGE) python -m pytest

docker-sh: docker
	docker run --rm -it $(IMAGE) bash
