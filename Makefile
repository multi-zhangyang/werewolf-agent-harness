PYTHON ?= python
PYTEST ?= pytest
FRONTEND_DIR ?= frontend

.PHONY: install install-py install-ui dev-api dev-ui build-ui test test-py test-ui smoke-real stats-dryrun

install: install-py install-ui

install-py:
	$(PYTHON) -m pip install -r requirements.txt

install-ui:
	cd $(FRONTEND_DIR) && npm install

dev-api:
	$(PYTHON) -m src.api.server

dev-ui:
	cd $(FRONTEND_DIR) && npm run dev

build-ui:
	cd $(FRONTEND_DIR) && npm run build

test: test-py test-ui

test-py:
	PYTHONPATH=. $(PYTHON) -m $(PYTEST) -q

test-ui:
	cd $(FRONTEND_DIR) && npm run build

smoke-real:
	PYTHONPATH=. $(PYTHON) tests/smoke_e2e.py

stats-dryrun:
	PYTHONPATH=. $(PYTHON) tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10
