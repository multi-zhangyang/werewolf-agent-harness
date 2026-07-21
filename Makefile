PYTHON ?= python
PYTEST ?= pytest
FRONTEND_DIR ?= frontend

.PHONY: install install-py install-ui dev-api dev-ui build-ui test test-py test-ui test-browser harness-real harness-core-real harness-verify

HARNESS_SEED ?= 100
HARNESS_RUNS ?= 1
HARNESS_POLICIES ?= fixed_round_robin
ARTIFACT_ROOT ?= artifacts
SMOKE_RUN_DIR ?=
CORE_SPEC ?=
CORE_PLUGIN ?=

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

test-browser: build-ui
	NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost WEREWOLF_RUN_BROWSER_E2E=1 PYTHONPATH=. $(PYTHON) -m $(PYTEST) -q tests/test_browser_replay_e2e.py tests/test_browser_live_matrix_e2e.py

harness-real:
	PYTHONPATH=. $(PYTHON) -m src.harness.cli --seed $(HARNESS_SEED) --runs $(HARNESS_RUNS) --turn-policies $(HARNESS_POLICIES) --artifact-root $(ARTIFACT_ROOT)

harness-core-real:
	@test -n "$(CORE_SPEC)" || (echo "CORE_SPEC is required" >&2; exit 2)
	@test -n "$(CORE_PLUGIN)" || (echo "CORE_PLUGIN is required" >&2; exit 2)
	PYTHONPATH=. $(PYTHON) -m src.harness.core_cli --spec "$(CORE_SPEC)" --plugin "$(CORE_PLUGIN)" --artifact-root $(ARTIFACT_ROOT) --verify-smoke

harness-verify:
	@test -n "$(SMOKE_RUN_DIR)" || (echo "SMOKE_RUN_DIR is required" >&2; exit 2)
	PYTHONPATH=. $(PYTHON) -m src.harness.smoke "$(SMOKE_RUN_DIR)"
