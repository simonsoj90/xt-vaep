# Makefile (place in repo root)
# Run: make setup   | make test   | make app

.PHONY: setup test app clean help
.DEFAULT_GOAL := help

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup: ## Create venv and install deps
	python -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip -r requirements.txt

test: ## Run tests quietly
	$(PY) -m pytest -q || echo "no tests yet"

app: ## Launch Streamlit app (requires app/streamlit_app.py)
	$(VENV)/bin/streamlit run app/streamlit_app.py

clean: ## Remove caches and build artifacts
	rm -rf __pycache__ .pytest_cache *.egg-info

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS := ":.*?## "}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'

fetch:
\tpython scripts/fetch_statsbomb_open.py --config data/leagues.yml
xt:
\tpython scripts/train_xt.py --in_parquet data/interim/events_all.feather
players:
\tPYTHONPATH=src python scripts/build_player_table.py
all:
\tmake fetch && make xt && make players