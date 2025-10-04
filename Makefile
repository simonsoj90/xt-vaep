.PHONY: setup install fetch fetch-force xt vaep players app test clean all
VENV:=.venv
PY:=$(VENV)/bin/python
PIP:=$(VENV)/bin/pip

setup:
	python -m venv $(VENV)

install:
	$(PIP) install -U pip -r requirements.txt
	$(PIP) install -e .

fetch:
	$(PY) scripts/fetch_statsbomb_open.py --config data/leagues.yml

fetch-force:
	$(PY) scripts/fetch_statsbomb_open.py --config data/leagues.yml --force

xt:
	$(PY) scripts/train_xt.py --in_parquet data/interim/events_all.feather

vaep:
	$(PY) scripts/train_vaep_lite.py --in_events data/processed/events_with_xt.parquet --k 5

players:
	PYTHONPATH=src $(PY) scripts/build_player_table.py

app:
	$(VENV)/bin/streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info

all:
	make fetch && make xt && make vaep && make players
