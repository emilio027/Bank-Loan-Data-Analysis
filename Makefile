.PHONY: setup data app train evaluate report lint test

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

data:
	python scripts/make_demo_data.py

app:
	python - <<'PY'
print("No app defined for this repo.")
PY

train:
	python - <<'PY'
print("Training placeholder. Add model training here.")
PY

evaluate:
	pytest -q

report:
	python scripts/build_report.py

lint:
	ruff check .

test:
	pytest -q
