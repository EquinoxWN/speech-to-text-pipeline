.PHONY: setup lint test samples model run bench audit ci

PY ?= python

setup:
	$(PY) -m pip install --upgrade pip && $(PY) -m pip install -e ".[dev]"

# ruff, ruff format and mypy --strict.
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check . && $(PY) -m mypy

# Download the 8 FLEURS clips (about 2.5 MB, SHA-256 checked) into .tmp/samples.
samples:
	stt fetch-samples --dest .tmp/samples

# Download the pinned Whisper tiny model (about 75 MB) into HF_HOME.
model:
	stt fetch-model --model tiny

# All tests; the model tests fail instead of skipping when the clips or model are missing.
test: samples model
	STT_REQUIRE_MODEL=1 $(PY) -m pytest -q

# Language, silence removed and error rate on the 8 clips, Whisper tiny on CPU.
run: samples model
	stt run-samples --dest .tmp/samples

bench:
	@echo "M3: per-language WER/DER on full test sets and model-size comparison are not built yet"

# Known vulnerabilities in the installed Python dependencies.
audit:
	$(PY) -m pip_audit --skip-editable --cache-dir .tmp/pip-audit

ci: setup lint test
