PYTHON ?= python

.PHONY: run smoke-input lint test safety check

run:
	$(PYTHON) -m marketing_allocation --project-root local-runs/latest

smoke-input:
	MPLCONFIGDIR=/tmp/matplotlib $(PYTHON) -m marketing_allocation --input-weekly-response data/sample/weekly_response_fixture.csv --budget 1200000 --test-weeks 6 --validation-weeks 5 --project-root /tmp/marketing-allocation-input-smoke

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m unittest discover -s tests -v

safety:
	$(PYTHON) scripts/check_sensitive.py

check: lint test safety
