PYTHON ?= python

.PHONY: run lint test safety check

run:
	$(PYTHON) -m marketing_allocation --project-root .

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m unittest discover -s tests -v

safety:
	$(PYTHON) scripts/check_sensitive.py

check: lint test safety

