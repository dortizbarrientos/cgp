PYTHON ?= python3

.PHONY: check figures reproduce

check:
	$(PYTHON) src/validate.py

figures:
	$(PYTHON) src/analyse.py

reproduce: check figures
