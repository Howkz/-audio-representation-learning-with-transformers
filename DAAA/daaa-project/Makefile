PYTHON ?= python
CONFIG ?= configs/baseline.yaml

.PHONY: data train test package lint

data:
	PYTHONPATH=. $(PYTHON) scripts/run_data.py --config $(CONFIG)

train:
	PYTHONPATH=. $(PYTHON) scripts/run_train.py --config $(CONFIG)

test:
	PYTHONPATH=. $(PYTHON) scripts/run_test.py --config $(CONFIG)

package:
	PYTHONPATH=. $(PYTHON) scripts/package_submission.py --config $(CONFIG)

lint:
	$(PYTHON) -m compileall src scripts
