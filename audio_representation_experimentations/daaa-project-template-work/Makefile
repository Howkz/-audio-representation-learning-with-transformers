PYTHON ?= python
CONFIG ?= configs/baseline.yaml

.PHONY: data train test package lint

data:
	PYTHONPATH=. $(PYTHON) scripts/run_data.py --config $(CONFIG)

train:
	PYTHONPATH=. $(PYTHON) scripts/run_train.py --config $(CONFIG)

test:
	PYTHONPATH=. $(PYTHON) scripts/run_test.py --config $(CONFIG)

suite:
	PYTHONPATH=. $(PYTHON) scripts/run_experiment_suite.py --suite-config configs/suite_e00_e11.yaml --verbose

report-template:
	PYTHONPATH=. $(PYTHON) scripts/generate_report_template.py --suite-config configs/suite_e00_e11.yaml

package:
	PYTHONPATH=. $(PYTHON) scripts/package_submission.py --config $(CONFIG)

lint:
	$(PYTHON) -m compileall src scripts
