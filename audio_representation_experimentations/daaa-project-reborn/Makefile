PYTHON ?= python3
CONFIG ?= configs/baseline.yaml
SUITE_CONFIG ?= configs/final_tp/suite_tp_5seeds.yaml

.PHONY: data train test package lint suite suite-tp suite-tp-pair suite-tp-triple merge-tp-shards compliance report-tp

data:
	PYTHONPATH=. $(PYTHON) scripts/run_data.py --config $(CONFIG)

train:
	PYTHONPATH=. $(PYTHON) scripts/run_train.py --config $(CONFIG)

test:
	PYTHONPATH=. $(PYTHON) scripts/run_test.py --config $(CONFIG)

suite:
	PYTHONPATH=. $(PYTHON) scripts/run_experiment_suite.py --suite-config configs/suite_e00_e11.yaml --verbose

suite-tp:
	PYTHONPATH=. $(PYTHON) scripts/run_experiment_suite.py --suite-config $(SUITE_CONFIG) --verbose

suite-tp-pair:
	PYTHONPATH=. $(PYTHON) scripts/run_tp5_pair.py --verbose

suite-tp-triple:
	PYTHONPATH=. $(PYTHON) scripts/run_tp5_triple.py --verbose

merge-tp-shards:
	PYTHONPATH=. $(PYTHON) scripts/merge_tp_seed_shards.py --suite-config $(SUITE_CONFIG) --primary-root results/experiments --peer-root results_peer/experiments --output-root results/experiments --regenerate-tables

compliance:
	PYTHONPATH=. $(PYTHON) scripts/verify_tp_compliance.py --config $(CONFIG)

report-tp:
	PYTHONPATH=. $(PYTHON) scripts/generate_tp_5seeds_tables.py --suite-config $(SUITE_CONFIG)

report-template:
	PYTHONPATH=. $(PYTHON) scripts/generate_report_template.py --suite-config configs/suite_e00_e11.yaml

package:
	PYTHONPATH=. $(PYTHON) scripts/package_submission.py --config $(CONFIG)

lint:
	$(PYTHON) -m compileall src scripts
