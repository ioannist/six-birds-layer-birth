.PHONY: install test smoke dry-run upstream-smoke paper

install:
	python -m pip install -e .[test]

test:
	pytest

smoke:
	python -m layerbirth.smoke

dry-run:
	python -m layerbirth.manifest dry-run --experiment-id contract_smoke --output-root artifacts/dryrun

upstream-smoke:
	python scripts/run_upstream_smoke.py

paper:
	python3 scripts/build_paper.py

all: install test smoke dry-run upstream-smoke
