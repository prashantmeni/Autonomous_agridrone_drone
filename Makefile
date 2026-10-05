.PHONY: install test lint run sim
install:
	pip install -e ".[dev]"
test:
	pytest tests/unit -q
lint:
	ruff check src tests
format:
	ruff format src tests
run:
	python -m drone.main --config config/development.yaml
sim:
	python -m drone.main --config config/simulation.yaml
