.PHONY: install lint format typecheck test coverage ci clean

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src

test:
	pytest -v

coverage:
	pytest --cov=rat --cov-report=term-missing --cov-report=html

ci: lint typecheck test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
