.PHONY: install dev test lint format clean train ui docker

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist build *.egg-info

train-predator-prey:
	python -m src.cli train --env predator_prey --algo mappo --episodes 500

train-coin-game:
	python -m src.cli train --env coin_game --algo ippo --episodes 500

train-spread:
	python -m src.cli train --env simple_spread --algo mappo --episodes 500

ui:
	streamlit run src/viz/app.py

demo:
	python examples/demo.py

docker-build:
	docker build -t multi-agent-rl .

docker-run:
	docker run -p 8501:8501 multi-agent-rl
