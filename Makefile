# Uses .RECIPEPREFIX so recipes do not need literal tab characters.
.RECIPEPREFIX = >
.PHONY: install install-dev lint format docstyle test test-fast coverage check run clean

VIDEO ?= data/raw/clip.mp4

install:
> uv sync --all-extras

install-dev:
> uv sync

lint:
> uv run ruff check .

format:
> uv run ruff format .
> uv run ruff check --fix .

docstyle:
> uv run pydoclint src tests scripts demo

test:
> uv run pytest

test-fast:
> uv run pytest --no-cov

coverage:
> uv run pytest --cov=football_vision --cov-branch --cov-report=term-missing --cov-report=html

check: lint docstyle test

run:
> uv run python scripts/run_pipeline.py --config configs/walking_skeleton.yaml --set video.path=$(VIDEO)

clean:
> rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
> find . -name "__pycache__" -type d -prune -exec rm -rf {} +
