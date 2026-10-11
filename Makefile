# Uses .RECIPEPREFIX so recipes do not need literal tab characters.
.RECIPEPREFIX = >
.PHONY: install lint format docstyle test check run clean

VIDEO ?= data/raw/clip.mp4

install:
> pip install -e ".[dev]"

lint:
> ruff check .

format:
> ruff format .
> ruff check --fix .

docstyle:
> pydoclint src tests scripts demo

test:
> pytest

check: lint docstyle test

run:
> python scripts/run_pipeline.py --config configs/walking_skeleton.yaml --set video.path=$(VIDEO)

clean:
> rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
> find . -name "__pycache__" -type d -prune -exec rm -rf {} +
