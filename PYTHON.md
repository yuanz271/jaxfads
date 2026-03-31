# Python Project Rules

<!-- Generated from pydevtools.com — the Python Developer Tooling Handbook -->
<!-- Last verified against: uv 0.7, ruff 0.11, ty 0.0.1, pytest 8.x -->
<!-- Full explanations: https://pydevtools.com/handbook/explanation/modern-python-project-setup-guide-for-ai-assistants/ -->

## Package management

This project uses uv. Do not use pip, pip-tools, poetry, or conda.

- Add dependencies: `uv add <package>`
- Add dev dependencies: `uv add --dev <package>`
- Remove dependencies: `uv remove <package>`
- Sync environment: `uv sync`
- Update lockfile: `uv lock`

## Running code

Always use `uv run` to execute Python code and tools. Never call `python`, `pytest`, `ruff`, or other tools directly — they may not resolve to the project's virtual environment.

- Run a script: `uv run python script.py`
- Run a module: `uv run python -m module_name`
- Run a tool: `uv run pytest`, `uv run ruff check .`
- One-off tool (not a project dependency): `uvx <tool>`

## Creating new projects

- Application or script: `uv init project-name`
- Library or distributable package: `uv init --package project-name`
- Always use `pyproject.toml` for metadata (PEP 621). Never create `setup.py`, `setup.cfg`, or `requirements.txt`.

## Testing

- Framework: pytest
- Run tests: `uv run pytest`
- Test files go in `tests/` at the project root
- Test file naming: `test_*.py`
- Test function naming: `test_*`
- No `__init__.py` needed in `tests/`

## Linting and formatting

- Tool: ruff (handles both linting and formatting)
- Lint: `uv run ruff check .`
- Lint and auto-fix: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Check formatting: `uv run ruff format --check .`
- Configuration lives in `pyproject.toml` under `[tool.ruff]`

## Type checking

- Tool: ty (if configured) or mypy
- Run: `uv run ty check` or `uv run mypy .`
- Configuration lives in `pyproject.toml`

## Code style

- Follow ruff's defaults for formatting (88 char line length, double quotes, spaces)
- Import sorting is handled by ruff (`isort` rules enabled via `select = ["I"]`)
- Do not add `# type: ignore` comments without an error code

## Pre-commit hooks

- Tool: prek (preferred) or pre-commit
- Install prek hooks: `uvx prek install`
- Install pre-commit hooks: `uvx pre-commit install`
- Do not install pre-commit or prek with pip. Use `uvx`.

## What NOT to do

- Do not create or activate virtual environments manually. uv manages `.venv/` automatically.
- Do not install packages globally or with `pip install`.
- Do not create `requirements.txt` for dependency management. Use `pyproject.toml` and `uv.lock`.
- Do not run `python setup.py` commands.
- Do not add dependencies to pyproject.toml by hand. Use `uv add`.
