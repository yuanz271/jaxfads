# Repository Guidelines

## Project Structure & Module Organization
The JAX implementation lives in `src/jaxfads/`, with modules grouped by responsibility: `core.py` collects filtering primitives, `dynamics.py` and `observations.py` encode system models, while `nn.py`, `encoders.py`, and `trainer.py` provide neural blocks and training orchestration. Shared helpers sit in `util.py`. Tests mirror the public surface in `tests/` (for example `test_smoother.py` and `test_trainer.py`). Keep the repository root lean aside from workspace packages declared in `pyproject.toml`; the empty `gearax/` directory exists for editable workspace components referenced by `[tool.uv.sources]`.

## Build, Test, and Development Commands
Use `uv` to manage the environment and run tooling:
- `uv sync` — create or update the virtual environment with runtime and `dev` extras.
- `uv run pytest` — execute the full unit-test suite.
- `uv run python -m pytest tests/test_smoother.py -k "my_case"` — target a specific module or test expression.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and type annotations for public APIs. Docstrings use the NumPy format (section headers such as `Parameters`, `Returns`, `Notes`) shown in `src/jaxfads/core.py`; include shapes and dtypes where helpful. Prefer descriptive snake_case for functions and variables and PascalCase for classes. Keep imports sorted in standard-library, third-party, local order. Run `uv run ruff check .` before submitting; fixable issues can be auto-corrected with `uv run ruff check . --fix`.

## Testing Guidelines
Extend coverage with `uv run pytest` whenever logic under `src/jaxfads/` changes. Add targeted tests in the corresponding `tests/test_<module>.py` file and name cases `test_<behavior>`. Use parametrization for shape or device variants, and isolate randomness with `jax.random.PRNGKey(...)`. For stochastic routines, assert against tolerances (e.g., `jnp.allclose`) rather than exact equality.

## Commit & Pull Request Guidelines
Commits follow short, imperative summaries, as seen in `git log` (for example `fix smoother init`). Group related edits together and include rationale in the body when the change is non-trivial. For pull requests, provide a scope summary, the verification commands you ran, and references to issues or papers when relevant. Attach plots or logs if you alter training dynamics, and call out any new dependencies added to `pyproject.toml`.
