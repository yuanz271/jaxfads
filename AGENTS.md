# JAXFADS Agent Instructions

This file contains repository-specific instructions for agents and contributors.
It is not the source of truth for the public design or API. Use the normative
documents in `docs/` for that:

- `docs/design.md` — architecture and component contracts;
- `docs/algorithm.md` — inference and prediction semantics;
- `docs/training.md` — training and model-transformation behavior;
- `docs/reproducibility.md` — reproducibility and artifact principles; and
- `docs/roadmap.md` — deferred architectural direction.

## Research-code priorities

When implementing or reviewing changes:

1. prioritize mathematical correctness;
2. preserve mathematical consistency across components and inference/training
   contracts;
3. preserve numerical stability and conditioning; and
4. keep the implementation direct, readable, and minimal.

This is research code. Prefer mathematical clarity and algorithmic fluency over
defensive abstraction or speculative compatibility:

- Keep core algorithms pure and direct. Express the mathematical operation at
  its use site instead of adding wrappers or layers without a concrete second
  use.
- Validate inputs at public boundaries and configuration resolution. Inside
  numerical kernels and JAX-transformed code, rely on documented contracts and
  natural Python/JAX failures unless a guard is required for numerical
  stability, data integrity, or a demonstrated failure mode.
- Do not add speculative validation, fallback behavior, compatibility
  scaffolding, or general-purpose extension mechanisms without a concrete
  caller, requirement, or discriminative test.
- Prefer focused tests for mathematical identities, invariants, conditioning,
  numerical stability, and scientific behavior. Test behavior rather than
  implementation details.

## Repository workflow

- Follow the existing source layout and naming conventions.
- Use `uv` for environment and package management; do not use pip, poetry,
  conda, or manually managed virtual environments. Use `uv run` for Python,
  pytest, Ruff, and related tools.
- Add/remove/update dependencies with `uv add`, `uv remove`, or `uv lock`; do
  not edit dependency declarations or lockfiles by hand. Use `uvx` for a
  one-off tool that is not a project dependency.
- During iteration, run tests scoped to the changed code. Run the full suite
  only when explicitly requested or as the final pre-push validation.
- For changes affecting training, inference, serialization, or device behavior,
  include the smallest relevant regression test and update the corresponding
  public documentation in `docs/`.
### Python project rules

- The project uses `pyproject.toml` and `uv.lock`; do not create
  `setup.py`, `setup.cfg`, or `requirements.txt`.
- Tests use pytest, live under `tests/`, and follow `test_*.py` and `test_*`
  naming. A test package does not need `__init__.py`.
- Ruff provides linting, formatting, and import sorting. Use:
  ```bash
  uv run ruff check .
  uv run ruff check --fix .
  uv run ruff format .
  uv run ruff format --check .
  ```
- Use `uv run ty check` or `uv run mypy .` when type checking is configured.
- Follow the configured Ruff style: double quotes, spaces, and 88-character
  formatting; do not add bare `# type: ignore` comments.
- Pre-commit hooks use `prek` when available; install with
  `uvx prek install` or use `uvx pre-commit install`.
- Do not install packages globally or run `python setup.py`.

- Device integration tests are opt-in:
  ```bash
  JAXFADS_RUN_DEVICE_INTEGRATION=1 uv run pytest -q -m integration \
      tests/integration/test_device_parity.py
  ```
- Do not modify or inspect the contents of the untracked `gearax/` directory.
- Before committing, review `git diff`, run `git diff --check`, and ensure no
  unrelated files are included.
