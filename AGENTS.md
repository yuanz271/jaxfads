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
- Use `uv run` for Python, pytest, Ruff, and related tools; consult `PYTHON.md`
  for Python-specific commands.
- During iteration, run tests scoped to the changed code. Run the full suite
  only when explicitly requested or as the final pre-push validation.
- For changes affecting training, inference, serialization, or device behavior,
  include the smallest relevant regression test and update the corresponding
  public documentation in `docs/`.
- Device integration tests are opt-in:
  ```bash
  JAXFADS_RUN_DEVICE_INTEGRATION=1 uv run pytest -q -m integration \
      tests/integration/test_device_parity.py
  ```
- Do not modify or inspect the contents of the untracked `gearax/` directory.
- Before committing, review `git diff`, run `git diff --check`, and ensure no
  unrelated files are included.
