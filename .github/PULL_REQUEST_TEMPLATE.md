## Description

Please include a summary of the change and the issue it addresses. List any dependencies that are required for this change.

Fixes # (issue)

## Type of Change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Quality Check Checklist

Before submitting this pull request, please verify:

- [ ] My code follows the style guidelines of this project (run `uv run ruff check .` and `uv run black --check .`)
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation (e.g. `docs/ARCHITECTURE.md`)
- [ ] My changes generate no new warnings or type errors (run `uv run mypy src/ tests/`)
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] All 180+ tests pass locally (run `uv run pytest`)
