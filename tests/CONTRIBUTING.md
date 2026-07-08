# Contributing

## Setup

```bash
git clone https://github.com/nseth7/regime-aware-transformer.git
cd regime-aware-transformer
make install   # installs package + dev deps, sets up pre-commit hooks
```

## Development workflow

```bash
make lint        # ruff check
make format       # ruff format
make typecheck    # mypy on src/
make test          # pytest
make coverage      # pytest with coverage report (htmlcov/)
make ci            # lint + typecheck + test, same as CI
```

Pre-commit hooks run ruff lint/format automatically on `git commit`. CI
(`.github/workflows/ci.yml`) runs lint, mypy, and the test suite (Python
3.10 and 3.11) on every push and PR to `main`.

## Before opening a PR

- `make ci` passes locally.
- New behavior has a corresponding test in `tests/`.
- Public functions have type hints and a docstring explaining *why*, not
  just *what* — see existing modules for the expected style.
- If you touch model architecture, data leakage guards, or the backtest
  engine, update `DESIGN.md` accordingly.

## Project structure

```
src/rat/
  models/       # BaselineTransformer, MacroConcatTransformer, RegimeAwareTransformer
  training/     # loss functions, training loop
  evaluation/   # inference, backtest engine, plotting
  config.py     # typed dataclass config for data/model/train/backtest
configs/        # YAML configs consumed via config.py
scripts/        # CLI entry points (prepare_data, train, backtest, compare_models)
tests/          # pytest suite
```

See `DESIGN.md` for architecture rationale.
