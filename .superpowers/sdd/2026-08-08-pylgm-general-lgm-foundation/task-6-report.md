# Task 6 Report: Public Exports, Documentation, and Regression Gate

## Implementation

- `src/pylgm/__init__.py`: exported `RW2`, `Hyperparameter`, `GaussianPrior`,
  and `PCPrecision`; retained every legacy public export; released package
  version `0.3.0`.
- `pyproject.toml`: released project version `0.3.0` and updated its summary.
- `tests/test_package.py`: added the public-contract regression test covering
  the complete new and legacy top-level API, plus package/project version
  agreement.
- `tests/artifacts/test_experiment_artifacts.py`: updated the emitted package
  version expectation from `0.2.0` to `0.3.0`.
- `README.md`: introduced the general LGM API, stated exact Gaussian as the
  only stable 0.3 engine, named the absent PySpark/Laplace/INLA/spatial/HMC
  features accurately, linked the approved architecture, and grouped existing
  forecast commands under **Legacy forecasting utilities**.

## Decisions

The required README lead names Pandas and PySpark adapters. Its following
paragraph explicitly limits 0.3 runtime support to Pandas and marks the
PySpark adapter as a planned follow-up. This avoids presenting unfinished
support as available. Legacy `Pipeline`, `Experiment`, `ComparisonResult`,
`CandidateFailure`, and `FailureCause` remain importable and part of
`pylgm.__all__`.

## TDD evidence

The new package-contract test was written before changing production code:

```text
$ PYTHONPATH=src pytest tests/test_package.py -q
F.  [100%]
FAILED test_general_lgm_api_is_exported_without_removing_legacy_api
AssertionError: assert False
1 failed, 1 passed in 3.02s
```

After implementing exports and the version, the focused check passed:

```text
$ PYTHONPATH=src pytest tests/test_package.py -q
..  [100%]
2 passed
```

The full release gate initially found the environment-artifact fixture still
expecting `pylgm` version `0.2.0`; that expectation was updated and its focused
test passed before the complete gate was repeated.

## Verification

```text
$ PYTHONPATH=src ruff check src tests
All checks passed!

$ PYTHONPATH=src pytest -q
398 passed in 11.86s

$ PYTHONPATH=src pylgm --help
Usage: pylgm [OPTIONS] COMMAND [ARGS]...
Run local pyLGM commands.
Commands: fit, compare
exit 0

$ git diff --check
exit 0
```

## Offline install limitation

The baseline editable-install attempt could not run offline because the build
backend dependency `hatchling` is unavailable and network access is disabled.
Per task direction, it was not retried. The regression, lint, and CLI checks
above were run with `PYTHONPATH=src` instead.

## Self-review

- Public API contains every required new name and preserves the legacy API.
- Documentation confines 0.3 to the exact Gaussian engine and does not claim
  PySpark, Laplace, INLA, spatial, or HMC support.
- Full legacy and new regression coverage is green (398 tests), including
  artifact version reporting.
- `CompiledLGM`, `Gaussian`, and `GaussianResult` ownership was not changed;
  the task only publishes their surrounding public API.
- The README links the approved staged architecture, which retains follow-up
  plans for the PySpark adapter and `contrib.forecasting` relocation.

## Commit

- Implementation: `aeae94c docs: publish general LGM foundation API`
