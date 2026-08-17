# pyLGM PySpark Adapter Design

**Status:** Approved 2026-08-17

## Purpose

This milestone adds an official PySpark data adapter to pyLGM. It lets the
declarative `LGM` API accept a Spark DataFrame while retaining a single compiler
and inference implementation. It is the next step after the 0.3 general LGM
foundation and directly supports the project's PySpark-first data workflow.

The adapter is a data boundary, not a distributed inference engine. Spark
performs schema checks, relational validation, projection, and canonical
ordering. The compact model table is collected to the driver, converted to the
existing `CanonicalPanel`, and compiled and fitted by the existing sparse
compiler and exact-Gaussian reference engine.

## Scope

### Included

- `LGM.fit(spark_df, engine="exact_gaussian")` for classic PySpark DataFrames.
- Optional PySpark dependency installation.
- Distributed validation of required columns, null keys, and duplicate panel
  keys before driver collection.
- Automatic minimal column projection using Formulaic's
  `Formula(...).required_variables`, the response, panel/time keys, and effect
  indices.
- Deterministic canonical ordering by panel keys then time.
- A bounded, configurable driver-row preflight.
- Spark/Pandas parity fixtures for the compiled model and Gaussian fit.
- Spark prediction keys in canonical order for joining results back to source
  data.

### Excluded

- Distributed factorization, executor-side inference, and Spark ML pipelines.
- A new general-model CLI command or Spark session configuration in YAML.
- Non-Gaussian likelihoods, Laplace, INLA-style integration, spatial effects,
  AR1, or HMC-Laplace.
- A separate Spark compiler or Spark-specific model language.
- Moving forecasting utilities into `pylgm.contrib.forecasting`.

## Public Contract

The public entry point remains the existing model method:

```python
result = model.fit(spark_df, engine="exact_gaussian")
```

PySpark support is installed through the optional package extra:

```bash
python -m pip install "pylgm[spark]"
```

Pandas input retains its current contract: predictive arrays align with the
caller's positional row order. A Spark DataFrame has no stable source-row order,
so Spark predictive arrays are in canonical `(panel..., time)` order. The result
must expose immutable prediction keys for that order, allowing callers to join
predictions to their source table by model keys.

Spark input requires `LGM.time` to be explicitly set. The Pandas-only private
synthetic row key is not available for Spark because it would not define a stable
cross-partition source order.

## Adapter Flow

```text
Spark DataFrame
  -> validate schema and model columns in Spark
  -> validate non-null canonical keys in Spark
  -> validate unique (panel..., time) keys in Spark
  -> select response, canonical keys, formula variables, effect indices
  -> orderBy canonical keys and enforce driver-row limit
  -> collect selected rows only
  -> Pandas DataFrame
  -> CanonicalPanel -> compile_lgm -> exact Gaussian fit
  -> LGMResult with canonical prediction keys
```

The adapter must never call `toPandas()` on the original Spark DataFrame. It may
collect only after projection and canonical ordering. For exact Gaussian fitting,
the resulting driver table must remain small enough for the current dense
reference solver; this adapter does not claim to make that solver distributed.

## Required Columns

The adapter determines its projection deterministically:

- `model.response`;
- every `model.panel` field;
- `model.time`;
- all `Formula(fixed_formula).required_variables` for `Fixed` effects;
- every index field for IID, RW1, and RW2 effects.

The adapter validates that all required columns are present before any collect.
Formula parse failures are normalized to the same compilation-facing error type
as the Pandas compiler.

## Canonical Keys and Prediction Keys

The canonical key sequence is `(*model.panel, model.time)`. Spark validates the
following before collection:

- each canonical key is non-null;
- the key tuple is unique;
- the keys can be used by Spark `orderBy`.

After collection, the adapter constructs the existing `CanonicalPanel`; this is
an intentional second boundary check, not an independent implementation of the
panel contract. The adapter passes the canonical key frame into the result as an
immutable, defensive Pandas value. The keys have one row per predictive array
element and preserve the canonical order exactly.

## Driver-Row Preflight

`LGM.fit()` accepts a Spark-only keyword parameter:

```python
result = model.fit(spark_df, max_driver_rows=100_000)
```

The default is `100_000`. The adapter uses `limit(max_driver_rows + 1).count()`
after projection to reject oversized input before collecting it. `None` disables
this adapter limit but does not bypass the exact-Gaussian latent-dimension and
memory preflight. Values must be positive ordinary integers when supplied.

## Package Boundaries

```text
src/pylgm/
  data/
    panel.py       # existing backend-neutral canonical object
    spark.py       # optional Spark adapter and Spark-only validation
  model.py         # dispatches Pandas versus Spark; exposes keys on result
  inference/
    result.py      # immutable prediction-key metadata
```

The core package never imports PySpark at module import time. `data.spark` uses
lazy imports and raises an installation-facing `DataContractError` only when a
Spark adapter is requested without the extra installed.

## Errors

- Missing projected columns, null keys, duplicate keys, or non-orderable keys:
  `DataContractError` with the model columns named.
- No explicit `LGM.time` for a Spark input: `DataContractError` explaining that
  Spark requires a deterministic time key.
- Driver-row limit exceeded: `DataContractError` with the limit and override.
- Formula parsing/data coercion failures: existing `CompilationError` behavior.
- Expected Spark analysis errors caused by the above validation: normalized to
  the matching typed pyLGM error.
- Unexpected Spark, JVM, memory, or network/runtime failures: propagated
  unchanged rather than misclassified as a data error.

## Testing and Verification

Tests use a local Spark session only when PySpark and Java are available;
otherwise they skip explicitly. The provisioned CI/release environment must run
them unskipped.

The suite verifies:

- optional-import behavior without PySpark;
- required-column projection and no raw-frame collection;
- missing-column, null-key, duplicate-key, no-time, and row-limit errors;
- deterministic canonical keys and immutable result key metadata;
- equality of Spark/Pandas `CanonicalPanel`, compiled labels/design/precision/
  constraints, and exact-Gaussian results for the same logical panel;
- unchanged Pandas caller-order predictive arrays and legacy pipeline behavior.

## Acceptance Criteria

1. `LGM.fit()` accepts a supported Spark DataFrame with explicit time and returns
   the exact-Gaussian result for a small supported panel.
2. The same logical Spark and sorted-Pandas data compile to equivalent model IR
   and numerical result values within existing Gaussian tolerances.
3. No PySpark import or dependency is required for Pandas-only installation and
   tests.
4. Spark predictions can be joined back to the source by immutable canonical
   prediction keys.
5. Oversized data fails before collection unless the user explicitly disables
   the adapter row limit.
