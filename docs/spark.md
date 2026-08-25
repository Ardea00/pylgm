# Optional PySpark input

`LGM.fit` accepts a Spark DataFrame when the optional extra is installed:

```bash
python -m pip install "pylgm[spark]"
```

Spark is a **data boundary only**: it validates required columns, non-null and
unique `(*panel, time)` keys, projects just the response, canonical keys,
Formulaic fixed-effect variables, and structured-effect indices, orders the rows
canonically, and collects that compact table to the driver. The exact-Gaussian
fit still runs on the driver through the same `CanonicalPanel` compiler and
reference engine as Pandas input; there is no distributed inference and no
separate Spark model language. The original Spark DataFrame is never collected
with `toPandas()`.

Spark input **requires an explicit `LGM.time`** — a Spark DataFrame has no stable
row order, so there is no synthetic row key. For the same reason, Spark
predictive arrays are returned in canonical `(*panel, time)` order (not caller
order), and the result carries immutable `result.prediction_keys` (one row per
prediction) for joining predictions back to the source table:

```python
result = model.fit(spark_df)              # engine="exact_gaussian" by default
keys = result.prediction_keys             # canonical (*panel, time) rows
means = result.predictive_mean            # aligned to keys
```

`model.fit(spark_df, max_driver_rows=100_000)` bounds the driver collection: the
adapter rejects oversized input **before** collecting it. Pass `max_driver_rows=None`
to disable this adapter preflight (the exact-Gaussian dense guards still apply).
Pandas input is unaffected: it keeps caller-row prediction order and no
`prediction_keys`. A runnable example lives at
[`examples/general_lgm/run_spark.py`](https://github.com/Ardea00/pylgm/blob/main/examples/general_lgm/README.md).

### On Databricks

The cluster runtime already ships PySpark, so install **plain `pylgm`** — the
`[spark]` extra would pull a second PySpark into the notebook environment and
can shadow the runtime's:

```python
%pip install pylgm
```

Pass the DataFrame straight to `fit` using the notebook's pre-provided `spark`
session. Inference runs on the **driver**, so the collected table must fit in
driver memory — keep `max_driver_rows` sane and size the driver node
accordingly:

```python
sdf = spark.read.table("catalog.schema.panel")
result = model.fit(sdf, max_driver_rows=1_000_000)
keys = result.prediction_keys            # join predictions back to the source table
```

