# Grouped: a spatial field that persists across years

`Replicated(effect, over=r)` gives `R` **independent** copies of an effect
sharing its hyperparameters — precision `I_R ⊗ Q_E`. `Grouped` replaces that
identity with a real between-group precision, `Q_S ⊗ Q_E`, so the copies borrow
strength from each other. This is R-INLA's `f(index, model=..., group=g,
control.group=list(model=...))`.

Here the copies are years and the effect is a spatial field over 8 regions on a
chain graph. The truth is a spatial pattern that *persists*: each year's field
is `0.9` times the previous year's plus a fresh spatial innovation — exactly
`kron(AR1(0.9), Besag)`. One noisy observation per (region, year), 80 in total.

Two fits on identical data:

```python
Replicated(spatial, over="year")                                # years independent
Grouped(spatial, over="year", structure=AR1Structure(rho=0.9))  # years correlated
```

The independent fit can only smooth *within* a year, so each year's estimate
sees 8 observations. The correlated fit also smooths *across* years, so a
region borrows from its own past and future.

```
PYTHONPATH=src python examples/grouped_panel/run.py
```

Recovering the latent field is about a quarter more accurate, and the
correlated model wins on marginal likelihood — over 8 seeds, in every one:

| seed | independent RMSE | correlated RMSE | gain |
|---|---|---|---|
| 0 | 0.3972 | 0.2998 | 24.5% |
| 1 | 0.4751 | 0.3570 | 24.8% |
| 2 | 0.4968 | 0.3137 | 36.8% |
| 3 | 0.4864 | 0.3630 | 25.4% |
| 4 | 0.3961 | 0.3493 | 11.8% |
| 5 | 0.3986 | 0.2973 | 25.4% |
| 6 | 0.3851 | 0.2673 | 30.6% |
| 7 | 0.4831 | 0.3720 | 23.0% |

Mean gain 25.3%; the correlated model has the higher log marginal likelihood in
8 of 8.

## Why `AR1Structure` and not `SpaceTime`

Knorr-Held's four interaction types pair {iid, structured} with {iid,
structured}, where "structured" means Besag or a random walk. An **AR1** between
groups is outside that family, so this model is not expressible as a
`SpaceTime` interaction. The three structures that *are* — `IIDStructure`,
`RW1Structure`/`RW2Structure` and `BesagStructure` — reproduce all four types,
which is how `Grouped`'s composition is checked
(`tests/test_grouped_spacetime_oracle.py`).

`rho` is fixed here, not estimated: a `Hyperparameter` on a between-group
structure's own parameters is not supported yet. See
[research status](../../docs/research-status.md) for that and the other
limits, including the Sørbye-Rue scaling divergence between a plain `RW1`/`RW2`
and `RW1Structure`/`RW2Structure`.

See [`docs/effects.md`](../../docs/effects.md#grouped) for the full reference.
