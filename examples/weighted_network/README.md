# Weighted-network BYM2 (firm exposures)

A `BYM2` spatial effect over a **weighted** graph, where edge weights are
economic coupling strength (interbank exposure / ownership share / supply
dependence) rather than geographic adjacency. Six firms sit in two
tightly-coupled blocks joined by one weak cross-block link; each firm's
estimate borrows strength in proportion to exposure, so strongly-coupled
partners pull together while the weak link barely transmits.

The script fits the weighted graph and the stripped 0/1 version over the same
edges to show the weights actually change the smoothing.

```
PYTHONPATH=src python examples/weighted_network/run.py
```

See [`docs/spatial-effects.md`](../../docs/spatial-effects.md#weighted-neighbour-graphs)
for the weighted-graph input format and its requirements (finite, strictly
positive, symmetric weights).
