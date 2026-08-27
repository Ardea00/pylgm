# Restricted (Parametric) MIDAS — Design Spec

**Date:** 2026-08-27
**Slice:** S2 of the economic-expansion roadmap (`docs/design/plans/2026-08-26-pylgm-economic-expansion.md`, S2 at lines 92–102).
**Depends on:** S1 U-MIDAS lag-design code (`effects/midas.py`), the `ParametricBlock` seam (`ir/family.py`), the transform framework (`optimization/transforms.py`).

## Goal

Add classic **restricted MIDAS**: collapse `K` high-frequency lag columns into a
single regressor `β · Σ_k w(k; θ) · x_{t,k}`, where the lag-weight shape `w(·; θ)`
is a parametric kernel (exp-Almon or Beta) whose shape parameters `θ` are
estimated by EB / integrated by INLA. The loading `β` is a single coefficient
with a fixed vague Gaussian prior.

This differs from the shipped S1 U-MIDAS, where every lag is its own latent
coefficient tied by a random-walk penalty. Here there is one coefficient and a
θ-shaped weighting.

## Global constraints

- No new runtime dependency (numpy / scipy / pandas / stdlib only).
- Deterministic approximation only (no MCMC).
- Pristine test output: no stray warnings beyond ones explicitly asserted via
  `pytest.warns`; feature tests pass under `-W error::UserWarning`.
- Git identity: **Ardea00 only**; never pass an `--author` override.
- Follow existing effect/compiler/prediction patterns (AR1, ProperCAR, MIDAS).

## The core IR problem

The existing `ParametricBlock` (`ir/family.py`) rebuilds **only the precision**
matrix per θ and carries `block.design` frozen (used by ProperCAR, AR1, BYM2,
and MIDAS-τ). Restricted MIDAS needs the opposite: the single regressor column
`x_agg(t; θ) = Σ_k w(k; θ) · x_{t,k}` is a function of θ, so the **design** must
be rebuilt per θ while the precision (β's fixed vague prior) stays constant.

The new IR concept is therefore the exact mirror of `ParametricBlock`:

| Block | `build(θ)` returns | Frozen |
|---|---|---|
| `ParametricBlock` | precision | design, labels, constraints |
| `ParametricDesignBlock` | design | precision, labels, constraints |

Because `β` has a proper vague prior, the block carries **no constraints** — so
only the design flows through `materialize`.

## Components

### 1. `IdentityTransform` (`optimization/transforms.py`)

exp-Almon's two shape parameters are real-valued (unconstrained). The transform
framework only has `LogTransform` (positive) and `LogitTransform` (bounded
interval). Add a minimal identity transform for real-line parameters:

```python
class IdentityTransform:
    """Unconstrained parameters: u = theta, theta = u."""
    def to_internal(self, theta): return float(theta)
    def from_internal(self, u): return float(u)
    def log_abs_jacobian(self, u): return 0.0
    def contains(self, theta): return bool(np.isfinite(theta))
    def domain_description(self): return "the real line"
    def __eq__(self, other): return isinstance(other, IdentityTransform)
    def __hash__(self): return hash(IdentityTransform)
```

Beta's two parameters (`a, b > 0`) reuse `LogTransform`. Faking the real line
with a wide `LogitTransform` would distort the prior — rejected.

### 2. `ParametricDesignBlock` (`ir/family.py`)

```python
@dataclass(frozen=True)
class ParametricDesignBlock:
    block: LatentBlock                       # template at initial theta
    parameters: tuple[str, ...]              # theta names it consumes
    build: Callable[[Mapping[str, float]], csr_matrix]   # -> new design

    def materialize(self, resolved: Mapping[str, float]) -> LatentBlock:
        design = self.build(resolved)
        if not np.isfinite(design.data).all():
            raise NumericalError(...)
        return LatentBlock(
            self.block.name, self.block.labels, design,
            self.block.precision, self.block.constraints,
        )
```

Wiring:
- `_materialize_blocks`: add `elif isinstance(b, ParametricDesignBlock): out.append(b.materialize(resolved))`.
- `CompiledGaussianFamily` and `CompiledFamily` validators: add
  `ParametricDesignBlock` to the accepted-block isinstance tuples and enforce
  `parameters ⊆ parameter_names` (same check applied to `ParametricBlock`).

### 3. Weight kernels (`effects/midas.py`)

Lags indexed `k = 0 … K-1`, `K = len(columns)`. Convention: `columns[0]` is the
contemporaneous / shallowest lag, `columns[K-1]` the deepest. Weights are
normalized to sum to 1 (β carries the magnitude). Both kernels compute weights
via a log-sum-exp softmax for overflow safety.

**exp-Almon** (params `θ1, θ2` real):
```
log w_k = θ1 · k + θ2 · k²
w = softmax(log w)          # additive-constant invariant; θ2 < 0 => decay
```

**Beta** (params `a, b > 0`):
```
x_k = (k + 1) / (K + 1)              # strictly inside (0, 1), avoids log(0)
log w_k = (a - 1) · log(x_k) + (b - 1) · log(1 - x_k)
w = softmax(log w)
```

Helper: `midas_weights(kernel, K, theta) -> np.ndarray` (length K, sums to 1).

Aggregate design column for a frame with lag matrix `V` of shape `(N, K)`:
`x_agg = V @ w`, shape `(N, 1)`, as a `csr_matrix`. This is what `build(θ)`
returns.

### 4. Builder (`effects/midas.py`)

```python
def build_midas_parametric(
    frame, name, columns, kernel, theta, prior_precision
) -> LatentBlock:
    # validate columns present + finite (reuse S1 validation)
    V = frame[list(columns)].to_numpy(dtype=float)   # (N, K)
    w = midas_weights(kernel, len(columns), theta)     # (K,)
    design = csr_matrix((V @ w).reshape(-1, 1))        # (N, 1)
    precision = csr_matrix([[prior_precision]])        # 1x1, vague
    constraints = np.empty((0, 1), dtype=float)
    return LatentBlock(name, (name,), design, precision, constraints)
```

Used directly for the fixed-θ path.

### 5. Effect spec (`effects/spec.py`)

```python
@dataclass(frozen=True)
class MIDASParametric(_ComposableEffect):
    """Restricted MIDAS: HF lag columns collapsed into one regressor
    beta * sum_k w(k; theta) * x_{t,k}, with a parametric lag-weight kernel
    (exp-Almon or Beta) whose shape parameters are estimated/integrated."""
    name: str
    columns: tuple[str, ...]
    kernel: str = "beta"                                  # "beta" | "exp_almon"
    shape1: float | Hyperparameter | None = None          # None -> kernel default
    shape2: float | Hyperparameter | None = None          # None -> kernel default
    prior_precision: float = 1e-6                         # mirrors Fixed
```

`__post_init__` validation (defaults resolved here because they are
kernel-conditional and a frozen field cannot express that):
- `name` non-empty; `columns` non-empty tuple of non-empty strings; at least 2
  columns (a kernel over <2 lags is degenerate).
- `kernel ∈ {"beta", "exp_almon"}`.
- `prior_precision` positive real (reuse `_positive_real`).
- If `shape1`/`shape2` is `None`, fill the kernel-appropriate default (below);
  otherwise each must be a float or `Hyperparameter`.

Kernel-appropriate default shape values (weak, sane starting shapes):
- Beta: `a = b = 2.0` (symmetric interior hump).
- exp-Almon: `θ1 = 0.0`, `θ2 = -0.1` (gentle monotone decay).

When a shape is supplied as a plain float, that θ is fixed (design bakes in).
When supplied as a `Hyperparameter`, it is estimated (EB) / integrated (INLA),
with the transform chosen by kernel (Beta → `LogTransform`, exp-Almon →
`IdentityTransform`). Register the effect in the `_ComposableEffect` `__add__`
and any `__post_init__` isinstance tuples alongside the other effects, and
export `MIDASParametric` from `pylgm/__init__.py`.

### 6. Compiler wiring (`compiler.py`)

**`compile_family` (θ estimated)** — mirror the AR1/ProperCAR pattern:
- Build a `template` LatentBlock via `build_midas_parametric` at the initial θ.
- Define `build(values) -> csr_matrix` that reads the resolved shape params,
  computes `midas_weights`, and returns the `(N,1)` aggregated design.
- Append `ParametricDesignBlock(template, theta_names, build)`.
- Register each estimated shape param's bounds/prior:
  - Beta params via `_log_bounds` (positive).
  - exp-Almon params via a new `_real_bounds` helper backed by
    `IdentityTransform`.
  - Bind priors exactly as `rho` / `phi` do (`prior.bind(...)` when present).
- The loading β is not a hyperparameter — its precision is the fixed
  `prior_precision` baked into the template.

**Fixed-θ path** (`compile_family` non-optimized branch and `compile_lgm`):
`build_midas_parametric` at the given floats → wrap as
`ScalableBlock(block, None, 1.0)` (no per-θ rebuild needed).

**Why no new inference code:** `optimize` evaluates the marginal-likelihood
objective on a fresh `CompiledLGM` per θ; `integrate` does it per grid node.
A θ-dependent design is just a different linear model at each θ — the Gaussian
marginal-likelihood machinery already recomputes everything from the compiled
model (the same way ProperCAR's ρ enters the precision nonlinearly). No analytic
gradient is assumed.

**Prediction design-entry:** emit `("midas_parametric", (name, columns, kernel,
theta_hat))` where `theta_hat` is the fitted/plug-in shape tuple.

### 7. Prediction (`inference/prediction.py`)

```python
def _midas_parametric_block(entry, new_data):
    name, columns, kernel, theta = entry
    # validate columns present in new_data (reuse _midas_block's missing-cols check)
    V = new_data[list(columns)].to_numpy(dtype=float)
    w = midas_weights(kernel, len(columns), theta)
    return (V @ w).reshape(-1, 1)
```

Dispatch: add `elif kind == "midas_parametric": blocks.append(_midas_parametric_block(payload, new_data))`.

## Data flow

```
MIDASParametric(kernel, shape1, shape2)             # effect spec
      │  compile_family
      ▼
template LatentBlock (θ_init) + build(θ) closure
      │
ParametricDesignBlock(template, θ_names, build)     # IR seam
      │  optimize / integrate: materialize(θ) per θ
      ▼
CompiledLGM(design = x_agg(θ), precision = [[prior_precision]])
      │  fit
      ▼
θ̂, β̂  ──►  predict(): _midas_parametric_block rebuilds x_agg(θ̂) for new rows
```

## Testing

New test files under `tests/`:

1. **Kernels** (`test_midas_parametric_kernels.py`): weights sum to 1 for both
   kernels; exp-Almon decays when θ2 < 0; Beta produces an interior hump;
   overflow-safe at extreme θ (e.g. θ1 large positive) — no `inf`/`nan`.
2. **Transform** (extend `test_transforms.py` or new): `IdentityTransform`
   round-trip (`from_internal(to_internal(x)) == x`), zero log-jacobian,
   `contains` rejects non-finite.
3. **Recovery** (`test_midas_parametric_effect.py`): simulate
   `y = β·Σ w_k(θ_true) x_k + noise`; fit; assert θ̂ near θ_true (loose bound),
   fitted-vs-truth correlation above a sanity floor, β̂ finite/positive as
   simulated.
4. **Contract**: the block is proper and unconstrained → accepted under
   `latent_strategy="laplace"` (unlike RW2/Besag/SpaceTime II–IV). Assert a fit
   under full Laplace is *not* rejected.
5. **Prediction**: `predict` on held-out rows aggregates correctly at θ̂
   (compare against a hand-computed `V @ w(θ̂)`).

All feature tests must pass under `-W error::UserWarning`.

## Files

| File | Change |
|---|---|
| `src/pylgm/optimization/transforms.py` | add `IdentityTransform` |
| `src/pylgm/ir/family.py` | add `ParametricDesignBlock`; `_materialize_blocks` branch; both validator tuples |
| `src/pylgm/effects/midas.py` | `midas_weights` kernels + `build_midas_parametric` |
| `src/pylgm/effects/spec.py` | `MIDASParametric` dataclass + composition/validation |
| `src/pylgm/__init__.py` | export `MIDASParametric` |
| `src/pylgm/compiler.py` | compile_family (θ-estimated) + fixed-θ branch + `_real_bounds` helper + prediction design-entry |
| `src/pylgm/inference/prediction.py` | `_midas_parametric_block` + dispatch |
| `docs/effects.md`, `docs/roadmap.md` | document the effect; update shipped list |
| `tests/test_midas_parametric_kernels.py`, `tests/test_midas_parametric_effect.py` (+ transform test) | kernels, recovery, contract, prediction |

## Out of scope (YAGNI)

- Multi-covariate parametric-MIDAS blocks (compose multiple effects instead).
- Config/YAML frontend for the effect (Python API only this slice).
- Estimated precision on β (fixed vague prior only).
- Kernels beyond exp-Almon and Beta.
