# Joint Models via Multi-Likelihood Stacking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let pyLGM fit joint models — several responses, each with its own likelihood, sharing latent fields through a possibly-estimated scaling — by stacking them into one ordinary `CompiledLGM`.

**Architecture:** Responses stack as `y = (y^(1), ..., y^(K))`, sub-model `k` occupying rows `R_k`. Sub-model-private latent blocks are zero-padded outside their slice; shared blocks get scaled rows in every slice they enter. Both `CompiledLGM` invariants (`design == hstack(blocks)`, `precision == block_diag(blocks)`) are preserved, so the IR does not change. The likelihood becomes a row-dispatching `CompiledMixture`, which the Laplace engine consumes through its existing row-separable interface.

**Tech Stack:** Python 3.11+, numpy 2.x, scipy 1.14+ (sparse), pandas 2.2+, pytest 8.3+. No new dependencies.

**Spec:** `docs/design/specs/2026-09-01-pylgm-joint-multi-likelihood-design.md`

## Global Constraints

- **No new runtime dependency.** numpy / scipy / pandas / formulaic / pydantic / pyarrow / pyyaml / typer only.
- **No MCMC.** Deterministic approximation only.
- **`CompiledLGM` invariants are preserved.** `design == hstack(blocks)`, `precision == block_diag(blocks)`. Off-block-diagonal precision coupling is out of scope.
- **No solver change beyond the `restrict` hook in Task 1.** `inference/gaussian.py` and the body of `inference/laplace.py` are otherwise untouched.
- **Existing tests pass unchanged at every commit.** `python -m pytest -q` from the repo root.
- **Ruff clean:** `ruff check src tests`, line length 100, rules `E4,E7,E9,F`.
- **Run tests as:** `PYTHONPATH=src python -m pytest ...`
- **Frozen dataclasses** with validation in `__post_init__` or a custom `__init__`, matching every existing spec type.
- **Errors** raise the existing exception types from `pylgm.exceptions` (`CompilationError`, `DataContractError`, `ModelValidationError`), never bare `Exception`.

## File Structure

| File | Responsibility |
|---|---|
| `src/pylgm/likelihoods.py` (modify) | Add `CompiledMixture`; add the `restrict` hook to `_CompiledLikelihood`. |
| `src/pylgm/inference/laplace.py` (modify, 2 lines) | Call `restrict(observed)` before binding aux. |
| `src/pylgm/joint.py` (create) | `Joint`, `Shared` specs; `_pad_block_rows`; `Joint.fit`. |
| `src/pylgm/compiler.py` (modify) | `compile_joint`, `compile_joint_family`, `build_joint_prediction_contexts`. |
| `src/pylgm/inference/prediction.py` (modify) | `column_slices` on `PredictionContext`; `_shared_block`; scatter in `_design_for`. |
| `src/pylgm/inference/result.py` (modify) | `predict(new_data, outcome=None)`. |
| `src/pylgm/__init__.py` (modify) | Export `Joint`, `Shared`. |
| `tests/test_mixture_likelihood.py` (create) | `CompiledMixture` unit tests. |
| `tests/test_joint_spec.py` (create) | `Joint` / `Shared` validation and rejections. |
| `tests/test_joint_compile.py` (create) | Reduction, factorisation, padding. |
| `tests/test_joint_fit.py` (create) | End-to-end fit, estimated `delta` recovery. |
| `tests/test_joint_predict.py` (create) | Per-outcome prediction. |
| `examples/shared_component/` (create) | KHB oral + larynx reference example. |

---

### Task 1: `CompiledMixture` — row-dispatching likelihood

**Files:**
- Modify: `src/pylgm/likelihoods.py` (add to end, before the spec classes at line 536)
- Modify: `src/pylgm/inference/laplace.py:28-30`
- Test: `tests/test_mixture_likelihood.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CompiledMixture(parts: tuple[tuple[np.ndarray, object], ...], n_rows: int)`, implementing the full `_CompiledLikelihood` interface. `_CompiledLikelihood.restrict(observed: np.ndarray) -> _CompiledLikelihood`, default `self`.

**Why `restrict` is needed:** `inference/laplace.py:29-30` binds aux by reaching for `.trials` and slicing it to observed rows. A mixture has no `.trials`, so it would be handed `for_observations(None)` and keep full-row masks while the engine evaluates on `y[observed]`. `restrict` re-indexes the masks into observed-row space. It defaults to `self`, so no existing likelihood changes behaviour.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mixture_likelihood.py`:

```python
import numpy as np
import pytest

from pylgm.exceptions import ModelValidationError
from pylgm.likelihoods import CompiledGaussian, CompiledMixture, CompiledPoisson


def _parts(n=6):
    mask_a = np.zeros(n, dtype=bool)
    mask_a[:3] = True
    mask_b = ~mask_a
    return ((mask_a, CompiledGaussian(1.5)), (mask_b, CompiledPoisson()))


def test_mixture_dispatches_each_method_per_row():
    n = 6
    parts = _parts(n)
    mixture = CompiledMixture(parts, n)
    eta = np.array([0.1, -0.2, 0.3, 0.4, 0.5, 0.6])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert mixture.log_likelihood(eta, y) == pytest.approx(
        parts[0][1].log_likelihood(eta[parts[0][0]], y[parts[0][0]])
        + parts[1][1].log_likelihood(eta[parts[1][0]], y[parts[1][0]])
    )
    for method in ("gradient", "working_weights", "third_derivative", "pointwise_log_density"):
        got = getattr(mixture, method)(eta, y)
        for mask, likelihood in parts:
            assert got[mask] == pytest.approx(getattr(likelihood, method)(eta[mask], y[mask]))


def test_mixture_rejects_overlapping_masks():
    n = 4
    overlap = np.array([True, True, True, False])
    other = np.array([False, True, True, True])
    with pytest.raises(ModelValidationError, match="disjoint"):
        CompiledMixture(((overlap, CompiledGaussian(1.0)), (other, CompiledPoisson())), n)


def test_mixture_rejects_uncovered_rows():
    n = 4
    partial = np.array([True, True, False, False])
    with pytest.raises(ModelValidationError, match="cover"):
        CompiledMixture(((partial, CompiledGaussian(1.0)),), n)


def test_mixture_validate_response_checks_each_part_on_its_own_rows():
    # A negative value on the Gaussian rows is fine; on the Poisson rows it is not.
    mixture = CompiledMixture(_parts(6), 6)
    mixture.validate_response(np.array([-1.0, 0.5, 2.0, 1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        mixture.validate_response(np.array([-1.0, 0.5, 2.0, -1.0, 2.0, 3.0]))


def test_restrict_reindexes_masks_into_observed_space():
    mixture = CompiledMixture(_parts(6), 6)
    observed = np.array([True, False, True, True, False, True])
    restricted = mixture.restrict(observed)
    assert restricted.n_rows == 4
    assert [mask.tolist() for mask, _ in restricted.parts] == [
        [True, True, False, False],
        [False, False, True, True],
    ]


def test_restrict_defaults_to_self_for_ordinary_likelihoods():
    likelihood = CompiledPoisson()
    assert likelihood.restrict(np.array([True, False])) is likelihood
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_mixture_likelihood.py -q`
Expected: FAIL with `ImportError: cannot import name 'CompiledMixture'`

- [ ] **Step 3: Add the `restrict` hook**

In `src/pylgm/likelihoods.py`, add to `_CompiledLikelihood` (after `for_observations`, line 46):

```python
    def restrict(self, observed: np.ndarray) -> "_CompiledLikelihood":
        """Re-index any row-indexed internal state into the observed-row subspace.

        Default is a no-op: ordinary likelihoods carry no row masks, and their
        per-row aux vectors are bound by ``for_observations`` instead. Only
        :class:`CompiledMixture` overrides this.
        """
        return self
```

- [ ] **Step 4: Implement `CompiledMixture`**

In `src/pylgm/likelihoods.py`, insert before `class Poisson:` (line 536):

```python
@dataclass(frozen=True, init=False)
class CompiledMixture(_CompiledLikelihood):
    """A likelihood that dispatches per row to one of several sub-likelihoods.

    ``parts`` pairs a boolean row mask with the likelihood governing those rows.
    The masks must be disjoint and together cover every row: an overlap would
    double-count an observation and a gap would silently drop one from the
    likelihood, both of which produce a plausible-looking wrong answer.

    Every method of the likelihood interface is row-separable, so this is a
    scatter over the parts and needs no change to the inference engines.
    """

    parts: tuple[tuple[np.ndarray, object], ...]
    n_rows: int

    def __init__(self, parts, n_rows: int) -> None:
        parts = tuple((np.asarray(mask, dtype=bool), lk) for mask, lk in parts)
        if not parts:
            raise ModelValidationError("mixture must have at least one part")
        n_rows = int(n_rows)
        for mask, _ in parts:
            if mask.ndim != 1 or mask.size != n_rows:
                raise ModelValidationError(
                    f"mixture masks must be 1-D boolean arrays of length {n_rows}"
                )
        counts = np.sum([mask.astype(np.int64) for mask, _ in parts], axis=0)
        if np.any(counts > 1):
            raise ModelValidationError("mixture masks must be disjoint")
        if np.any(counts < 1):
            raise ModelValidationError("mixture masks must cover every row")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "n_rows", n_rows)

    def restrict(self, observed: np.ndarray) -> "CompiledMixture":
        observed = np.asarray(observed, dtype=bool)
        return CompiledMixture(
            tuple((mask[observed], lk) for mask, lk in self.parts), int(observed.sum())
        )

    def for_observations(self, aux) -> "CompiledMixture":
        if aux is None:
            return self
        return CompiledMixture(
            tuple(
                (mask, lk.for_observations(_mask_aux(aux, mask))) for mask, lk in self.parts
            ),
            self.n_rows,
        )

    def _scatter(self, method: str, *arrays: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_rows, dtype=float)
        for mask, likelihood in self.parts:
            out[mask] = getattr(likelihood, method)(*(a[mask] for a in arrays))
        return out

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        return float(
            sum(lk.log_likelihood(eta[mask], y[mask]) for mask, lk in self.parts)
        )

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._scatter("gradient", eta, y)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._scatter("working_weights", eta, y)

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._scatter("third_derivative", eta, y)

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._scatter("pointwise_log_density", eta, y)

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._scatter("cdf", eta, y)

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return self._scatter("response_mean", eta)

    def response_prediction(
        self, eta_mean: np.ndarray, eta_variance: np.ndarray
    ) -> np.ndarray:
        return self._scatter("response_prediction", eta_mean, eta_variance)

    def validate_response(self, y: np.ndarray) -> None:
        for mask, likelihood in self.parts:
            likelihood.validate_response(y[mask])


def _mask_aux(aux, mask: np.ndarray):
    """Slice each aux vector down to one part's rows; ``None`` entries pass through."""
    if aux is None:
        return None
    sliced = {
        key: (None if value is None else np.asarray(value)[mask])
        for key, value in aux.items()
    }
    return sliced if any(v is not None for v in sliced.values()) else None
```

Add `ModelValidationError` to the imports at the top of `likelihoods.py` if not already present:

```python
from pylgm.exceptions import ModelValidationError
```

- [ ] **Step 5: Wire `restrict` into the Laplace engine**

In `src/pylgm/inference/laplace.py`, replace lines 28-30:

```python
    # Binomial carries a per-row trials vector; the fit loop works on the observed
    # rows, so bind their trials. For every other likelihood this returns self.
    _trials = getattr(likelihood, "trials", None)
    lk_obs = likelihood.for_observations({"trials": _trials[observed]} if _trials is not None else None)
```

with:

```python
    # Binomial carries a per-row trials vector; the fit loop works on the observed
    # rows, so bind their trials. For every other likelihood this returns self.
    # `restrict` re-indexes any row-indexed internal state (a mixture's masks)
    # into observed-row space first; it is a no-op for every other likelihood.
    likelihood = likelihood.restrict(observed)
    _trials = getattr(likelihood, "trials", None)
    lk_obs = likelihood.for_observations({"trials": _trials[observed]} if _trials is not None else None)
```

- [ ] **Step 6: Run the new tests and the full suite**

Run: `PYTHONPATH=src python -m pytest tests/test_mixture_likelihood.py -q`
Expected: PASS, 6 passed

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS, no new failures (the `restrict` default is a no-op)

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/likelihoods.py src/pylgm/inference/laplace.py tests/test_mixture_likelihood.py
git commit -m "feat(likelihoods): row-dispatching CompiledMixture

Every method of the likelihood interface is row-separable, so a mixture
over disjoint covering row masks needs no solver change. The one hook the
engine needed is restrict(observed), which re-indexes the masks into the
observed-row subspace; it defaults to self for every other likelihood.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `_pad_block_rows` — vertical zero-padding of a latent block

**Files:**
- Create: `src/pylgm/joint.py`
- Test: `tests/test_joint_compile.py`

**Interfaces:**
- Consumes: `LatentBlock` from `pylgm.ir.model`.
- Produces: `_pad_block_rows(block: LatentBlock, before: int, after: int) -> LatentBlock`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_joint_compile.py`:

```python
import numpy as np
from scipy.sparse import csr_matrix

from pylgm.ir.model import LatentBlock
from pylgm.joint import _pad_block_rows


def _block():
    design = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    precision = csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    constraints = np.array([[1.0, 1.0]])
    return LatentBlock("u", ("a", "b"), design, precision, constraints)


def test_pad_block_rows_zero_pads_design_and_preserves_everything_else():
    block = _block()
    padded = _pad_block_rows(block, before=2, after=1)

    assert padded.design.shape == (6, 2)
    assert np.allclose(padded.design.toarray()[:2], 0.0)
    assert np.allclose(padded.design.toarray()[2:5], block.design.toarray())
    assert np.allclose(padded.design.toarray()[5:], 0.0)

    assert padded.name == block.name
    assert padded.labels == block.labels
    assert np.allclose(padded.precision.toarray(), block.precision.toarray())
    assert np.allclose(padded.constraints, block.constraints)


def test_pad_block_rows_with_no_padding_is_an_identity_on_the_design():
    block = _block()
    padded = _pad_block_rows(block, before=0, after=0)
    assert np.allclose(padded.design.toarray(), block.design.toarray())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_joint_compile.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pylgm.joint'`

- [ ] **Step 3: Create `src/pylgm/joint.py` with the helper**

```python
"""Joint latent Gaussian models: several responses stacked into one CompiledLGM.

A joint model is an ordinary :class:`~pylgm.ir.model.CompiledLGM` with more
rows. Responses stack as ``y = (y^(1), ..., y^(K))``; sub-model ``k`` occupies a
contiguous row slice. Private latent blocks are zero-padded outside their slice,
shared blocks carry scaled rows in every slice they enter, and the likelihood
becomes a row-dispatching :class:`~pylgm.likelihoods.CompiledMixture`. Both IR
invariants -- ``design == hstack(blocks)`` and ``precision == block_diag(blocks)``
-- are preserved.
"""

from scipy.sparse import csr_matrix, vstack

from pylgm.ir.model import LatentBlock


def _pad_block_rows(block: LatentBlock, before: int, after: int) -> LatentBlock:
    """Zero-pad a block's design rows into the stacked row space.

    Precision, labels and constraints are row-independent and pass through
    untouched, so a Besag sum-to-zero still constrains exactly what it did.
    """
    if before == 0 and after == 0:
        return block
    width = block.design.shape[1]
    pieces = []
    if before:
        pieces.append(csr_matrix((before, width)))
    pieces.append(block.design)
    if after:
        pieces.append(csr_matrix((after, width)))
    return LatentBlock(
        block.name,
        block.labels,
        vstack(pieces, format="csr"),
        block.precision,
        block.constraints,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_joint_compile.py -q`
Expected: PASS, 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/joint.py tests/test_joint_compile.py
git commit -m "feat(joint): _pad_block_rows for stacking latent blocks

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `Joint` and `Shared` specs with validation

**Files:**
- Modify: `src/pylgm/joint.py`
- Test: `tests/test_joint_spec.py`

**Interfaces:**
- Consumes: `_pad_block_rows` (Task 2); `LGM` from `pylgm.model`; `Hyperparameter` from `pylgm.parameters`.
- Produces:
  - `Shared(effect, scale)` — frozen dataclass. `.scales_for(k: int) -> tuple[float | Hyperparameter, ...]` expands the scalar shorthand.
  - `Joint(submodels: tuple[LGM, ...], shared: tuple[Shared, ...] = ())` — frozen dataclass. `.outcomes -> tuple[str, ...]` returns the sub-model response names in order.

- [ ] **Step 1: Write the failing test**

Create `tests/test_joint_spec.py`:

```python
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, Poisson
from pylgm.joint import Joint, Shared
from pylgm.parameters import Hyperparameter


def _sub(response):
    return LGM(response=response, likelihood=Poisson(), predictor=Fixed("1"))


def test_joint_exposes_outcomes_in_declaration_order():
    joint = Joint([_sub("oral"), _sub("larynx")])
    assert joint.outcomes == ("oral", "larynx")


def test_joint_rejects_duplicate_response_names():
    with pytest.raises(ValueError, match="unique"):
        Joint([_sub("oral"), _sub("oral")])


def test_joint_requires_at_least_two_submodels():
    with pytest.raises(ValueError, match="at least two"):
        Joint([_sub("oral")])


def test_scalar_hyperparameter_scale_expands_to_delta_and_inverse():
    shared = Shared(IID("u", index="district"), scale=Hyperparameter("delta", initial=1.0))
    scales = shared.scales_for(2)
    assert scales[0].name == "delta"
    assert scales[1] == ("delta", "inverse")


def test_scalar_hyperparameter_scale_rejected_beyond_two_submodels():
    shared = Shared(IID("u", index="district"), scale=Hyperparameter("delta", initial=1.0))
    with pytest.raises(ValueError, match="exactly two"):
        shared.scales_for(3)


def test_explicit_tuple_scale_passes_through():
    shared = Shared(IID("u", index="district"), scale=(1.0, 2.0, 0.5))
    assert shared.scales_for(3) == (1.0, 2.0, 0.5)


def test_scale_tuple_length_must_match_submodel_count():
    shared = Shared(IID("u", index="district"), scale=(1.0, 2.0))
    with pytest.raises(ValueError, match="length"):
        shared.scales_for(3)


def test_float_scale_broadcasts_to_every_submodel():
    shared = Shared(IID("u", index="district"), scale=1.0)
    assert shared.scales_for(3) == (1.0, 1.0, 1.0)


def test_allow_ragged_defaults_to_false_and_must_be_a_bool():
    assert Shared(IID("u", index="district")).allow_ragged is False
    with pytest.raises(TypeError, match="allow_ragged"):
        Shared(IID("u", index="district"), allow_ragged="yes")


def test_mixed_likelihoods_are_allowed():
    joint = Joint([
        LGM(response="cd4", likelihood=Gaussian(sigma=1.0), predictor=Fixed("1")),
        LGM(response="event", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    assert joint.outcomes == ("cd4", "event")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_joint_spec.py -q`
Expected: FAIL with `ImportError: cannot import name 'Joint' from 'pylgm.joint'`

- [ ] **Step 3: Implement the specs**

Append to `src/pylgm/joint.py`:

```python
from dataclasses import dataclass, field

from pylgm.parameters import Hyperparameter

# A scaled shared field enters slice k as `scale_k * u`. The sentinel
# ("<name>", "inverse") means "the reciprocal of the hyperparameter <name>",
# which is how the Knorr-Held & Best (delta, delta^-1) pairing is carried
# through compilation without inventing an expression language.
InverseOf = tuple[str, str]


@dataclass(frozen=True)
class Shared:
    """One latent field entering several sub-models with a per-sub-model scaling.

    ``scale`` is a float (broadcast to every sub-model), a ``Hyperparameter``
    (shorthand for the Knorr-Held & Best ``(delta, delta^-1)`` pairing, and
    therefore valid only for exactly two sub-models), or an explicit
    per-sub-model tuple of floats and/or ``Hyperparameter``s.
    """

    effect: object
    scale: object = 1.0
    allow_ragged: bool = False
    """Accept a shared index whose level set differs between sub-models.

    The latent always spans the union of levels; this only silences the report.
    Off by default because an unintended mismatch weakens the shared field
    without any visible symptom.
    """

    def __post_init__(self) -> None:
        if not hasattr(self.effect, "name"):
            raise TypeError("Shared effect must be a latent effect spec")
        if not isinstance(self.allow_ragged, bool):
            raise TypeError("Shared allow_ragged must be a bool")
        scale = self.scale
        if isinstance(scale, (tuple, list)):
            entries = tuple(scale)
            if not entries:
                raise ValueError("Shared scale tuple must be non-empty")
            for entry in entries:
                if not isinstance(entry, (int, float, Hyperparameter)):
                    raise TypeError(
                        "Shared scale entries must be floats or Hyperparameters"
                    )
            object.__setattr__(self, "scale", entries)
        elif not isinstance(scale, (int, float, Hyperparameter)):
            raise TypeError("Shared scale must be a float, Hyperparameter, or tuple")

    @property
    def name(self) -> str:
        return self.effect.name

    def scales_for(self, count: int) -> tuple:
        """Expand ``scale`` to one entry per sub-model."""
        scale = self.scale
        if isinstance(scale, tuple):
            if len(scale) != count:
                raise ValueError(
                    f"Shared {self.name!r} scale tuple has length {len(scale)}, "
                    f"but the joint has {count} sub-models"
                )
            return scale
        if isinstance(scale, Hyperparameter):
            if count != 2:
                raise ValueError(
                    f"Shared {self.name!r} has a scalar Hyperparameter scale, which is "
                    "the (delta, delta^-1) shorthand and requires exactly two "
                    f"sub-models; this joint has {count}. Pass an explicit "
                    "per-sub-model tuple instead."
                )
            return (scale, (scale.name, "inverse"))
        return tuple(float(scale) for _ in range(count))


@dataclass(frozen=True)
class Joint:
    """Several `LGM` sub-models fitted as one stacked latent Gaussian model."""

    submodels: tuple = ()
    shared: tuple = ()

    def __init__(self, submodels, shared=()) -> None:
        submodels = tuple(submodels)
        if len(submodels) < 2:
            raise ValueError("Joint requires at least two sub-models")
        responses = [model.response for model in submodels]
        if len(responses) != len(set(responses)):
            raise ValueError("Joint sub-model response names must be unique")
        shared = tuple(shared)
        for entry in shared:
            if not isinstance(entry, Shared):
                raise TypeError("Joint shared entries must be Shared instances")
            entry.scales_for(len(submodels))
        shared_names = [entry.name for entry in shared]
        if len(shared_names) != len(set(shared_names)):
            raise ValueError("Joint shared effect names must be unique")
        object.__setattr__(self, "submodels", submodels)
        object.__setattr__(self, "shared", shared)

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(model.response for model in self.submodels)
```

Move the `from dataclasses import dataclass, field` and `from pylgm.parameters import Hyperparameter` imports up to the module header rather than leaving them mid-file, and drop `field` if unused.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_joint_spec.py -q`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/joint.py tests/test_joint_spec.py
git commit -m "feat(joint): Joint and Shared specs with scale expansion

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `compile_joint` with fixed scales

**Files:**
- Modify: `src/pylgm/compiler.py` (add after `compile_lgm`, line 551)
- Test: `tests/test_joint_compile.py` (append)

**Interfaces:**
- Consumes: `Joint`, `Shared`, `_pad_block_rows` (Tasks 2-3); `CompiledMixture` (Task 1); `compile_lgm`'s effect-builder dispatch.
- Produces: `compile_joint(joint: Joint, panels: dict[str, CanonicalPanel]) -> CompiledLGM`.

**Design notes for the implementer:**
- Extract the per-effect builder dispatch currently inlined in `compile_lgm` (`compiler.py:362-465`) into a module-level `_build_effect_block(effect, frame) -> LatentBlock` so `compile_joint` reuses it verbatim rather than duplicating a 100-line `if/elif` chain. `compile_lgm` then calls the same helper — behaviour must be identical, which the existing suite proves.
- Block names are namespaced by prefixing the **block name**, not the label: `LatentBlock(f"{outcome}:{block.name}", ...)`. `CompiledLGM` derives labels as `f"{block.name}:{label}"`.
- Block order is: all of sub-model 1's private blocks, then sub-model 2's, ..., then the shared blocks last. Fix this order and keep it; the prediction context asserts against it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_joint_compile.py`:

```python
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson
from pylgm.compiler import compile_joint, compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data.panel import CanonicalPanel
from pylgm.joint import Joint, Shared


def _frame():
    return pd.DataFrame({
        "district": ["a", "b", "c", "a", "b", "c"],
        "oral": [3.0, 5.0, 2.0, None, None, None],
        "larynx": [None, None, None, 4.0, 1.0, 6.0],
        "row": range(6),
    })


def _panel(frame, response):
    sub = frame[frame[response].notna()].reset_index(drop=True)
    return CanonicalPanel.from_frame(sub, DataConfig(time="row", response=response, panel=()))


def test_single_submodel_joint_matches_the_equivalent_lgm():
    # The strongest cheap guard: with one sub-model and no sharing, stacking is
    # the identity, so the compiled artefacts must agree exactly.
    frame = _frame()
    panel = _panel(frame, "oral")
    model = LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1") + IID("d", index="district"))

    expected = compile_lgm(model, panel)
    got = compile_joint(Joint._unchecked((model,), ()), {"oral": panel})

    assert got.design.shape == expected.design.shape
    assert (got.design != expected.design).nnz == 0
    assert (got.precision != expected.precision).nnz == 0
    assert got.labels == tuple(f"oral:{label}" for label in expected.labels)


def test_two_submodels_stack_rows_and_block_diagonalise_the_latent():
    frame = _frame()
    panels = {"oral": _panel(frame, "oral"), "larynx": _panel(frame, "larynx")}
    joint = Joint([
        LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
        LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    compiled = compile_joint(joint, panels)

    assert compiled.design.shape[0] == 6
    assert compiled.labels == ("oral:fixed:Intercept", "larynx:fixed:Intercept")
    dense = compiled.design.toarray()
    # Each sub-model's intercept column is zero outside its own row slice.
    assert dense[3:, 0].tolist() == [0.0, 0.0, 0.0]
    assert dense[:3, 1].tolist() == [0.0, 0.0, 0.0]


def test_shared_effect_with_fixed_scales_enters_both_slices():
    frame = _frame()
    panels = {"oral": _panel(frame, "oral"), "larynx": _panel(frame, "larynx")}
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district"), scale=(1.0, 2.0))],
    )
    compiled = compile_joint(joint, panels)

    shared = [b for b in compiled.blocks if b.name == "u"][0]
    dense = shared.design.toarray()
    assert dense.shape == (6, 3)
    assert dense[:3].sum() == pytest.approx(3.0)   # scale 1.0 on three oral rows
    assert dense[3:].sum() == pytest.approx(6.0)   # scale 2.0 on three larynx rows
```

Add an `_unchecked` classmethod to `Joint` for the reduction test, which deliberately builds a one-sub-model joint:

```python
    @classmethod
    def _unchecked(cls, submodels, shared=()):
        """Bypass the two-sub-model minimum. Test-only: used by the reduction test."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "submodels", tuple(submodels))
        object.__setattr__(obj, "shared", tuple(shared))
        return obj
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_joint_compile.py -q`
Expected: FAIL with `ImportError: cannot import name 'compile_joint'`

- [ ] **Step 3: Extract the effect-builder dispatch**

In `src/pylgm/compiler.py`, move the body of the `for effect in model.predictor.effects:` loop (lines 362-465) into:

```python
def _build_effect_block(effect, frame) -> LatentBlock:
    """Build one latent block from an effect spec. Shared by compile_lgm and compile_joint."""
```

returning `(block, precision_or_none)`. `compile_lgm` calls it in its loop; behaviour is unchanged, which the existing suite proves.

- [ ] **Step 4: Run the existing suite to prove the extraction is behaviour-preserving**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS, no new failures

- [ ] **Step 5: Implement `compile_joint`**

Append to `src/pylgm/compiler.py`:

```python
def compile_joint(joint, panels: "dict[str, CanonicalPanel]") -> CompiledLGM:
    """Compile a Joint into one stacked CompiledLGM.

    Row slices follow sub-model declaration order. Block order is every
    sub-model's private blocks in order, then the shared blocks -- fixed here
    because the prediction contexts assert against it.
    """
    outcomes = joint.outcomes
    frames = [panels[name].frame for name in outcomes]
    sizes = [len(frame) for frame in frames]
    starts, total = [], 0
    for size in sizes:
        starts.append(total)
        total += size

    blocks: list[LatentBlock] = []
    precisions: dict[str, float] = {}
    for position, (outcome, model, frame) in enumerate(zip(outcomes, joint.submodels, frames)):
        before, after = starts[position], total - starts[position] - sizes[position]
        for effect in model.predictor.effects:
            try:
                block, precision = _build_effect_block(effect, frame)
            except CompilationError as error:
                raise CompilationError(f"{error} for outcome {outcome!r}") from error
            named = LatentBlock(
                f"{outcome}:{block.name}", block.labels, block.design,
                block.precision, block.constraints,
            )
            blocks.append(_pad_block_rows(named, before, after))
            if precision is not None:
                precisions[f"{outcome}:{effect.name}"] = precision

    for entry in joint.shared:
        blocks.append(
            _shared_block(entry, joint, frames, starts, sizes, total, resolved={})
        )

    y = np.concatenate([
        frame[name].fillna(0.0).to_numpy(dtype=float)
        for name, frame in zip(outcomes, frames)
    ])
    observed = np.concatenate([panels[name].observed for name in outcomes])
    offset = np.concatenate([_offset_vector(model, frame) for model, frame in zip(joint.submodels, frames)])

    parts = []
    for position, (outcome, model, frame) in enumerate(zip(outcomes, joint.submodels, frames)):
        mask = np.zeros(total, dtype=bool)
        mask[starts[position] : starts[position] + sizes[position]] = True
        scalar = _estimable_scalar(model.likelihood)
        values = {scalar.name: scalar.initial} if scalar is not None else {}
        compiled = model.likelihood.materialize(values)
        aux = _likelihood_columns(model, frame)
        parts.append((mask, compiled.for_observations(aux)))
    likelihood = CompiledMixture(tuple(parts), total)

    width = sum(block.design.shape[1] for block in blocks)
    design = hstack([block.design for block in blocks], format="csr")
    precision = block_diag([block.precision for block in blocks], format="csr")
    constraints = _block_constraints(tuple(blocks), width)
    labels = _qualified_labels(blocks)
    try:
        return CompiledLGM(
            y=y, observed=observed, offset=offset, design=design, precision=precision,
            constraints=constraints, labels=labels, likelihood=likelihood,
            blocks=tuple(blocks),
        )
    except (TypeError, ValueError, ModelValidationError) as error:
        raise CompilationError(f"compiled joint model is invalid: {error}") from error
```

Implement the two helpers `compile_joint` needs:

```python
def _offset_vector(model, frame) -> np.ndarray:
    if model.offset is None:
        return np.zeros(len(frame))
    if model.offset not in frame.columns:
        raise DataContractError(f"offset column not found: {model.offset!r}")
    values = frame[model.offset].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise CompilationError(f"offset column {model.offset!r} must be finite")
    return values


def _shared_incidences(entry, frames, starts, sizes, total, outcomes=()):
    """Per-slice incidence matrices A_k of the shared index over the union of levels.

    The latent spans the union, so a level seen in only some sub-models still
    gets a latent entry -- it is simply informed by fewer rows. That is
    legitimate for genuinely ragged data and a data bug otherwise, so it is
    reported rather than silently absorbed.
    """
    index = entry.effect.index
    per_frame: list[set] = []
    levels: list[str] = []
    for frame in frames:
        if index not in frame.columns:
            raise CompilationError(
                f"shared effect {entry.name!r} indexes column {index!r}, "
                "which is missing from at least one sub-model's frame"
            )
        seen = set()
        for value in frame[index].astype(str):
            seen.add(value)
            if value not in levels:
                levels.append(value)
        per_frame.append(seen)

    union = set(levels)
    ragged = {
        (outcomes[i] if i < len(outcomes) else str(i)): sorted(union - seen)
        for i, seen in enumerate(per_frame)
        if union - seen
    }
    if ragged and not entry.allow_ragged:
        detail = "; ".join(
            f"{outcome} is missing {missing[:5]}{'...' if len(missing) > 5 else ''}"
            for outcome, missing in ragged.items()
        )
        raise CompilationError(
            f"shared effect {entry.name!r} has a ragged index {index!r}: {detail}. "
            "The latent spans the union of levels, so this is supported, but it is "
            "reported because an unintended mismatch silently weakens the shared "
            "field. Pass allow_ragged=True on the Shared to accept it."
        )
    position_of = {level: i for i, level in enumerate(levels)}
    incidences = []
    for start, size, frame in zip(starts, sizes, frames):
        rows = np.arange(start, start + size)
        cols = np.array([position_of[v] for v in frame[index].astype(str)])
        incidences.append(
            csr_matrix((np.ones(size), (rows, cols)), shape=(total, len(levels)))
        )
    return tuple(levels), incidences


def _resolve_scale(scale, resolved: "dict[str, float]") -> float:
    """Turn a scale entry into a number, given current hyperparameter values."""
    if isinstance(scale, Hyperparameter):
        return float(resolved.get(scale.name, scale.initial))
    if isinstance(scale, tuple):          # the ("<name>", "inverse") sentinel
        name, _ = scale
        return 1.0 / float(resolved.get(name, 1.0))
    return float(scale)


def _shared_block(entry, joint, frames, starts, sizes, total, resolved) -> LatentBlock:
    """Build the shared latent block: design = sum_k scale_k * A_k over the union index."""
    levels, incidences = _shared_incidences(
        entry, frames, starts, sizes, total, joint.outcomes
    )
    scales = entry.scales_for(len(joint.submodels))
    design = sum(
        _resolve_scale(scale, resolved) * incidence
        for scale, incidence in zip(scales, incidences)
    ).tocsr()
    template, _ = _build_effect_block(entry.effect, _levels_frame(entry.effect.index, levels))
    return LatentBlock(
        entry.name, template.labels, design, template.precision, template.constraints
    )


def _levels_frame(index: str, levels: "tuple[str, ...]") -> "pd.DataFrame":
    """A one-row-per-level frame, so the effect builder produces the union-index block."""
    return pd.DataFrame({index: list(levels)})
```

Add the imports `CompiledMixture` from `pylgm.likelihoods` and `_pad_block_rows`, `Joint`, `Shared` from `pylgm.joint` at the top of `compiler.py`.

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/test_joint_compile.py -q`
Expected: PASS, 5 passed

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS, no new failures

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/compiler.py src/pylgm/joint.py tests/test_joint_compile.py
git commit -m "feat(compiler): compile_joint stacks sub-models into one CompiledLGM

Extracts the per-effect builder dispatch from compile_lgm so compile_joint
reuses it verbatim. Private blocks are namespaced by block name and zero-padded
outside their row slice; shared blocks sum scaled per-slice incidences over the
union index. Both IR invariants hold.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `compile_joint_family` — estimated `delta`

**Files:**
- Modify: `src/pylgm/compiler.py`
- Test: `tests/test_joint_fit.py`

**Interfaces:**
- Consumes: `compile_joint` (Task 4); `ParametricDesignBlock`, `ScalableBlock` from `pylgm.ir.family`.
- Produces: `compile_joint_family(joint, panels) -> CompiledFamily | None`.

**Design note:** this mirrors the `MIDASParametric` registration at `compiler.py:698-735` exactly — build a template block at the initial scale, then wrap it in a `ParametricDesignBlock` whose `build(values)` re-forms the design. When no scale is a `Hyperparameter`, short-circuit to `ScalableBlock` as `MIDASParametric` does.

- [ ] **Step 1: Write the failing test**

First add the shared fixture to `tests/conftest.py` so Tasks 5, 6 and 8 all use one
generator — `tests/` has no `__init__.py`, so a cross-module test import would fail:

```python
@pytest.fixture
def shared_component_frame():
    """Simulated two-outcome shared-component data with a known delta.

    Returns ``(frame, true_delta)``. Both outcomes are Poisson over the same
    districts; `oral` carries `delta * u` and `larynx` carries `u / delta`.
    """
    def build(seed=0, n_districts=40, delta=1.6):
        rng = np.random.default_rng(seed)
        districts = [f"d{i}" for i in range(n_districts)]
        u = rng.normal(0.0, 0.7, size=n_districts)
        eta_oral = -0.3 + delta * u
        eta_larynx = 0.2 + u / delta
        frame = pd.DataFrame({
            "district": districts * 2,
            "outcome": ["oral"] * n_districts + ["larynx"] * n_districts,
            "oral": list(rng.poisson(np.exp(eta_oral)).astype(float)) + [np.nan] * n_districts,
            "larynx": [np.nan] * n_districts + list(rng.poisson(np.exp(eta_larynx)).astype(float)),
            "row": range(2 * n_districts),
        })
        return frame, delta

    return build
```

`tests/conftest.py` already imports `numpy as np` and `pandas as pd`; add them if not.

Create `tests/test_joint_fit.py`:

```python
import numpy as np
import pytest

from pylgm import Fixed, IID, LGM, Poisson
from pylgm.joint import Joint, Shared
from pylgm.parameters import Hyperparameter


def test_estimated_delta_is_recovered_from_simulated_shared_component_data(
    shared_component_frame,
):
    frame, true_delta = shared_component_frame()
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(
            IID("u", index="district", precision=Hyperparameter("tau_u", initial=1.0)),
            scale=Hyperparameter("delta", initial=1.0),
        )],
    )
    result = joint.fit(frame, engine="laplace")
    assert result.hyperparameters["delta"] == pytest.approx(true_delta, rel=0.5)


def test_fixed_scales_need_no_hyperparameter_and_still_fit(shared_component_frame):
    frame, _ = shared_component_frame()
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district", precision=1.0), scale=(1.0, 1.0))],
    )
    result = joint.fit(frame, engine="laplace")
    assert np.isfinite(result.log_marginal_likelihood)


def test_joint_rejects_the_exact_gaussian_engine(shared_component_frame):
    frame, _ = shared_component_frame()
    joint = Joint([
        LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
        LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    with pytest.raises(Exception, match="exact_gaussian"):
        joint.fit(frame, engine="exact_gaussian")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_joint_fit.py -q`
Expected: FAIL with `AttributeError: 'Joint' object has no attribute 'fit'`

- [ ] **Step 3: Implement `compile_joint_family`**

Append to `src/pylgm/compiler.py`:

```python
def compile_joint_family(joint, panels) -> "CompiledFamily | None":
    """Family form of compile_joint: rebuild scale-dependent designs per draw."""
    outcomes = joint.outcomes
    frames = [panels[name].frame for name in outcomes]
    sizes = [len(frame) for frame in frames]
    starts, total = [], 0
    for size in sizes:
        starts.append(total)
        total += size

    scalable: list = []
    parameter_names: list[str] = []
    parameter_bounds: dict[str, OptimizationBounds] = {}
    parameter_priors: dict[str, object] = {}

    # Reuse compile_family per sub-model rather than duplicating its 130-line
    # effect chain, then pad and rename what it produced. A sub-model with no
    # declared Hyperparameter returns None, in which case its blocks are plain
    # ScalableBlocks built from the compile_joint path.
    for position, (outcome, model, frame) in enumerate(zip(outcomes, joint.submodels, frames)):
        before, after = starts[position], total - starts[position] - sizes[position]
        sub_family = compile_family(model, panels[outcome])
        if sub_family is None:
            for effect in model.predictor.effects:
                block, _ = _build_effect_block(effect, frame)
                named = LatentBlock(
                    f"{outcome}:{block.name}", block.labels, block.design,
                    block.precision, block.constraints,
                )
                scalable.append(ScalableBlock(_pad_block_rows(named, before, after), None, 1.0))
            continue

        for item in sub_family.blocks:
            scalable.append(_restack_family_block(item, outcome, before, after))

        for name in sub_family.parameter_names:
            if name in parameter_names:
                raise CompilationError(
                    f"hyperparameter name {name!r} is declared by more than one "
                    "sub-model. Joint sub-models share one hyperparameter namespace, "
                    "so give each its own name (e.g. 'tau_oral', 'tau_larynx')."
                )
            parameter_names.append(name)
            if name in sub_family.parameter_bounds:
                parameter_bounds[name] = sub_family.parameter_bounds[name]
            if name in sub_family.parameter_priors:
                parameter_priors[name] = sub_family.parameter_priors[name]

    for entry in joint.shared:
        scales = entry.scales_for(len(joint.submodels))
        estimated = [s for s in scales if isinstance(s, Hyperparameter)]
        template = _shared_block(entry, joint, frames, starts, sizes, total, resolved={})
        if not estimated:
            scalable.append(ScalableBlock(template, None, 1.0))
            continue

        _, incidences = _shared_incidences(
            entry, frames, starts, sizes, total, joint.outcomes
        )

        def build(values, scales=scales, incidences=incidences):
            return sum(
                _resolve_scale(scale, values) * incidence
                for scale, incidence in zip(scales, incidences)
            ).tocsr()

        names = tuple(dict.fromkeys(s.name for s in estimated))
        scalable.append(ParametricDesignBlock(template, names, build))
        for hyper in estimated:
            if hyper.name in parameter_names:
                continue
            parameter_names.append(hyper.name)
            parameter_bounds[hyper.name] = _log_bounds(hyper)
            if hyper.prior is not None:
                parameter_priors[hyper.name] = hyper.prior

    if not parameter_names:
        return None

    y = np.concatenate([
        frame[name].fillna(0.0).to_numpy(dtype=float)
        for name, frame in zip(outcomes, frames)
    ])
    observed = np.concatenate([panels[name].observed for name in outcomes])
    offset = np.concatenate([
        _offset_vector(model, frame) for model, frame in zip(joint.submodels, frames)
    ])

    masks = []
    for position in range(len(outcomes)):
        mask = np.zeros(total, dtype=bool)
        mask[starts[position] : starts[position] + sizes[position]] = True
        masks.append(mask)

    def likelihood_factory(values, masks=masks, submodels=joint.submodels, frames=frames):
        """Rebuild the mixture at the current hyperparameter values.

        Each sub-model's estimable scalar (Gaussian sigma, NegBin/Gamma/Beta phi,
        Weibull alpha) is resolved from `values` if it is optimised, else left at
        its fixed value -- the same resolution compile_lgm does at its initial.
        """
        parts = []
        for mask, model, frame in zip(masks, submodels, frames):
            scalar = _estimable_scalar(model.likelihood)
            resolved = (
                {scalar.name: float(values[scalar.name])}
                if scalar is not None and scalar.name in values
                else ({scalar.name: scalar.initial} if scalar is not None else {})
            )
            compiled = model.likelihood.materialize(resolved)
            parts.append((mask, compiled.for_observations(_likelihood_columns(model, frame))))
        return CompiledMixture(tuple(parts), len(mask))

    return CompiledFamily(
        y=y,
        observed=observed,
        offset=offset,
        blocks=tuple(scalable),
        parameter_names=tuple(parameter_names),
        likelihood_factory=likelihood_factory,
        parameter_bounds=parameter_bounds,
        parameter_priors=parameter_priors,
    )
```

And the block-restacking helper the loop above uses:

```python
def _restack_family_block(item, outcome: str, before: int, after: int):
    """Rename and row-pad one family block from a sub-model's CompiledFamily.

    ScalableBlock and ParametricBlock vary only their *precision* with the
    hyperparameters, which is row-independent, so padding the template is
    enough. ParametricDesignBlock rebuilds a *design* over the sub-frame's rows,
    so its build output must be padded on every draw too.
    """
    inner = item.block
    named = LatentBlock(
        f"{outcome}:{inner.name}", inner.labels, inner.design,
        inner.precision, inner.constraints,
    )
    padded = _pad_block_rows(named, before, after)

    if isinstance(item, ParametricDesignBlock):
        def build(values, inner_build=item.build, before=before, after=after,
                  width=inner.design.shape[1]):
            design = inner_build(values)
            pieces = []
            if before:
                pieces.append(csr_matrix((before, width)))
            pieces.append(design)
            if after:
                pieces.append(csr_matrix((after, width)))
            return vstack(pieces, format="csr") if len(pieces) > 1 else design

        return ParametricDesignBlock(padded, item.parameters, build)

    if isinstance(item, ParametricBlock):
        return ParametricBlock(padded, item.parameters, item.build)

    return ScalableBlock(padded, item.parameter, item.scale)
```

**Deviation from the spec, deliberate.** The spec says hyperparameter names are
prefixed per outcome. Prefixing would require rewriting names *inside* the
closures `compile_family` already built, which capture `values[name]` by the
original name — fragile and easy to get subtly wrong. Rejecting duplicate names
with a clear message is a smaller change and a better failure mode. Update the
spec's "Label and hyperparameter namespacing" section to match when this lands.

`ScalableBlock`'s fields are `.block`, `.parameter`, `.scale`
(`ir/family.py:92-95`); `ParametricBlock` and `ParametricDesignBlock` both carry
`.block`, `.parameters`, `.build`. Verified against the current tree.

- [ ] **Step 4: Implement `Joint.fit`**

Append to `src/pylgm/joint.py`, mirroring `LGM._fit_pandas` (`model.py:458-493`):

```python
    def fit(self, frame, engine: str = "laplace", *, hyperparameters: str = "optimize",
            latent_strategy: str = "gaussian"):
        """Compile and fit this joint model. Only ``engine='laplace'`` is supported."""
        import pandas as pd

        from pylgm.compiler import compile_joint, compile_joint_family
        from pylgm.config.schema import DataConfig
        from pylgm.data.panel import CanonicalPanel
        from pylgm.exceptions import DataContractError, UnsupportedEngineError
        from pylgm.inference.laplace import fit_laplace

        if engine != "laplace":
            raise UnsupportedEngineError(
                "Joint models require engine='laplace'; the exact_gaussian engine "
                "needs a single CompiledGaussian likelihood, and a mixture is not one. "
                "Laplace is exact for an all-Gaussian stack anyway."
            )
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("frame must be a Pandas DataFrame")

        panels = {}
        for model in self.submodels:
            sub = frame[frame[model.response].notna()].reset_index(drop=True)
            time = model.time or "__pylgm_row__"
            if model.time is None:
                sub = sub.assign(**{time: range(len(sub))})
            panels[model.response] = CanonicalPanel.from_frame(
                sub, DataConfig(time=time, response=model.response, panel=model.panel)
            )

        family = compile_joint_family(self, panels)
        if hyperparameters == "integrate":
            if family is None:
                raise ValueError(
                    "hyperparameters='integrate' requires a declared Hyperparameter"
                )
            result = self._run_inla(family, latent_strategy)
        elif family is None:
            result = fit_laplace(compile_joint(self, panels))
        else:
            result = self._run_empirical_bayes(family)

        contexts = build_joint_prediction_contexts(
            self, panels, compile_joint(self, panels), result
        )
        return _rebuild_result(result, prediction_context=contexts)

    def _family_optimization_inputs(self, family):
        """Bounds, initial values and the prior penalty for this joint's hyperparameters.

        Mirrors ``LGM._family_optimization_inputs`` (model.py:405-426) but reads
        the declared Hyperparameters from every sub-model plus the shared scales,
        which is where a joint's parameters actually live.
        """
        from pylgm.compiler import _model_hyperparameters
        from pylgm.optimization.transforms import OptimizationBounds

        declared = []
        for model in self.submodels:
            declared.extend(hp for _, hp in _model_hyperparameters(model))
        for entry in self.shared:
            for scale in entry.scales_for(len(self.submodels)):
                if isinstance(scale, Hyperparameter):
                    declared.append(scale)

        bounds = (
            dict(family.parameter_bounds)
            if family.parameter_bounds
            else {hp.name: OptimizationBounds(hp.initial, hp.lower, hp.upper) for hp in declared}
        )
        initial = {hp.name: hp.initial for hp in declared if hp.name in family.parameter_names}
        family_priors = dict(getattr(family, "parameter_priors", {}) or {})
        priored = [hp for hp in declared if hp.prior is not None]
        penalty = None
        if family_priors or priored:
            def penalty(values, priored=priored):
                return sum(float(hp.prior.logpdf(values[hp.name])) for hp in priored)
        return bounds, initial, penalty

    def _run_empirical_bayes(self, family):
        """Type-II ML / MAP-II fit. Mirrors LGM._run_empirical_bayes (model.py:428)."""
        import warnings

        from pylgm.compiler import _parameters_at_bound
        from pylgm.inference.laplace import fit_laplace
        from pylgm.model import _attach_estimates
        from pylgm.optimization.empirical_bayes import optimize_empirical_bayes

        bounds, initial, penalty = self._family_optimization_inputs(family)
        eb = optimize_empirical_bayes(
            family, bounds, initial=initial, fit=fit_laplace, penalty=penalty
        )
        diagnostics = dict(eb.fit.diagnostics)
        diagnostics["empirical_bayes_converged"] = eb.diagnostics.converged
        diagnostics["empirical_bayes_evaluations"] = eb.diagnostics.evaluations
        diagnostics["hyperparameter_penalized"] = penalty is not None
        pinned = _parameters_at_bound(dict(eb.parameters), bounds)
        diagnostics["hyperparameters_at_bound"] = ", ".join(pinned)
        if pinned:
            warnings.warn(
                f"empirical-Bayes estimate(s) {list(pinned)} landed on the edge of "
                "the declared interval, so the bound rather than the data is "
                "setting the value. Widen lower/upper on those Hyperparameters "
                "and refit.",
                UserWarning,
                stacklevel=3,
            )
        return _attach_estimates(eb.fit, dict(eb.parameters), diagnostics)

    def _run_inla(self, family, latent_strategy: str = "gaussian"):
        """INLA grid integration. Mirrors LGM._run_inla (model.py:450)."""
        from pylgm.inference.laplace import fit_laplace
        from pylgm.optimization.inla import integrate_inla

        bounds, initial, penalty = self._family_optimization_inputs(family)
        return integrate_inla(
            family, bounds, initial=initial, fit=fit_laplace, penalty=penalty,
            latent_strategy=latent_strategy,
        )
```

`_model_hyperparameters`, `_parameters_at_bound` and `_attach_estimates` are
module-private today but already importable; no signature change is needed.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/test_joint_fit.py -q`
Expected: PASS, 3 passed

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS, no new failures

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/compiler.py src/pylgm/joint.py tests/test_joint_fit.py
git commit -m "feat(joint): estimated shared scaling via ParametricDesignBlock

Mirrors the MIDASParametric registration: a template block at the initial
scale wrapped in a ParametricDesignBlock that re-forms the design per draw,
short-circuiting to ScalableBlock when every scale is fixed.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Factorisation guard

**Files:**
- Test: `tests/test_joint_compile.py` (append)

**Interfaces:**
- Consumes: `Joint.fit` (Task 5); `LGM.fit`.
- Produces: nothing — this task is a correctness proof, not new code.

**Why this task exists separately:** with no shared effect the joint likelihood factorises, so the stacked posterior must equal the two separate posteriors. If it does not, the padding, masking, or offset concatenation is wrong. This catches those independently of the shared-component machinery, and a reviewer can reject it on its own.

- [ ] **Step 1: Write the test**

```python
def test_unshared_joint_factorises_into_the_separate_fits(shared_component_frame):
    frame, _ = shared_component_frame()
    joint = Joint([
        LGM(response="oral", likelihood=Poisson(),
            predictor=Fixed("1") + IID("d", index="district", precision=1.0)),
        LGM(response="larynx", likelihood=Poisson(),
            predictor=Fixed("1") + IID("d", index="district", precision=1.0)),
    ])
    together = joint.fit(frame, engine="laplace")

    separate = []
    for response in ("oral", "larynx"):
        sub = frame[frame[response].notna()].reset_index(drop=True)
        separate.append(
            LGM(response=response, likelihood=Poisson(),
                predictor=Fixed("1") + IID("d", index="district", precision=1.0)
                ).fit(sub, engine="laplace")
        )

    assert together.log_marginal_likelihood == pytest.approx(
        separate[0].log_marginal_likelihood + separate[1].log_marginal_likelihood, rel=1e-8
    )
    joint_oral = together.mean[[i for i, la in enumerate(together.labels) if la.startswith("oral:")]]
    assert joint_oral == pytest.approx(separate[0].mean, rel=1e-6, abs=1e-8)
```

- [ ] **Step 2: Run the test**

Run: `PYTHONPATH=src python -m pytest tests/test_joint_compile.py::test_unshared_joint_factorises_into_the_separate_fits -q`
Expected: PASS

If it fails, the bug is in Task 4's padding/masking/offset concatenation, not in this test. Fix there.

- [ ] **Step 3: Commit**

```bash
git add tests/test_joint_compile.py
git commit -m "test(joint): unshared joint factorises into the separate fits

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: `column_slices` — generalise `_design_for` with no behaviour change

**Files:**
- Modify: `src/pylgm/inference/prediction.py:41-57` and `:237-272`
- Test: existing `tests/test_predict.py`, `tests/inference/test_prediction.py` (guards)

**Interfaces:**
- Consumes: nothing.
- Produces: `PredictionContext.column_slices: tuple[tuple[int, int], ...] = ()`. Empty means "contiguous and gapless", i.e. today's behaviour.

**This task must not change any behaviour.** Its whole value is that the existing prediction suite passes untouched, proving the scatter is a generalisation rather than a special case.

- [ ] **Step 1: Add the field**

In `src/pylgm/inference/prediction.py`, add to `PredictionContext` after `width`:

```python
    column_slices: tuple[tuple[int, int], ...] = ()
    """Per-entry ``(start, stop)`` column spans in the fitted latent.

    Empty means the entries are contiguous and gapless from column 0, which is
    every single-response model. A joint model's per-outcome context sets this
    because its entries occupy scattered spans of the stacked latent.
    """
```

- [ ] **Step 2: Scatter instead of hstack in `_design_for`**

Replace the `design = np.hstack(blocks) if blocks else ...` line and the width check with:

```python
    if context.column_slices:
        if len(context.column_slices) != len(blocks):
            raise ValueError(
                "predict() context column_slices must align one-to-one with entries"
            )
        design = np.zeros((len(new_data), context.width))
        for block, (start, stop) in zip(blocks, context.column_slices):
            if stop - start != block.shape[1]:
                raise ValueError(
                    f"predict() rebuilt a block of width {block.shape[1]} for a "
                    f"column span of width {stop - start}"
                )
            design[:, start:stop] = block
    else:
        design = np.hstack(blocks) if blocks else np.empty((len(new_data), 0))
```

Keep both existing checks (row count, then total width) exactly as they are, after this block.

- [ ] **Step 3: Run the full prediction suite unchanged**

Run: `PYTHONPATH=src python -m pytest tests/test_predict.py tests/inference/test_prediction.py tests/test_spacetime_predict.py tests/test_midas_parametric_predict.py -q`
Expected: PASS, no failures — nothing sets `column_slices` yet, so every path takes the `else` branch.

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS, no new failures

- [ ] **Step 4: Commit**

```bash
git add src/pylgm/inference/prediction.py
git commit -m "refactor(prediction): scatter design blocks into explicit column spans

column_slices defaults to empty, meaning contiguous-and-gapless, so every
existing model takes the unchanged hstack path. Joint per-outcome contexts
will set it because their entries occupy scattered spans of the stacked latent.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Per-outcome prediction

**Files:**
- Modify: `src/pylgm/compiler.py` (add `build_joint_prediction_contexts`)
- Modify: `src/pylgm/inference/prediction.py` (add `_shared_block` entry kind)
- Modify: `src/pylgm/inference/result.py:755-780`
- Modify: `src/pylgm/joint.py` (attach contexts in `fit`)
- Test: `tests/test_joint_predict.py`

**Interfaces:**
- Consumes: `column_slices` (Task 7); `compile_joint` block order (Task 4).
- Produces:
  - `JointPredictionContext(contexts: dict[str, PredictionContext])` in `pylgm/inference/prediction.py`.
  - `_BaseResult.predict(new_data, outcome: str | None = None)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_joint_predict.py`:

```python
import numpy as np
import pytest

from pylgm import Fixed, IID, LGM, Poisson
from pylgm.joint import Joint, Shared
from pylgm.parameters import Hyperparameter


def _fitted(shared_component_frame):
    frame, _ = shared_component_frame()
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district", precision=1.0),
                       scale=Hyperparameter("delta", initial=1.0))],
    )
    return frame, joint.fit(frame, engine="laplace")


def test_predict_on_fit_rows_reproduces_the_fitted_means_per_outcome(shared_component_frame):
    frame, result = _fitted(shared_component_frame)
    oral_rows = frame[frame["oral"].notna()].reset_index(drop=True)
    prediction = result.predict(oral_rows, outcome="oral")
    fitted = result.predictive_mean[: len(oral_rows)]
    assert prediction.predictive_mean == pytest.approx(fitted, rel=1e-6, abs=1e-8)


def test_predict_requires_an_outcome_on_a_joint_result(shared_component_frame):
    frame, result = _fitted(shared_component_frame)
    oral_rows = frame[frame["oral"].notna()].reset_index(drop=True)
    with pytest.raises(ValueError, match="outcome"):
        result.predict(oral_rows)


def test_predict_rejects_an_unknown_outcome_and_lists_the_valid_ones(shared_component_frame):
    frame, result = _fitted(shared_component_frame)
    oral_rows = frame[frame["oral"].notna()].reset_index(drop=True)
    with pytest.raises(ValueError, match="oral"):
        result.predict(oral_rows, outcome="nonsense")


def test_predict_rejects_outcome_on_a_single_response_result(shared_component_frame):
    frame, _ = shared_component_frame()
    sub = frame[frame["oral"].notna()].reset_index(drop=True)
    result = LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")).fit(
        sub, engine="laplace"
    )
    with pytest.raises(ValueError, match="not a joint"):
        result.predict(sub, outcome="oral")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_joint_predict.py -q`
Expected: FAIL — `predict()` takes no `outcome` argument

- [ ] **Step 3: Add the shared prediction entry kind**

In `src/pylgm/inference/prediction.py`:

```python
def _shared_design_block(entry, new_data: pd.DataFrame) -> np.ndarray:
    """Rebuild a shared field's design for one outcome: scale_k * incidence.

    Named to distinguish it from ``compiler._shared_block``, which builds the
    fit-time LatentBlock; this one rebuilds the dense predict-time design.
    """
    name, index, labels, scale_spec, fitted = entry
    if index not in new_data.columns:
        raise ValueError(f"predict() new_data is missing the index column {index!r}")
    position_of = {label: i for i, label in enumerate(labels)}
    block = np.zeros((len(new_data), len(labels)))
    scale = fitted if isinstance(scale_spec, str) else float(scale_spec)
    for row, value in enumerate(new_data[index].astype(str)):
        if value not in position_of:
            raise ValueError(
                f"predict() new_data has an unseen level {value!r} in {index!r} "
                f"for shared effect {name!r}"
            )
        block[row, position_of[value]] = scale
    return block
```

Register `elif kind == "shared": blocks.append(_shared_design_block(payload, new_data))` in `_design_for`'s dispatch.

- [ ] **Step 4: Add `JointPredictionContext` and the `predict` argument**

In `src/pylgm/inference/prediction.py`:

```python
@dataclass(frozen=True)
class JointPredictionContext:
    """Per-outcome prediction contexts for a joint model."""

    contexts: "Mapping[str, PredictionContext]"

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(self.contexts)
```

In `src/pylgm/inference/result.py`, change `predict` (line 755):

```python
    def predict(self, new_data, outcome: str | None = None):
        """Score new rows against this result's latent posterior.

        ``outcome`` selects the sub-model on a joint result; it is required
        there and rejected on a single-response result.
        """
        from pylgm.inference.prediction import (
            JointPredictionContext, predict_from, predict_from_sparse,
        )

        context = self.prediction_context
        if isinstance(context, JointPredictionContext):
            if outcome is None:
                raise ValueError(
                    "predict() on a joint result requires outcome=, one of "
                    f"{context.outcomes}"
                )
            if outcome not in context.contexts:
                raise ValueError(
                    f"predict() got unknown outcome {outcome!r}; "
                    f"this joint model has {context.outcomes}"
                )
            context = context.contexts[outcome]
        elif outcome is not None:
            raise ValueError(
                "predict() got outcome=, but this result is not a joint model"
            )

        if self._covariance is None:
            sparse_posterior = getattr(self, "_sparse_posterior", None)
            if sparse_posterior is None:
                raise NotImplementedError(_NO_POSTERIOR)
            if context is None:
                raise ValueError(
                    "this result carries no prediction context; predict() is available "
                    "on results produced by LGM.fit"
                )
            return predict_from_sparse(context, self.mean, sparse_posterior, new_data)
        if context is None:
            raise ValueError(
                "this result carries no prediction context; predict() is available "
                "on results produced by LGM.fit"
            )
        return predict_from(context, self.mean, self.covariance, new_data)
```

The only change to the existing body is reading the local `context` instead of
`self.prediction_context`; both `None` guards and both branches are otherwise
untouched, so single-response prediction is bit-identical.

- [ ] **Step 5: Build the contexts in the compiler**

Add to `compiler.py`:

```python
def build_joint_prediction_contexts(joint, panels, compiled, result):
    """One PredictionContext per outcome, each spanning the full stacked latent.

    Block order is fixed by compile_joint: every sub-model's private blocks in
    declaration order, then the shared blocks. The column span of each block is
    read back off `compiled.blocks` so the two cannot drift apart silently.
    """
    spans, cursor = {}, 0
    for block in compiled.blocks:
        width = block.design.shape[1]
        spans[block.name] = (cursor, cursor + width)
        cursor += width

    fitted = dict(result.hyperparameters or {})
    contexts = {}
    for outcome, model in zip(joint.outcomes, joint.submodels):
        panel = panels[outcome]
        entries, slices, implied = [], [], []

        for effect in model.predictor.effects:
            block = next(
                b for b in compiled.blocks if b.name == f"{outcome}:{effect.name}"
            )
            entries.append(_prediction_entry(effect, model, panel, block))
            slices.append(spans[block.name])
            implied.extend(f"{block.name}:{label}" for label in block.labels)

        for entry in joint.shared:
            block = next(b for b in compiled.blocks if b.name == entry.name)
            scales = entry.scales_for(len(joint.submodels))
            scale = scales[joint.outcomes.index(outcome)]
            if isinstance(scale, Hyperparameter):
                spec, value = scale.name, float(fitted.get(scale.name, scale.initial))
            elif isinstance(scale, tuple):            # ("<name>", "inverse")
                name, _ = scale
                spec, value = name, 1.0 / float(fitted.get(name, 1.0))
            else:
                spec, value = float(scale), float(scale)
            entries.append(
                ("shared", (entry.name, entry.effect.index, block.labels, spec, value))
            )
            slices.append(spans[block.name])
            implied.extend(f"{block.name}:{label}" for label in block.labels)

        # Same guard build_prediction_context uses at compiler.py:1239: an entry
        # order that drifts from the compiled block order silently misaligns
        # every prediction, and nothing else would catch it.
        covered = {label for start, stop in slices for label in compiled.labels[start:stop]}
        if set(implied) != covered:
            raise CompilationError(
                f"joint prediction context for outcome {outcome!r} does not match "
                "the compiled block order; this would silently misalign predict()"
            )

        contexts[outcome] = PredictionContext(
            entries=tuple(entries),
            likelihood=_submodel_likelihood(model, panel, fitted),
            offset=model.offset,
            trials=model.likelihood.trials if isinstance(model.likelihood, Binomial) else None,
            width=compiled.design.shape[1],
            column_slices=tuple(slices),
        )
    return JointPredictionContext(contexts)


def _prediction_entry(effect, model, panel, block):
    """The predict-time descriptor for one effect. Same dispatch as
    build_prediction_context (compiler.py:1181-1226); extract that body into this
    helper and have both callers use it rather than duplicating the chain."""


def _submodel_likelihood(model, panel, fitted):
    """That sub-model's own compiled likelihood at the fitted scalar.

    Per-outcome prediction never uses the mixture: new_data is homogeneous, so
    response_prediction, trials and survival aux behave exactly as they do for a
    single-response model.
    """
    scalar = _estimable_scalar(model.likelihood)
    values = (
        {scalar.name: float(fitted[scalar.name])}
        if scalar is not None and scalar.name in fitted
        else ({scalar.name: scalar.initial} if scalar is not None else {})
    )
    return model.likelihood.materialize(values)
```

Extract the effect-to-entry dispatch from `build_prediction_context`
(`compiler.py:1181-1226`) into `_prediction_entry` and call it from both
places — the same de-duplication Task 4 does for `_build_effect_block`. The
existing prediction suite proves the extraction is behaviour-preserving.

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/test_joint_predict.py -q`
Expected: PASS, 4 passed

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS, no new failures

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/compiler.py src/pylgm/inference/prediction.py src/pylgm/inference/result.py src/pylgm/joint.py tests/test_joint_predict.py
git commit -m "feat(joint): per-outcome predict(new_data, outcome=)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Public exports and documentation

**Files:**
- Modify: `src/pylgm/__init__.py`
- Modify: `docs/effects.md`, `docs/roadmap.md`
- Modify: `tests/test_public_exports.py`
- Create: `docs/joint-models.md`
- Modify: `mkdocs.yml`

**Interfaces:**
- Consumes: `Joint`, `Shared` (Task 3).
- Produces: `pylgm.Joint`, `pylgm.Shared` importable from the package root.

- [ ] **Step 1: Add the failing export test**

In `tests/test_public_exports.py`, add `"Joint"` and `"Shared"` to the expected `__all__` list.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_public_exports.py -q`
Expected: FAIL — `Joint` not exported

- [ ] **Step 3: Export them**

In `src/pylgm/__init__.py` add `from pylgm.joint import Joint, Shared` and insert `"Joint"` and `"Shared"` into `__all__` in alphabetical position (after `"IID"` and after `"SAR"` respectively).

- [ ] **Step 4: Write `docs/joint-models.md`**

Cover: the stacking model and its row layout; the `Joint` / `Shared` API with the worked Knorr-Held & Best example; the `(delta, delta^-1)` shorthand and its two-sub-model restriction; `engine='laplace'` only; `predict(new_data, outcome=)`; and an explicit "not yet supported" list taken from the spec's Out of Scope section. Add it to `mkdocs.yml` nav after `spatial-effects.md`.

- [ ] **Step 5: Update the roadmap**

In `docs/roadmap.md`, add a **Joint models** bullet to "Shipped", and remove any "Next" item this supersedes.

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/__init__.py docs/ mkdocs.yml tests/test_public_exports.py
git commit -m "docs(joint): export Joint/Shared and document joint models

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Knorr-Held & Best reference example and R-INLA cross-check

**Files:**
- Create: `examples/shared_component/README.md`, `run.py`, `data.csv`, `graph.json`, `fetch_data.R`
- Create: `tests/integration/test_shared_component_reference.py`
- Modify: `docs/examples-shared-component.md`, `mkdocs.yml`

**Interfaces:**
- Consumes: everything above.
- Produces: the reference fit and its recorded tolerances.

**Data provenance:** oral cavity counts from `spam::Oral` (CRAN; 544 districts, 1986-1990, columns `Y`, `E`, `SMR`); larynx counts from `INLA::Germany` on the same districts and period; adjacency from the `spam` district structure. `fetch_data.R` regenerates `data.csv` and `graph.json` from those packages, and the README records the exact package versions used. `tests/test_package.py` runs `examples/*/run.py`, so `data.csv` must be committed — `fetch_data.R` is provenance, not a test-time dependency.

- [ ] **Step 1: Write `fetch_data.R` and generate the data**

```r
# Regenerates data.csv and graph.json. Not run by the test suite; the generated
# files are committed so the example runs with no R dependency.
library(spam)
library(INLA)
data(Oral)
data(Germany)
stopifnot(nrow(Oral) == 544, nrow(Germany) == 544)
out <- data.frame(
  district = seq_len(544),
  oral_Y = Oral$Y, oral_E = Oral$E,
  larynx_Y = Germany$Y, larynx_E = Germany$E,
  smoking = Germany$x
)
write.csv(out, "data.csv", row.names = FALSE)
# Adjacency: germany.graph ships with INLA as an R-INLA .graph file.
file.copy(system.file("demodata/germany.graph", package = "INLA"), "germany.graph")
```

Convert `germany.graph` to `graph.json` with the repo's existing `load_graph_file` helper, which already reads R-INLA `.graph` format.

- [ ] **Step 2: Write the failing reference test**

Create `tests/integration/test_shared_component_reference.py`:

```python
"""Knorr-Held & Best shared-component reference fit.

Reproduces the KHB *structure* on the public oral-cavity + larynx pair over the
544 German districts. The oesophageal counts KHB actually used as the second
outcome are not publicly available, so this validates the method against
R-INLA rather than reproducing the paper's posterior numbers.

Reference values in REFERENCE below come from fitting the same model in R-INLA;
regenerate with examples/shared_component/reference.R when the model changes.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, IID, Joint, LGM, Poisson, Shared
from pylgm.effects import load_graph_file
from pylgm.parameters import Hyperparameter

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "shared_component"

# Fitted in R-INLA; tolerances are deliberately loose because the two
# implementations differ in hyperparameter-grid placement, not in the model.
REFERENCE = {"delta": (1.0, 0.35)}   # (posterior mean, absolute tolerance)


@pytest.mark.slow
def test_shared_component_agrees_with_r_inla():
    frame = pd.read_csv(EXAMPLE / "data.csv")
    graph = load_graph_file(EXAMPLE / "graph.json")
    long = pd.concat([
        frame.assign(y=frame["oral_Y"], log_E=np.log(frame["oral_E"]), outcome="oral"),
        frame.assign(y=frame["larynx_Y"], log_E=np.log(frame["larynx_E"]), outcome="larynx"),
    ], ignore_index=True)
    long["oral"] = np.where(long["outcome"] == "oral", long["y"], np.nan)
    long["larynx"] = np.where(long["outcome"] == "larynx", long["y"], np.nan)

    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), offset="log_E",
             predictor=Fixed("1") + Besag("v_oral", index="district", graph=graph,
                                          precision=Hyperparameter("tau_v_oral", initial=1.0))),
         LGM(response="larynx", likelihood=Poisson(), offset="log_E",
             predictor=Fixed("1") + Besag("v_larynx", index="district", graph=graph,
                                          precision=Hyperparameter("tau_v_larynx", initial=1.0)))],
        shared=[Shared(
            Besag("u", index="district", graph=graph,
                  precision=Hyperparameter("tau_u", initial=1.0)),
            scale=Hyperparameter("delta", initial=1.0),
        )],
    )
    result = joint.fit(long, engine="laplace")

    assert np.isfinite(result.log_marginal_likelihood)
    mean, tolerance = REFERENCE["delta"]
    assert result.hyperparameters["delta"] == pytest.approx(mean, abs=tolerance)
```

Register the marker in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
markers = ["slow: reference fits on full-size data; deselect with -m 'not slow'"]
```

- [ ] **Step 3: Fit the model in R-INLA and record the reference values**

Write `examples/shared_component/reference.R` fitting the same model with `INLA::inla`, run it, and replace the placeholder in `REFERENCE` with the actual posterior mean. **Do not leave the placeholder `1.0` in the committed test** — that would assert nothing. If the R fit cannot be run, delete the `delta` assertion and keep only the finite-marginal-likelihood check, and say so in the test docstring.

- [ ] **Step 4: Write `run.py` and the README**

`run.py` fits the model, prints the posterior summary for `delta` and the shared component, and writes a map of the shared spatial component to `docs/img/shared_component.png`. The README states the data provenance, the deviation from KHB's oesophageal outcome, and how to regenerate.

- [ ] **Step 5: Run the example and the test**

Run: `PYTHONPATH=src python examples/shared_component/run.py`
Expected: completes, writes the figure

Run: `PYTHONPATH=src python -m pytest tests/integration/test_shared_component_reference.py -q`
Expected: PASS

Run: `PYTHONPATH=src python -m pytest -q -m "not slow"`
Expected: PASS, the reference fit deselected

- [ ] **Step 6: Commit**

```bash
git add examples/shared_component tests/integration/test_shared_component_reference.py docs/ mkdocs.yml pyproject.toml
git commit -m "test(joint): Knorr-Held & Best shared-component reference fit

Reproduces the KHB structure on the public oral-cavity + larynx pair over the
544 German districts, cross-checked against R-INLA. The oesophageal counts KHB
used are not public, so this validates the method rather than the paper's
numbers; the README records the deviation.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Notes for the implementer

**A pre-existing bug you will notice and should NOT fix here.** `inference/laplace.py` binds only `trials` to observed rows. `CompiledWeibullSurv` carries `event`/`entry` bound over *all* rows, so a survival model with any unobserved row misaligns its aux vectors against `y[observed]`. Task 1's `restrict` hook is the natural place to fix it, but doing so widens this slice's blast radius into survival. File it separately.

**Block order is load-bearing.** Task 4 fixes it as "every sub-model's private blocks in declaration order, then shared blocks". Task 8's `column_slices` computes spans from it. Changing the order in one place without the other produces silently misaligned predictions, which no type checker will catch — `build_joint_prediction_contexts` must assert its implied labels against `compiled.labels`, exactly as `build_prediction_context` does at `compiler.py:1239`.
