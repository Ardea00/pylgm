import numpy as np
import pandas as pd
import pytest

from pylgm.effects.fixed import build_fixed
from pylgm.exceptions import DataContractError, ModelValidationError
from pylgm.inference.laplace import fit_laplace
from pylgm.ir.model import CompiledLGM
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
    with pytest.raises(DataContractError):
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


def _mixture_model_with_unobserved_row() -> CompiledLGM:
    """A single-intercept model, one row unobserved, likelihood a CompiledMixture.

    Rows 0-1 are Gaussian, rows 2-4 are Poisson; row 4 is unobserved. This is
    the shape that trips Finding 1: `restrict` shrinks the mixture's masks to
    the 4 observed rows, so if the Laplace engine ever indexed those shrunk
    masks against the full 5-row `design`/`predictive_mean` arrays, `_scatter`
    would raise an IndexError (or silently misalign, were the sizes to match).
    """
    n = 5
    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0, 5.0, np.nan]})
    fixed_block = build_fixed(frame, "1", 1e-6)
    mask_gaussian = np.array([True, True, False, False, False])
    mask_poisson = ~mask_gaussian
    likelihood = CompiledMixture(
        ((mask_gaussian, CompiledGaussian(1.0)), (mask_poisson, CompiledPoisson())), n
    )
    return CompiledLGM(
        y=frame["y"].to_numpy(dtype=float),
        observed=np.array([True, True, True, True, False]),
        offset=np.zeros(n),
        design=fixed_block.design,
        precision=fixed_block.precision,
        constraints=fixed_block.constraints,
        labels=tuple(f"{fixed_block.name}:{label}" for label in fixed_block.labels),
        likelihood=likelihood,
        blocks=(fixed_block,),
    )


def test_fit_laplace_completes_for_mixture_with_unobserved_row():
    model = _mixture_model_with_unobserved_row()
    result = fit_laplace(model)
    assert result.predictive_mean.shape == (5,)
    assert np.isfinite(result.predictive_mean).all()


def test_fit_laplace_reports_mixture_link_name():
    model = _mixture_model_with_unobserved_row()
    result = fit_laplace(model)
    assert result.link_name == "mixture"
