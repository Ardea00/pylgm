"""An empirical-Bayes estimate pinned at its bound must not be silent."""

import warnings

import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, Hyperparameter, LGM, RW1
from pylgm.model import _parameters_at_bound
from pylgm.optimization.empirical_bayes import OptimizationBounds
from pylgm.optimization.transforms import IdentityTransform


def _series(n=30, seed=4):
    rng = np.random.default_rng(seed)
    level = np.cumsum(rng.standard_normal(n) * 0.1) + 2.0
    return pd.DataFrame({"t": range(n), "y": level + 0.25 * rng.standard_normal(n)})


def _fit(**bound_kwargs):
    model = LGM(
        response="y",
        predictor=Fixed("1") + RW1(
            "trend", "t",
            precision=Hyperparameter("trend.precision", initial=10.0, **bound_kwargs),
        ),
        likelihood=Gaussian(sigma=0.25),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(_series())
    pinned_warnings = [
        w for w in caught if "landed on the edge" in str(w.message)
    ]
    return result, pinned_warnings


def test_detects_a_bound_hit_on_the_transform_scale():
    """The optimizer stops near, not exactly on, the bound.

    9999.98 against an upper bound of 10000 is pinned for every practical
    purpose but is 0.02 away in natural units, so closeness is measured on the
    transform's own scale.
    """
    bounds = {"tau": OptimizationBounds(10.0, 0.01, 10_000.0)}
    assert _parameters_at_bound({"tau": 9999.98}, bounds) == ("tau",)
    assert _parameters_at_bound({"tau": 0.010001}, bounds) == ("tau",)
    assert _parameters_at_bound({"tau": 10.0}, bounds) == ()


def test_identity_transform_bounds_are_handled():
    bounds = {"g": OptimizationBounds(0.0, -2.0, 2.0, transform=IdentityTransform())}
    assert _parameters_at_bound({"g": 2.0}, bounds) == ("g",)
    assert _parameters_at_bound({"g": -2.0}, bounds) == ("g",)
    assert _parameters_at_bound({"g": 0.0}, bounds) == ()


def test_missing_and_degenerate_entries_are_skipped():
    bounds = {"a": OptimizationBounds(1.0, 0.5, 2.0)}
    assert _parameters_at_bound({}, bounds) == ()          # not estimated
    assert _parameters_at_bound({"b": 1.0}, bounds) == ()   # unrelated name


def test_pinned_estimate_is_reported_and_warned():
    result, pinned_warnings = _fit()  # default bounds: initial*1e-3 .. initial*1e3
    reported = result.diagnostics["hyperparameters_at_bound"]
    if reported:
        assert "trend.precision" in reported
        assert pinned_warnings, "a pinned estimate must warn, not only be recorded"
        assert "widen" in str(pinned_warnings[0].message).lower()
    else:
        assert not pinned_warnings


def test_widened_bounds_are_silent():
    result, pinned_warnings = _fit(lower=1e-6, upper=1e12)
    assert result.diagnostics["hyperparameters_at_bound"] == ""
    assert not pinned_warnings


def test_diagnostic_key_is_always_present_for_empirical_bayes():
    result, _ = _fit(lower=1e-6, upper=1e12)
    assert "hyperparameters_at_bound" in result.diagnostics
    # diagnostics values must stay immutable scalars
    assert isinstance(result.diagnostics["hyperparameters_at_bound"], str)


def test_fixed_hyperparameters_do_not_warn():
    """No estimation, so nothing can be pinned."""
    model = LGM(
        response="y",
        predictor=Fixed("1") + RW1("trend", "t", precision=2.0),
        likelihood=Gaussian(sigma=0.25),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(_series())
    assert not [w for w in caught if "landed on the edge" in str(w.message)]


def test_yaml_config_with_non_ascii_labels_loads(tmp_path):
    """Config files are read as UTF-8, not the platform's locale encoding.

    Region names with accents are ordinary in spatial models. Reading them
    with the default encoding raises UnicodeDecodeError on Windows (cp1252),
    which is how this was found.
    """
    from pylgm.config import load_model

    path = tmp_path / "model.yaml"
    path.write_text(
        "response: y\n"
        "likelihood:\n  family: gaussian\n  sigma: 1.0\n"
        "data:\n  panel: [region]\n"
        "predictor:\n"
        "  fixed: '1'\n"
        "  effects:\n"
        "    - {name: region, type: besag, index: region, precision: 1.0,\n"
        "       graph: {Zürich: [Genève], Genève: [Zürich]}}\n",
        encoding="utf-8",
    )
    model = load_model(path)
    assert model.response == "y"
    labels = dict(model.predictor.effects[1].graph)
    assert "Zürich" in labels and "Genève" in labels
